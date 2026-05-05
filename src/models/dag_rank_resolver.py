"""
LogicRAG의 DAG rank 단위 retrieval / resolution 모듈.

논문 반영 범위:
- Eq. (1): parent-answer conditioned retrieval
- Eq. (2): subproblem answer generation
- Eq. (3): rolling memory 기반 context pruning
- Eq. (4): same-rank unified query 기반 graph pruning
- Algorithm 1의 rank 순차 처리 및 중간 답 저장

제외 범위:
- Algorithm 1의 새로운 unresolved subproblem 동적 추가는 다른 모듈에서 담당한다.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.utils.utils import get_response_with_retry, fix_json_response


logger = logging.getLogger(__name__)


def _as_int(value: Any) -> Optional[int]:
    """bool을 제외하고 int 변환 가능한 값만 int로 변환한다."""
    if isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    """LLM이 true/false를 문자열로 반환해도 안전하게 bool로 변환한다."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False

    return bool(value)


def _clean_text(value: Any) -> str:
    """문자열 값만 정리해서 반환한다."""
    if not isinstance(value, str):
        return ""
    return value.strip()


def _normalize_text_key(value: Any) -> str:
    """subproblem text matching용 정규화 key를 만든다."""
    return _clean_text(value).lower()


def build_node_text_by_id(dag_result: Dict[str, Any]) -> Dict[int, str]:
    """
    dag_result에서 node_id -> subproblem text mapping을 만든다.

    dag_result["input_node_ids"]와 dag_result["input_dependencies"]는
    verify_non_cyclicity.py가 같은 순서로 반환한 값이다.
    """
    node_ids = list(dag_result.get("input_node_ids", []) or [])
    dependencies = list(dag_result.get("input_dependencies", []) or [])

    if len(node_ids) != len(dependencies):
        raise ValueError(
            "Invalid dag_result: input_node_ids and input_dependencies length mismatch."
        )

    node_text_by_id: Dict[int, str] = {}

    for idx, raw_node_id in enumerate(node_ids):
        node_id = _as_int(raw_node_id)
        if node_id is None:
            continue

        text = _clean_text(dependencies[idx])
        if not text:
            continue

        node_text_by_id[node_id] = text

    return node_text_by_id


def build_parent_ids_by_node_id(dag_result: Dict[str, Any]) -> Dict[int, List[int]]:
    """
    valid_dependency_edges를 이용해 child node별 parent node 목록을 만든다.

    edge 방향:
        prerequisite_id -> dependent_id

    의미:
        prerequisite_id가 parent node
        dependent_id가 child node
    """
    node_ids = [
        node_id
        for node_id in (
            _as_int(raw_id)
            for raw_id in dag_result.get("input_node_ids", []) or []
        )
        if node_id is not None
    ]

    node_id_set = set(node_ids)

    parent_ids_by_node_id: Dict[int, List[int]] = {
        node_id: []
        for node_id in node_ids
    }

    seen_edges = set()

    for edge in dag_result.get("valid_dependency_edges", []) or []:
        if not isinstance(edge, dict):
            continue

        prerequisite_id = _as_int(edge.get("prerequisite_id"))
        dependent_id = _as_int(edge.get("dependent_id"))

        if prerequisite_id is None or dependent_id is None:
            continue

        if prerequisite_id not in node_id_set or dependent_id not in node_id_set:
            continue

        if prerequisite_id == dependent_id:
            continue

        edge_key = (prerequisite_id, dependent_id)
        if edge_key in seen_edges:
            continue

        seen_edges.add(edge_key)
        parent_ids_by_node_id[dependent_id].append(prerequisite_id)

    return {
        node_id: sorted(parent_ids)
        for node_id, parent_ids in parent_ids_by_node_id.items()
    }


def build_rank_groups_with_nodes(
    dag_result: Dict[str, Any],
    topological_rank_result: Dict[str, Any],
    sorted_dependencies: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    topological rank 결과를 rank-level resolver에서 쓰기 좋은 형태로 바꾼다.

    반환 형태:
        [
            {
                "rank": 0,
                "nodes": [
                    {"node_id": 0, "subproblem": "..."},
                    {"node_id": 2, "subproblem": "..."}
                ],
                "subproblems": ["...", "..."]
            },
            ...
        ]

    중요한 점:
    - parent answer를 찾으려면 subproblem text만으로는 부족하다.
    - 반드시 node_id를 함께 보존해야 한다.
    """
    node_text_by_id = build_node_text_by_id(dag_result)

    raw_rank_groups = {}
    if isinstance(topological_rank_result, dict):
        raw_rank_groups = topological_rank_result.get("rank_groups", {}) or {}

    groups: List[Dict[str, Any]] = []

    if isinstance(raw_rank_groups, dict) and raw_rank_groups:
        for raw_rank, raw_node_ids in raw_rank_groups.items():
            rank = _as_int(raw_rank)
            if rank is None:
                continue

            if not isinstance(raw_node_ids, list):
                continue

            nodes: List[Dict[str, Any]] = []
            seen_node_ids = set()

            for raw_node_id in raw_node_ids:
                node_id = _as_int(raw_node_id)
                if node_id is None:
                    continue

                if node_id in seen_node_ids:
                    continue

                subproblem = node_text_by_id.get(node_id, "")
                if not subproblem:
                    continue

                seen_node_ids.add(node_id)
                nodes.append({
                    "node_id": node_id,
                    "subproblem": subproblem,
                })

            nodes = sorted(nodes, key=lambda item: item["node_id"])

            if nodes:
                groups.append({
                    "rank": rank,
                    "nodes": nodes,
                    "subproblems": [node["subproblem"] for node in nodes],
                })

    # rank_groups가 없을 때의 fallback.
    # 정상 논문 baseline에서는 topological_rank_result["rank_groups"]가 있어야 한다.
    if not groups:
        sorted_node_ids = [
            node_id
            for node_id in (
                _as_int(raw_id)
                for raw_id in dag_result.get("sorted_node_ids", []) or []
            )
            if node_id is not None
        ]

        if not sorted_node_ids and sorted_dependencies:
            for rank, dependency in enumerate(sorted_dependencies):
                dependency = _clean_text(dependency)
                if not dependency:
                    continue

                groups.append({
                    "rank": rank,
                    "nodes": [
                        {
                            "node_id": rank,
                            "subproblem": dependency,
                        }
                    ],
                    "subproblems": [dependency],
                })

            return groups

        for rank, node_id in enumerate(sorted_node_ids):
            subproblem = node_text_by_id.get(node_id, "")
            if not subproblem:
                continue

            groups.append({
                "rank": rank,
                "nodes": [
                    {
                        "node_id": node_id,
                        "subproblem": subproblem,
                    }
                ],
                "subproblems": [subproblem],
            })

    return sorted(groups, key=lambda item: item["rank"])


def collect_parent_answers_for_nodes(
    nodes: List[Dict[str, Any]],
    parent_ids_by_node_id: Dict[int, List[int]],
    node_text_by_id: Dict[int, str],
    resolved_answers_by_node_id: Dict[int, Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    현재 rank에 있는 각 node에 대해 이미 해결된 parent answer들을 모은다.

    논문 Eq. (1) 대응:
        현재 subproblem retrieval은 parent node들의 이전 answer에 condition된다.

    여기서 parent answer는 answer 생성 prompt에 직접 넣기보다는
    retrieval query를 구체화하는 데 사용한다.
    """
    parent_answers_by_node_id: Dict[int, List[Dict[str, Any]]] = {}

    for node in nodes:
        node_id = _as_int(node.get("node_id"))
        if node_id is None:
            continue

        parent_answer_items: List[Dict[str, Any]] = []

        for parent_id in parent_ids_by_node_id.get(node_id, []):
            parent_result = resolved_answers_by_node_id.get(parent_id)
            if not isinstance(parent_result, dict):
                continue

            answer = _clean_text(parent_result.get("answer", ""))
            if not answer:
                continue

            parent_answer_items.append({
                "parent_node_id": parent_id,
                "parent_subproblem": node_text_by_id.get(parent_id, ""),
                "answer": answer,
                "is_answered": _as_bool(parent_result.get("is_answered", False)),
                "evidence_summary": _clean_text(parent_result.get("evidence_summary", "")),
            })

        parent_answers_by_node_id[node_id] = parent_answer_items

    return parent_answers_by_node_id


class ParentConditionedRankResolver:
    """
    LogicRAG Stage 5 담당 클래스.

    담당:
    - 같은 rank의 node들을 묶는다.
    - parent answer를 retrieval query 생성에 반영한다.
    - unified query로 한 번 retrieval한다.
    - retrieved context를 rolling memory로 요약한다.
    - rolling memory로 현재 rank의 node answer를 생성한다.
    - generated answer를 다음 rank용 rolling memory에 반영한다.
    """

    def __init__(self, rag: Any):
        self.rag = rag

    @staticmethod
    def _rank_payload(
        nodes: List[Dict[str, Any]],
        parent_answers_by_node_id: Dict[int, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        unified query 생성과 memory 요약 prompt에 넣을 rank payload를 만든다.
        """
        payload: List[Dict[str, Any]] = []

        for node in nodes:
            node_id = _as_int(node.get("node_id"))
            if node_id is None:
                continue

            payload.append({
                "node_id": node_id,
                "subproblem": _clean_text(node.get("subproblem", "")),
                "resolved_parent_answers": parent_answers_by_node_id.get(node_id, []),
            })

        return payload

    @staticmethod
    def _nodes_payload(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        answer 생성 prompt에 넣을 node 목록을 정리한다.

        parent answer는 retrieval query conditioning에 이미 사용되므로,
        answer prompt에는 node_id와 subproblem text만 넣는다.
        """
        payload: List[Dict[str, Any]] = []

        for node in nodes:
            node_id = _as_int(node.get("node_id"))
            subproblem = _clean_text(node.get("subproblem", ""))

            if node_id is None or not subproblem:
                continue

            payload.append({
                "node_id": node_id,
                "subproblem": subproblem,
            })

        return payload

    @staticmethod
    def _fallback_parent_conditioned_query(
        question: str,
        rank: int,
        nodes: List[Dict[str, Any]],
        parent_answers_by_node_id: Dict[int, List[Dict[str, Any]]],
    ) -> str:
        """
        LLM query merge가 실패했을 때 쓰는 deterministic fallback query.

        parent answer가 있으면 반드시 query 문자열에 포함한다.
        """
        payload = ParentConditionedRankResolver._rank_payload(
            nodes=nodes,
            parent_answers_by_node_id=parent_answers_by_node_id,
        )

        lines = [
            f"Original question: {question}",
            f"Topological rank: {rank}",
            "Retrieve evidence for the following same-rank subproblems.",
        ]

        for item in payload:
            lines.append(f"- Node {item['node_id']}: {item['subproblem']}")

            parent_answers = item.get("resolved_parent_answers", [])
            if parent_answers:
                lines.append("  Resolved parent answers for retrieval conditioning:")
                for parent in parent_answers:
                    lines.append(
                        f"  - Parent node {parent['parent_node_id']}: "
                        f"{parent['answer']} "
                        f"(parent subproblem: {parent['parent_subproblem']})"
                    )

        return "\n".join(lines)

    def build_parent_conditioned_unified_query(
        self,
        question: str,
        rank: int,
        nodes: List[Dict[str, Any]],
        parent_answers_by_node_id: Dict[int, List[Dict[str, Any]]],
    ) -> str:
        """
        논문 Eq. (1)과 Eq. (4)를 함께 구현한다.

        Eq. (4):
            같은 rank의 subproblem S(r)을 하나의 unified query로 merge한다.

        Eq. (1):
            각 subproblem의 retrieval은 parent answer에 condition된다.

        rank-level 구현:
            q(r) = Merge(S(r), 각 node의 resolved parent answers)
        """
        nodes = [
            {
                "node_id": _as_int(node.get("node_id")),
                "subproblem": _clean_text(node.get("subproblem", "")),
            }
            for node in nodes
        ]

        nodes = [
            node
            for node in nodes
            if node["node_id"] is not None and node["subproblem"]
        ]

        if not nodes:
            return question

        has_parent_answers = any(
            parent_answers_by_node_id.get(node["node_id"])
            for node in nodes
        )

        if len(nodes) == 1 and not has_parent_answers:
            return nodes[0]["subproblem"]

        fallback_query = self._fallback_parent_conditioned_query(
            question=question,
            rank=rank,
            nodes=nodes,
            parent_answers_by_node_id=parent_answers_by_node_id,
        )

        rank_payload = self._rank_payload(
            nodes=nodes,
            parent_answers_by_node_id=parent_answers_by_node_id,
        )

        prompt = f"""
You are constructing ONE retrieval query for a topological rank in LogicRAG.

Original question Q:
{question}

Topological rank r:
{rank}

Same-rank subproblems S(r) with resolved parent answers:
{json.dumps(rank_payload, ensure_ascii=False, indent=2)}

Task:
Construct a unified retrieval query q(r) for S(r).

Rules:
- Merge same-rank subproblems into one query.
- Use resolved parent answers as concrete anchors when they exist.
- Preserve every entity, relation, date constraint, comparison target, and requested attribute.
- Use parent answers only to make the retrieval query concrete.
- Do not answer the subproblems.
- Do not introduce entities not present in the subproblems or parent answers.
- Prefer a concise factoid-style retrieval query.
- Return ONLY a JSON object.

Output schema:
{{
  "unified_query": string
}}
"""

        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            parsed = fix_json_response(response)

            if isinstance(parsed, dict):
                unified_query = parsed.get("unified_query", "")
                if isinstance(unified_query, str) and unified_query.strip():
                    return unified_query.strip()

        except Exception as e:
            logger.error("Error generating parent-conditioned unified query: %s", e)

        return fallback_query

    def summarize_rank_context_to_memory(
        self,
        question: str,
        rank: int,
        nodes: List[Dict[str, Any]],
        parent_answers_by_node_id: Dict[int, List[Dict[str, Any]]],
        unified_query: str,
        contexts: List[str],
        previous_memory: str = "",
    ) -> str:
        """
        논문 Eq. (3)과 Algorithm 1의 rolling memory update를 구현한다.

        흐름:
            C(r) = R(q(r))
            Mem(r) = Summarize(Mem(r-1) ∪ C(r))

        이 memory는 현재 rank의 subproblem answer를 생성하는 데 사용된다.
        """
        context_text = "\n\n".join(contexts or [])

        rank_payload = self._rank_payload(
            nodes=nodes,
            parent_answers_by_node_id=parent_answers_by_node_id,
        )

        prompt = f"""
You are updating LogicRAG rolling memory.

Original question Q:
{question}

Previous rolling memory Mem(r-1):
{previous_memory}

Topological rank r:
{rank}

Same-rank subproblems S(r) with retrieval-conditioning parent answers:
{json.dumps(rank_payload, ensure_ascii=False, indent=2)}

Unified query q(r):
{unified_query}

Retrieved documents C(r):
{context_text}

Task:
Produce Mem(r) by summarizing Mem(r-1) and C(r) with respect to Q.

Rules:
- Keep only salient facts useful for resolving the current rank, later dependent subproblems, or the final answer.
- Preserve exact entity names, dates, numbers, relations, and comparison targets.
- Remove irrelevant, redundant, or noisy context.
- Do not answer the final original question.
- Do not invent unsupported facts.
- Do not copy full retrieved passages.
- Return ONLY a JSON object.

Output schema:
{{
  "memory": string
}}
"""

        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            parsed = fix_json_response(response)

            if isinstance(parsed, dict):
                memory = parsed.get("memory", "")
                if isinstance(memory, str) and memory.strip():
                    return memory.strip()

        except Exception as e:
            logger.error("Error summarizing rank context to rolling memory: %s", e)

        fallback_parts = []

        if previous_memory:
            fallback_parts.append(previous_memory)

        fallback_parts.append(f"Rank {rank} unified query: {unified_query}")
        fallback_parts.append(f"Rank {rank} retrieved context:\n{context_text}")

        return "\n\n".join(fallback_parts).strip()

    def resolve_rank_with_memory(
        self,
        question: str,
        rank: int,
        nodes: List[Dict[str, Any]],
        unified_query: str,
        memory: str,
    ) -> Dict[str, Any]:
        """
        논문 Algorithm 1의 subproblem resolution 단계를 구현한다.

        각 p_i ∈ S(r)에 대해:
            Mem(r)를 사용해 p_i의 중간 답 a_i를 생성한다.

        주의:
        - raw retrieved context를 직접 넣지 않는다.
        - parent answer를 answer prompt에 직접 넣지 않는다.
        - parent answer는 retrieval query conditioning 단계에서 이미 사용되었다.
        """
        nodes = [
            {
                "node_id": _as_int(node.get("node_id")),
                "subproblem": _clean_text(node.get("subproblem", "")),
            }
            for node in nodes
        ]

        nodes = [
            node
            for node in nodes
            if node["node_id"] is not None and node["subproblem"]
        ]

        fallback_answers = [
            {
                "node_id": node["node_id"],
                "subproblem": node["subproblem"],
                "answer": "",
                "is_answered": False,
                "evidence_summary": "",
                "missing_info": "No parsed answer was produced.",
            }
            for node in nodes
        ]

        if not nodes:
            return {
                "node_answers": [],
                "rank_summary": "",
            }

        nodes_payload = self._nodes_payload(nodes)

        prompt = f"""
You are resolving same-rank subproblems in LogicRAG.

Original question Q:
{question}

Topological rank r:
{rank}

Same-rank subproblems S(r):
{json.dumps(nodes_payload, ensure_ascii=False, indent=2)}

Unified retrieval query q(r):
{unified_query}

Current rolling memory Mem(r):
{memory}

Task:
For each node, answer only that node's subproblem using Mem(r).

Rules:
- Do not answer the final original question.
- Return one item for every node in the same order.
- Use node_id exactly as provided.
- Use only Mem(r).
- Do not use outside knowledge.
- Do not invent unsupported facts.
- If Mem(r) is insufficient, set is_answered to false and explain missing_info.
- Keep each answer concise.
- Return ONLY a JSON object.

Output schema:
{{
  "node_answers": [
    {{
      "node_id": integer,
      "subproblem": string,
      "answer": string,
      "is_answered": boolean,
      "evidence_summary": string,
      "missing_info": string
    }}
  ],
  "rank_summary": string
}}
"""

        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            parsed = fix_json_response(response)

            if not isinstance(parsed, dict):
                raise ValueError("Rank resolution response is not a dict.")

            raw_answers = parsed.get("node_answers", [])
            if not isinstance(raw_answers, list):
                raw_answers = []

            node_ids = {node["node_id"] for node in nodes}
            text_to_node_id = {
                _normalize_text_key(node["subproblem"]): node["node_id"]
                for node in nodes
            }

            answer_by_node_id: Dict[int, Dict[str, Any]] = {}

            for raw_answer in raw_answers:
                if not isinstance(raw_answer, dict):
                    continue

                node_id = _as_int(raw_answer.get("node_id"))

                if node_id is None:
                    node_id = text_to_node_id.get(
                        _normalize_text_key(raw_answer.get("subproblem", ""))
                    )

                if node_id not in node_ids:
                    continue

                original_subproblem = next(
                    node["subproblem"]
                    for node in nodes
                    if node["node_id"] == node_id
                )

                answer_by_node_id[node_id] = {
                    "node_id": node_id,
                    "subproblem": original_subproblem,
                    "answer": _clean_text(raw_answer.get("answer", "")),
                    "is_answered": _as_bool(raw_answer.get("is_answered", False)),
                    "evidence_summary": _clean_text(raw_answer.get("evidence_summary", "")),
                    "missing_info": _clean_text(raw_answer.get("missing_info", "")),
                }

            normalized_answers: List[Dict[str, Any]] = []

            for node in nodes:
                node_id = node["node_id"]
                answer = answer_by_node_id.get(node_id)

                if answer is None:
                    answer = {
                        "node_id": node_id,
                        "subproblem": node["subproblem"],
                        "answer": "",
                        "is_answered": False,
                        "evidence_summary": "",
                        "missing_info": "No answer mapped to this node.",
                    }

                normalized_answers.append(answer)

            rank_summary = parsed.get("rank_summary", "")
            if not isinstance(rank_summary, str):
                rank_summary = ""

            return {
                "node_answers": normalized_answers,
                "rank_summary": rank_summary.strip(),
            }

        except Exception as e:
            logger.error("Error resolving rank with rolling memory: %s", e)

            return {
                "node_answers": fallback_answers,
                "rank_summary": "",
            }

    def distill_rank_result_to_memory(
        self,
        question: str,
        rank: int,
        nodes: List[Dict[str, Any]],
        unified_query: str,
        contexts: List[str],
        memory_for_resolution: str,
        rank_result: Dict[str, Any],
    ) -> str:
        """
        Framework 본문의 context pruning 설명을 반영한다.

        논문 본문은 subproblem이 resolved된 뒤,
        그 retrieved context와 answer a_i를 LLM summarization으로 distill해
        rolling memory에 반영한다고 설명한다.

        이 함수는 현재 rank의 answer들을 다음 rank에서 쓸 memory에 반영한다.
        """
        context_text = "\n\n".join(contexts or [])
        nodes_payload = self._nodes_payload(nodes)
        rank_result_text = json.dumps(rank_result, ensure_ascii=False, indent=2)

        prompt = f"""
You are updating LogicRAG rolling memory after resolving a topological rank.

Original question Q:
{question}

Topological rank r:
{rank}

Same-rank subproblems S(r):
{json.dumps(nodes_payload, ensure_ascii=False, indent=2)}

Unified query q(r):
{unified_query}

Retrieved documents C(r):
{context_text}

Memory used for resolution Mem(r):
{memory_for_resolution}

Resolved intermediate answers for this rank:
{rank_result_text}

Task:
Distill the retrieved context and the generated intermediate answers into the rolling memory for subsequent ranks.

Rules:
- Preserve only salient facts needed for later dependent subproblems or final answer composition.
- Preserve exact entity names, dates, numbers, relations, and comparison targets.
- Preserve generated intermediate answers when they are supported by the memory/context.
- Remove irrelevant, redundant, or noisy details.
- Do not answer the final original question.
- Do not invent unsupported facts.
- Return ONLY a JSON object.

Output schema:
{{
  "memory": string
}}
"""

        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            parsed = fix_json_response(response)

            if isinstance(parsed, dict):
                memory = parsed.get("memory", "")
                if isinstance(memory, str) and memory.strip():
                    return memory.strip()

        except Exception as e:
            logger.error("Error distilling rank result to rolling memory: %s", e)

        fallback_parts = []

        if memory_for_resolution:
            fallback_parts.append(memory_for_resolution)

        fallback_parts.append(f"Rank {rank} result: {rank_result_text}")

        return "\n\n".join(fallback_parts).strip()

    @staticmethod
    def build_final_subanswer_summary(
        dag_result: Dict[str, Any],
        resolved_answers_by_node_id: Dict[int, Dict[str, Any]],
    ) -> str:
        """
        최종 answer composition에 넣을 node별 중간 답 요약을 만든다.

        논문 Algorithm 1:
            A = Compose({a_i})
        """
        sorted_node_ids = [
            node_id
            for node_id in (
                _as_int(raw_id)
                for raw_id in dag_result.get("sorted_node_ids", []) or []
            )
            if node_id is not None
        ]

        if not sorted_node_ids:
            sorted_node_ids = sorted(resolved_answers_by_node_id.keys())

        items: List[Dict[str, Any]] = []

        for node_id in sorted_node_ids:
            answer_item = resolved_answers_by_node_id.get(node_id)
            if not isinstance(answer_item, dict):
                continue

            items.append({
                "node_id": node_id,
                "subproblem": answer_item.get("subproblem", ""),
                "answer": answer_item.get("answer", ""),
                "is_answered": _as_bool(answer_item.get("is_answered", False)),
                "evidence_summary": answer_item.get("evidence_summary", ""),
                "missing_info": answer_item.get("missing_info", ""),
            })

        return json.dumps(items, ensure_ascii=False, indent=2)

    def run(
        self,
        question: str,
        dag_result: Dict[str, Any],
        topological_rank_result: Dict[str, Any],
        sorted_dependencies: Optional[List[str]] = None,
        initial_memory: str = "",
        retrieved_chunks_set: Optional[set] = None,
        max_rounds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Parent-answer conditioned rank-level LogicRAG resolution.

        논문 baseline:
        - initial_memory=""이면 Mem(0)=∅ 이다.
        - max_rounds=None이면 모든 topological rank를 처리한다.
        - 각 rank group은 한 번만 처리한다.
        - 따라서 rank-batch 단위 sampling without replacement가 된다.

        max_rounds는 기존 코드 호환을 위해 남겨둔다.
        논문 baseline에서는 logic_rag.py에서 max_rounds=None을 넘긴다.
        """
        if not dag_result.get("is_dag", False):
            raise ValueError("Parent-conditioned rank resolution requires a valid DAG.")

        node_text_by_id = build_node_text_by_id(dag_result)
        parent_ids_by_node_id = build_parent_ids_by_node_id(dag_result)

        rank_groups = build_rank_groups_with_nodes(
            dag_result=dag_result,
            topological_rank_result=topological_rank_result,
            sorted_dependencies=sorted_dependencies,
        )

        if max_rounds is None:
            rank_groups_to_process = list(rank_groups)
        else:
            max_rounds = max(0, int(max_rounds))
            rank_groups_to_process = list(rank_groups[:max_rounds])

        memory = initial_memory or ""
        resolved_answers_by_node_id: Dict[int, Dict[str, Any]] = {}
        retrieval_history: List[Dict[str, Any]] = []
        last_contexts: List[str] = []

        for round_idx, rank_group in enumerate(rank_groups_to_process, start=1):
            rank = int(rank_group["rank"])
            nodes = rank_group.get("nodes", []) or []
            subproblems = rank_group.get("subproblems", []) or [
                node.get("subproblem", "")
                for node in nodes
            ]

            memory_before_rank = memory

            # 1. 부모 node 답을 수집한다.
            #    이 값은 answer 생성용이 아니라 retrieval query conditioning용이다.
            parent_answers_by_node_id = collect_parent_answers_for_nodes(
                nodes=nodes,
                parent_ids_by_node_id=parent_ids_by_node_id,
                node_text_by_id=node_text_by_id,
                resolved_answers_by_node_id=resolved_answers_by_node_id,
            )

            # 2. 같은 rank의 subproblem을 하나의 unified query로 묶는다.
            #    이때 각 node의 parent answer를 query 생성에 반영한다.
            unified_query = self.build_parent_conditioned_unified_query(
                question=question,
                rank=rank,
                nodes=nodes,
                parent_answers_by_node_id=parent_answers_by_node_id,
            )

            # 3. unified query로 retrieval을 한 번 수행한다.
            contexts = self.rag._retrieve_for_query(
                unified_query,
                retrieved_chunks_set=retrieved_chunks_set,
            )
            last_contexts = contexts

            # 4. retrieved context를 이전 memory와 합쳐 현재 rank용 memory로 요약한다.
            memory_for_resolution = self.summarize_rank_context_to_memory(
                question=question,
                rank=rank,
                nodes=nodes,
                parent_answers_by_node_id=parent_answers_by_node_id,
                unified_query=unified_query,
                contexts=contexts,
                previous_memory=memory_before_rank,
            )

            # 5. 현재 rank의 각 subproblem answer는 rolling memory로 생성한다.
            rank_result = self.resolve_rank_with_memory(
                question=question,
                rank=rank,
                nodes=nodes,
                unified_query=unified_query,
                memory=memory_for_resolution,
            )

            # 6. node_id 기준으로 중간 답을 저장한다.
            #    다음 rank의 parent-answer conditioned retrieval과 final Compose({a_i})에 사용된다.
            for node_answer in rank_result.get("node_answers", []) or []:
                node_id = _as_int(node_answer.get("node_id"))
                if node_id is None:
                    continue

                resolved_answers_by_node_id[node_id] = node_answer

            # 7. Framework 본문 설명에 맞게 retrieved context와 generated answer를
            #    다음 rank용 rolling memory에 반영한다.
            memory_after_rank = self.distill_rank_result_to_memory(
                question=question,
                rank=rank,
                nodes=nodes,
                unified_query=unified_query,
                contexts=contexts,
                memory_for_resolution=memory_for_resolution,
                rank_result=rank_result,
            )

            memory = memory_after_rank

            retrieval_history.append({
                "round": round_idx,
                "rank": rank,
                "nodes": nodes,
                "subproblems": subproblems,
                "parent_answers_by_node_id": parent_answers_by_node_id,
                "unified_query": unified_query,
                "contexts": contexts,
                "memory_before_rank": memory_before_rank,
                "memory_for_resolution": memory_for_resolution,
                "memory_after_rank": memory_after_rank,
                "rank_result": rank_result,
            })

        final_subanswer_summary = self.build_final_subanswer_summary(
            dag_result=dag_result,
            resolved_answers_by_node_id=resolved_answers_by_node_id,
        )

        return {
            "rank_groups": rank_groups,
            "processed_rank_groups": rank_groups_to_process,
            "resolved_answers_by_node_id": resolved_answers_by_node_id,
            "retrieval_history": retrieval_history,
            "last_contexts": last_contexts,
            "round_count": len(retrieval_history),
            "final_memory": memory,
            "final_subanswer_summary": final_subanswer_summary,
        }