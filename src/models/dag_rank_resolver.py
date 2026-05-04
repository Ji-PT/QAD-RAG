"""
Parent-answer conditioned rank-level greedy retrieval for LogicRAG.

역할:
- 검증된 DAG 결과와 topological rank 결과를 입력으로 받는다.
- 같은 rank의 node들을 하나의 batch로 묶는다.
- 각 node의 parent answer를 수집한다.
- same-rank subproblems + parent answers로 unified query를 만든다.
- unified query로 한 번 검색한다.
- 검색 결과를 다시 node별 subproblem answer로 분해한다.
- node_id 기준으로 중간 답을 저장한다.
- rolling memory를 업데이트한다.
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


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _normalize_text_key(value: Any) -> str:
    return _clean_text(value).lower()


def build_node_text_by_id(dag_result: Dict[str, Any]) -> Dict[int, str]:
    """
    dag_result에서 node_id -> subproblem text mapping을 만든다.

    dag_result["input_node_ids"]와 dag_result["input_dependencies"]는
    verify_non_cyclicity.py가 같은 순서로 만들어준 값이다.
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

    즉:
        prerequisite_id가 parent
        dependent_id가 child
    """
    node_ids = [
        node_id
        for node_id in (_as_int(raw_id) for raw_id in dag_result.get("input_node_ids", []) or [])
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
    topological rank 결과를 Stage 5에서 쓰기 좋은 형태로 바꾼다.

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
    - subproblem text만 들고 있으면 parent answer를 찾기 어렵다.
    - 반드시 node_id를 같이 보존해야 한다.
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

    # fallback: rank_groups가 없으면 sorted_node_ids를 하나씩 rank로 취급한다.
    if not groups:
        sorted_node_ids = [
            node_id
            for node_id in (_as_int(raw_id) for raw_id in dag_result.get("sorted_node_ids", []) or [])
            if node_id is not None
        ]

        if not sorted_node_ids and sorted_dependencies:
            # 마지막 fallback. node_id 연결이 없으므로 parent-conditioned retrieval에는 약하지만
            # 최소한 기존 sorted_dependencies 흐름은 유지한다.
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

    반환:
        {
            child_node_id: [
                {
                    "parent_node_id": int,
                    "parent_subproblem": str,
                    "answer": str,
                    "is_answered": bool,
                    "evidence_summary": str
                },
                ...
            ],
            ...
        }
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
                "is_answered": bool(parent_result.get("is_answered", False)),
                "evidence_summary": _clean_text(parent_result.get("evidence_summary", "")),
            })

        parent_answers_by_node_id[node_id] = parent_answer_items

    return parent_answers_by_node_id


class ParentConditionedRankResolver:
    """
    LogicRAG Stage 5 담당 클래스.

    이 클래스는 LogicRAG 인스턴스를 받아서 다음 메서드만 사용한다.
    - rag._retrieve_for_query(...)
    - rag.filter_repeats / retrieved_chunks_set는 호출부에서 관리
    """

    def __init__(self, rag: Any):
        self.rag = rag

    @staticmethod
    def _rank_payload(
        nodes: List[Dict[str, Any]],
        parent_answers_by_node_id: Dict[int, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
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
    def _fallback_parent_conditioned_query(
        question: str,
        rank: int,
        nodes: List[Dict[str, Any]],
        parent_answers_by_node_id: Dict[int, List[Dict[str, Any]]],
    ) -> str:
        """
        LLM query merge가 실패했을 때 쓰는 deterministic fallback query.

        핵심:
        - parent answer가 있으면 반드시 query 문자열에 포함한다.
        - 그래야 parent-answer conditioned retrieval이 보존된다.
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
                lines.append("  Resolved parent answers:")
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
        같은 rank의 subproblem들을 하나의 retrieval query로 합친다.
        이때 각 subproblem의 resolved parent answer를 같이 넣는다.

        subproblem이 하나이고 parent answer도 없으면 LLM 호출 없이 그대로 반환한다.
        subproblem이 하나이어도 parent answer가 있으면 parent answer를 반영해야 하므로
        fallback 또는 LLM merge를 사용한다.
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
You are constructing ONE retrieval query for a topological rank in a Query Logic DAG.

Original question:
{question}

Topological rank:
{rank}

Same-rank subproblems with resolved parent answers:
{json.dumps(rank_payload, ensure_ascii=False, indent=2)}

Task:
Create ONE unified retrieval query that retrieves evidence for all same-rank subproblems.
The query must be conditioned on the resolved parent answers whenever they exist.

Rules:
- Use resolved parent answers as concrete anchors for their child subproblems.
- Preserve every entity, relation, date constraint, comparison target, and requested attribute.
- Do not answer the subproblems.
- Do not introduce new entities that are not in the subproblems or parent answers.
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

    def resolve_rank_context_by_nodes(
        self,
        question: str,
        rank: int,
        nodes: List[Dict[str, Any]],
        parent_answers_by_node_id: Dict[int, List[Dict[str, Any]]],
        unified_query: str,
        contexts: List[str],
        current_memory: str = "",
    ) -> Dict[str, Any]:
        """
        unified query로 검색한 context를 이용해 같은 rank의 node별 answer를 만든다.

        기존 decompose_unified_context_by_subproblem()와 다른 점:
        - subproblem text가 아니라 node_id 기준으로 answer를 저장한다.
        - parent answers를 prompt에 넣는다.
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

        context_text = "\n\n".join(contexts or [])

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

        rank_payload = self._rank_payload(
            nodes=nodes,
            parent_answers_by_node_id=parent_answers_by_node_id,
        )

        prompt = f"""
You are resolving same-rank subproblems in a Query Logic DAG.

Original question:
{question}

Current rolling memory before this rank:
{current_memory}

Topological rank:
{rank}

Same-rank subproblems with resolved parent answers:
{json.dumps(rank_payload, ensure_ascii=False, indent=2)}

Unified retrieval query:
{unified_query}

Retrieved context:
{context_text}

Task:
For each node, answer only that node's subproblem using:
1. the retrieved context,
2. the current rolling memory,
3. the resolved parent answers for that node.

Rules:
- Do not answer the final original question.
- Return one item for every node in the same order.
- Use node_id exactly as provided.
- Do not invent unsupported facts.
- If the evidence is insufficient, set is_answered to false and explain missing_info.
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

                # LLM이 node_id를 빼먹었을 경우 subproblem text로 보조 매핑한다.
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
                    "is_answered": bool(raw_answer.get("is_answered", False)),
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
            logger.error("Error resolving rank context by nodes: %s", e)

            return {
                "node_answers": fallback_answers,
                "rank_summary": "",
            }

    def refine_memory_with_rank_result(
        self,
        question: str,
        rank: int,
        nodes: List[Dict[str, Any]],
        parent_answers_by_node_id: Dict[int, List[Dict[str, Any]]],
        unified_query: str,
        contexts: List[str],
        rank_result: Dict[str, Any],
        current_memory: str = "",
    ) -> str:
        """
        논문의 rolling memory에 해당하는 정보 요약 업데이트.

        기존 memory + 이번 rank의 retrieved context + node별 answer를 합쳐
        다음 rank에서 쓸 compact memory를 만든다.
        """
        context_text = "\n\n".join(contexts or [])

        rank_payload = self._rank_payload(
            nodes=nodes,
            parent_answers_by_node_id=parent_answers_by_node_id,
        )

        rank_result_text = json.dumps(rank_result, ensure_ascii=False, indent=2)

        prompt = f"""
You are updating the rolling memory for LogicRAG.

Original question:
{question}

Previous rolling memory:
{current_memory}

Topological rank just processed:
{rank}

Same-rank subproblems with resolved parent answers:
{json.dumps(rank_payload, ensure_ascii=False, indent=2)}

Unified query:
{unified_query}

Retrieved context:
{context_text}

Resolved node answers:
{rank_result_text}

Task:
Write the updated rolling memory to support later ranks and final answer generation.

Rules:
- Preserve facts that help answer the original question or later dependent subproblems.
- Preserve node-level resolved answers and important parent-child links.
- Remove irrelevant or redundant details.
- Do not add unsupported claims.
- Be concise but keep exact names, dates, numbers, and relations.

Updated rolling memory:
"""

        try:
            memory = get_response_with_retry(prompt)
            memory = _clean_text(memory)
            if memory:
                return memory

        except Exception as e:
            logger.error("Error refining rolling memory with rank result: %s", e)

        # fallback: 최소한 node answer는 memory에 남긴다.
        fallback_parts = []

        if current_memory:
            fallback_parts.append(current_memory)

        fallback_parts.append(f"Rank {rank} unified query: {unified_query}")
        fallback_parts.append(f"Rank {rank} result: {rank_result_text}")

        return "\n\n".join(fallback_parts).strip()

    @staticmethod
    def build_final_subanswer_summary(
        dag_result: Dict[str, Any],
        resolved_answers_by_node_id: Dict[int, Dict[str, Any]],
    ) -> str:
        """
        최종 answer generation에 넣을 node별 중간 답 요약.
        topological order 기준으로 정렬한다.
        """
        sorted_node_ids = [
            node_id
            for node_id in (_as_int(raw_id) for raw_id in dag_result.get("sorted_node_ids", []) or [])
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
                "is_answered": bool(answer_item.get("is_answered", False)),
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
        Parent-answer conditioned rank-level greedy retrieval 실행.

        max_rounds:
            None이면 모든 rank를 처리한다.
            int이면 앞에서부터 해당 개수의 rank만 처리한다.
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

            parent_answers_by_node_id = collect_parent_answers_for_nodes(
                nodes=nodes,
                parent_ids_by_node_id=parent_ids_by_node_id,
                node_text_by_id=node_text_by_id,
                resolved_answers_by_node_id=resolved_answers_by_node_id,
            )

            unified_query = self.build_parent_conditioned_unified_query(
                question=question,
                rank=rank,
                nodes=nodes,
                parent_answers_by_node_id=parent_answers_by_node_id,
            )

            contexts = self.rag._retrieve_for_query(
                unified_query,
                retrieved_chunks_set=retrieved_chunks_set,
            )
            last_contexts = contexts

            rank_result = self.resolve_rank_context_by_nodes(
                question=question,
                rank=rank,
                nodes=nodes,
                parent_answers_by_node_id=parent_answers_by_node_id,
                unified_query=unified_query,
                contexts=contexts,
                current_memory=memory,
            )

            for node_answer in rank_result.get("node_answers", []) or []:
                node_id = _as_int(node_answer.get("node_id"))
                if node_id is None:
                    continue

                resolved_answers_by_node_id[node_id] = node_answer

            memory = self.refine_memory_with_rank_result(
                question=question,
                rank=rank,
                nodes=nodes,
                parent_answers_by_node_id=parent_answers_by_node_id,
                unified_query=unified_query,
                contexts=contexts,
                rank_result=rank_result,
                current_memory=memory,
            )

            retrieval_history.append({
                "round": round_idx,
                "rank": rank,
                "nodes": nodes,
                "subproblems": subproblems,
                "parent_answers_by_node_id": parent_answers_by_node_id,
                "unified_query": unified_query,
                "contexts": contexts,
                "rank_result": rank_result,
                "memory_after_rank": memory,
            })

        final_subanswer_summary = self.build_final_subanswer_summary(
            dag_result=dag_result,
            resolved_answers_by_node_id=resolved_answers_by_node_id,
        )

        if final_subanswer_summary and final_subanswer_summary != "[]":
            final_memory = (
                f"{memory}\n\n"
                f"Resolved subproblem answers in topological order:\n"
                f"{final_subanswer_summary}"
            ).strip()
        else:
            final_memory = memory

        return {
            "rank_groups": rank_groups,
            "processed_rank_groups": rank_groups_to_process,
            "resolved_answers_by_node_id": resolved_answers_by_node_id,
            "retrieval_history": retrieval_history,
            "last_contexts": last_contexts,
            "round_count": len(retrieval_history),
            "final_memory": final_memory,
            "final_subanswer_summary": final_subanswer_summary,
        }