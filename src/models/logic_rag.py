import copy
import json
import logging
from typing import List, Dict, Tuple, Any, Optional

from src.models.base_rag import BaseRAG
from src.models.query_logic_dag import (
    QueryLogicDAGBuilder,
    QueryLogicDAG,
    SubproblemNode,
    DependencyEdge,
)
from src.models.verify_non_cyclicity import (
    verify_dag_and_topological_sort,
    DependencyGraphCycleError,
    build_partial_order_fallback,
)
from src.models.dag_topological_rank import (
    compute_topological_ranks_from_verification,
    attach_topological_ranks_to_dag_dict,
    TopologicalRankError,
)
from src.utils.utils import get_response_with_retry, fix_json_response
from colorama import Fore, Style, init

# Initialize colorama
init()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Few-shot 예시 상수 (논문 Section 3.2)
QUERY_DECOMPOSITION_FEW_SHOT_EXAMPLES = """
Example 1:
Question: "Who is the mayor of the capital of France?"
Subproblems:
[
  {"id": 0, "text": "What is the capital of France?"},
  {"id": 1, "text": "Who is the mayor of this capital city?"}
]

Example 2:
Question: "When was the director of 'Inception' born, and what award did the film win at the Oscars?"
Subproblems:
[
  {"id": 0, "text": "Who directed the film 'Inception'?"},
  {"id": 1, "text": "When was this director born?"},
  {"id": 2, "text": "What award did 'Inception' win at the Oscars?"}
]

Example 3:
Question: "What is the population of the country where the inventor of the telephone was born?"
Subproblems:
[
  {"id": 0, "text": "Who invented the telephone?"},
  {"id": 1, "text": "In which country was this inventor born?"},
  {"id": 2, "text": "What is the population of this country?"}
]

Example 4:
Question: "What is the tallest building in Tokyo?"
Subproblems:
[
  {"id": 0, "text": "What is the tallest building in Tokyo?"}
]
"""


class LogicRAG(BaseRAG):
    def __init__(
        self,
        corpus_path: str = None,
        cache_dir: str = "./cache",
        filter_repeats: bool = False
    ):
        """Initialize the LogicRAG system."""
        super().__init__(corpus_path, cache_dir)

        self.max_rounds = 3
        self.MODEL_NAME = "LogicRAG"
        self.filter_repeats = filter_repeats

        self.dag_builder = QueryLogicDAGBuilder()
        self.last_query_logic_dag = None

        self.max_dag_repair_attempts = 1
        self.dag_cycle_policy = "raise"

        # Dynamic DAG adaptation 안전장치.
        # 한 질문 처리 도중 새 subproblem을 몇 번까지 추가할 수 있는지의 상한.
        self.max_dynamic_adaptations = 3

    def set_max_rounds(self, max_rounds: int):
        self.max_rounds = max_rounds

    def refine_summary_with_context(self, question: str, new_contexts: List[str],
                                  current_summary: str = "") -> str:
        try:
            context_text = "\n".join(new_contexts)

            if not current_summary:
                prompt = f"""Please create a concise summary of the following information as it relates to answering this question:

Question: {question}

Information:
{context_text}

Your summary should:
1. Include all relevant facts that might help answer the question
2. Exclude irrelevant information
3. Be clear and concise
4. Preserve specific details, dates, numbers, and names that may be relevant

Summary:"""
            else:
                prompt = f"""Please refine the following information summary using newly retrieved information.

Question: {question}

Current summary:
{current_summary}

New information:
{context_text}

Your refined summary should:
1. Integrate new relevant facts with the existing summary
2. Remove redundancies
3. Remain concise while preserving all important information
4. Prioritize information that helps answer the question
5. Maintain specific details, dates, numbers, and names that may be relevant

Refined summary:"""

            return get_response_with_retry(prompt)

        except Exception as e:
            logger.error(f"{Fore.RED}Error generating/refining summary: {e}{Style.RESET_ALL}")
            if current_summary:
                return f"{current_summary}\n\nNew information:\n{context_text}"
            return context_text

    def warm_up_analysis(self, question: str, info_summary: str) -> Dict:
        """[Legacy] warm-up analysis (현재 흐름에서는 decompose_query()로 대체됨)."""
        try:
            prompt = f"""Question: {question}

Available Information:
{info_summary}

Based on the information provided, please analyze:
1. Can the question be answered completely with this information? (Yes/No)
2. What specific information is missing, if any?
3. What specific question should we ask to find the missing information?
4. Summarize our current understanding based on available information.
5. What are the key dependencies needed to answer this question?
6. Why is information missing? (max 20 words)

Please format your response as a JSON object with these keys:
- "can_answer": boolean
- "missing_info": string
- "subquery": string
- "current_understanding": string
- "dependencies": list of strings (key information dependencies)
- "missing_reason": string (brief explanation why info is missing, max 20 words)"""

            response = get_response_with_retry(prompt)
            response = response.strip()
            response = response.replace('```json', '').replace('```', '')

            result = fix_json_response(response)
            if result is None:
                return {
                    "can_answer": True, "missing_info": "", "subquery": question,
                    "current_understanding": "Failed to parse reflection response.",
                    "dependencies": ["Information relevant to the question"],
                    "missing_reason": "Parse error occurred"
                }

            required_fields = ["can_answer", "missing_info", "subquery", "current_understanding"]
            if not all(field in result for field in required_fields):
                logger.error(f"{Fore.RED}Missing required fields in response: {response}{Style.RESET_ALL}")
                raise ValueError("Missing required fields")

            if "dependencies" not in result:
                result["dependencies"] = ["Information relevant to the question"]
            if "missing_reason" not in result:
                result["missing_reason"] = "Additional context needed" if not result["can_answer"] else "No missing information"

            result["can_answer"] = bool(result["can_answer"])
            if not result["subquery"]:
                result["subquery"] = question

            return result

        except Exception as e:
            logger.error(f"{Fore.RED}Error in analyze_dependency_graph: {e}{Style.RESET_ALL}")
            return {
                "can_answer": True, "missing_info": "", "subquery": question,
                "current_understanding": f"Error during analysis: {str(e)}",
                "dependencies": ["Information relevant to the question"],
                "missing_reason": "Analysis error occurred"
            }

    # 논문 Section 3.2 - Query Decomposition Prompting
    # subproblem 분해 + Few-shot prompting을 하나의 Task로 합침
    def decompose_query(self, question: str) -> Dict:
        try:
            prompt = f"""You are an expert at decomposing complex questions into smaller, logically ordered subproblems.

Given a question, you must:
1. Decompose the question into a minimal set of subproblems. Each subproblem must have an "id" (integer, starting from 0) and a "text" (the subproblem question string).
2. If the question is simple (single-hop, no decomposition needed), output a single subproblem identical to the original question.

Here are some examples:
{QUERY_DECOMPOSITION_FEW_SHOT_EXAMPLES}

Now decompose the following question:
Question: "{question}"

Please format your response as a JSON object with these keys:
- "subproblems": list of objects, each with "id" (int) and "text" (string)
- "is_simple": boolean

Respond ONLY with the JSON object, no additional text."""

            response = get_response_with_retry(prompt)
            response = response.strip().replace('```json', '').replace('```', '')

            result = fix_json_response(response)

            if result is None:
                return {"subproblems": [{"id": 0, "text": question}], "is_simple": True}

            if "subproblems" not in result or not isinstance(result["subproblems"], list) or len(result["subproblems"]) == 0:
                result["subproblems"] = [{"id": 0, "text": question}]
            if "is_simple" not in result:
                result["is_simple"] = len(result["subproblems"]) <= 1

            return result

        except Exception as e:
            logger.error(f"{Fore.RED}Error in decompose_query: {e}{Style.RESET_ALL}")
            return {"subproblems": [{"id": 0, "text": question}], "is_simple": True}

    def dependency_aware_rag(self, question: str, info_summary: str, dependencies: List[str], idx: int) -> str:
        """[Legacy] dependency-aware analysis (현재 흐름에서는 사용되지 않음)."""
        try:
            prompt = f"""
            We pre-parsed the question into a list of dependencies, and the dependencies are sorted in a topological order, below is the question, the information summary, and the decomposed dependencies:

            Question: {question}

            Available Information:
            {info_summary}

            Decomposed dependencies:
            {dependencies}

            Current dependency to be answered:
            {dependencies[idx]}

            Please analyze the question and the information summary, and the decomposed dependencies, and answer the following questions:
            Please analyze:
            1. Can the question be answered completely with this information? (Yes/No)
            2. Summarize our current understanding based on available information.

            Please format your response as a JSON object with these keys:
            - "can_answer": boolean
            - "current_understanding": string
            """
            response = get_response_with_retry(prompt)
            return fix_json_response(response)
        except Exception as e:
            logger.error(f"{Fore.RED}Error in dependency_aware_rag: {e}{Style.RESET_ALL}")
            return {
                "can_answer": True,
                "current_understanding": f"Error during analysis: {str(e)}",
            }

    @staticmethod
    def _deduplicate_nonempty_strings(items: List[Any]) -> List[str]:
        cleaned: List[str] = []
        seen = set()

        for item in items or []:
            if not isinstance(item, str):
                continue

            text = item.strip()
            if not text:
                continue

            key = text.lower()
            if key in seen:
                continue

            seen.add(key)
            cleaned.append(text)

        return cleaned

    def build_unified_query(
        self,
        question: str,
        rank: int,
        subproblems: List[str],
    ) -> str:
        """같은 topological rank에 속한 여러 subquery를 하나의 unified retrieval query로 합침."""
        subproblems = self._deduplicate_nonempty_strings(subproblems)

        if not subproblems:
            return question

        if len(subproblems) == 1:
            return subproblems[0]

        fallback_query = (
            f"For the original question '{question}', retrieve the facts needed to answer: "
            + "; ".join(subproblems)
        )

        prompt = f"""
You are merging same-rank subproblems in a Query Logic DAG into one retrieval query.

Original question:
{question}

Topological rank:
{rank}

Same-rank subproblems:
{json.dumps(subproblems, ensure_ascii=False, indent=2)}

Task:
Create ONE unified retrieval query that can retrieve evidence for all same-rank subproblems at once.

Rules:
- Preserve every entity, relation, date constraint, comparison target, and attribute requested.
- Do not answer the subproblems.
- Do not introduce new entities.
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
            logger.error(f"{Fore.RED}Error generating unified query: {e}{Style.RESET_ALL}")

        return fallback_query

    def decompose_unified_context_by_subproblem(
        self,
        question: str,
        rank: int,
        subproblems: List[str],
        unified_query: str,
        contexts: List[str],
        current_summary: str = "",
    ) -> Dict[str, Any]:
        """unified query로 가져온 retrieval context를 다시 개별 subproblem별 answer로 분해."""
        subproblems = self._deduplicate_nonempty_strings(subproblems)
        context_text = "\n\n".join(contexts or [])

        fallback_answers = [
            {
                "subproblem": subproblem,
                "answer": "",
                "is_answered": False,
                "evidence_summary": "",
                "missing_info": "No parsed answer was produced.",
            }
            for subproblem in subproblems
        ]

        prompt = f"""
You are decomposing a unified retrieval result back into answers for individual same-rank subproblems.

Original question:
{question}

Current information summary before this rank:
{current_summary}

Topological rank:
{rank}

Same-rank subproblems:
{json.dumps(subproblems, ensure_ascii=False, indent=2)}

Unified retrieval query:
{unified_query}

Retrieved context for the unified query:
{context_text}

Task:
For each same-rank subproblem, extract the relevant answer from the retrieved context.

Rules:
- Use only the retrieved context and the current summary.
- Do not invent unsupported facts.
- Keep each answer concise.
- If the evidence is insufficient, set is_answered to false and explain missing_info.
- Return one item for every subproblem in the same order.
- Return ONLY a JSON object.

Output schema:
{{
  "subproblem_answers": [
    {{
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
                raise ValueError("Unified decomposition response is not a dict.")

            raw_answers = parsed.get("subproblem_answers", [])
            if not isinstance(raw_answers, list):
                raw_answers = []

            normalized_by_subproblem: Dict[str, Dict[str, Any]] = {}
            for raw_answer in raw_answers:
                if not isinstance(raw_answer, dict):
                    continue

                subproblem = raw_answer.get("subproblem", "")
                if not isinstance(subproblem, str):
                    continue

                key = subproblem.strip().lower()
                if not key:
                    continue

                normalized_by_subproblem[key] = {
                    "subproblem": subproblem.strip(),
                    "answer": str(raw_answer.get("answer", "") or "").strip(),
                    "is_answered": bool(raw_answer.get("is_answered", False)),
                    "evidence_summary": str(raw_answer.get("evidence_summary", "") or "").strip(),
                    "missing_info": str(raw_answer.get("missing_info", "") or "").strip(),
                }

            normalized_answers: List[Dict[str, Any]] = []
            for subproblem in subproblems:
                key = subproblem.strip().lower()
                answer = normalized_by_subproblem.get(key)
                if answer is None:
                    answer = {
                        "subproblem": subproblem,
                        "answer": "",
                        "is_answered": False,
                        "evidence_summary": "",
                        "missing_info": "No answer mapped to this subproblem.",
                    }
                else:
                    answer["subproblem"] = subproblem
                normalized_answers.append(answer)

            rank_summary = parsed.get("rank_summary", "")
            if not isinstance(rank_summary, str):
                rank_summary = ""

            return {
                "subproblem_answers": normalized_answers,
                "rank_summary": rank_summary.strip(),
            }

        except Exception as e:
            logger.error(f"{Fore.RED}Error decomposing unified context: {e}{Style.RESET_ALL}")
            return {
                "subproblem_answers": fallback_answers,
                "rank_summary": "",
            }

    def refine_summary_with_unified_rank_result(
        self,
        question: str,
        rank: int,
        subproblems: List[str],
        unified_query: str,
        contexts: List[str],
        decomposed_result: Dict[str, Any],
        current_summary: str = "",
    ) -> str:
        """rank 단위 retrieval 결과와 subproblem별 분해 결과를 기존 info_summary에 병합."""
        try:
            context_text = "\n\n".join(contexts or [])
            decomposed_text = json.dumps(decomposed_result, ensure_ascii=False, indent=2)

            prompt = f"""
Please refine the information summary using the latest rank-level unified retrieval result.

Original question:
{question}

Current summary:
{current_summary}

Topological rank:
{rank}

Same-rank subproblems:
{json.dumps(subproblems, ensure_ascii=False, indent=2)}

Unified query:
{unified_query}

Retrieved context:
{context_text}

Decomposed subproblem answers:
{decomposed_text}

Your refined summary should:
1. Integrate newly supported facts with the existing summary.
2. Preserve specific names, dates, numbers, and relations.
3. Keep separate facts for different subproblems clear.
4. Avoid unsupported claims.
5. Remain concise.

Refined summary:
"""
            return get_response_with_retry(prompt)

        except Exception as e:
            logger.error(f"{Fore.RED}Error refining unified rank summary: {e}{Style.RESET_ALL}")
            fallback = current_summary or ""
            return (
                f"{fallback}\n\n"
                f"Rank {rank} unified query: {unified_query}\n"
                f"Subproblem answers: {json.dumps(decomposed_result, ensure_ascii=False)}"
            ).strip()

    def rank_aware_rag(
        self,
        question: str,
        info_summary: str,
        rank: int,
        subproblems: List[str],
        unified_query: str,
        decomposed_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        [Legacy / Ablation] Rank별 can_answer 판정.

        논문 Algorithm 1에는 매 rank 후 종료 분기가 없으므로
        현재 main 흐름에서는 호출되지 않음.
        함수 정의는 ablation 비교용으로 보존한다.
        """
        try:
            prompt = f"""
We are performing rank-level retrieval over a Query Logic DAG.

Original question:
{question}

Available information summary:
{info_summary}

Current topological rank:
{rank}

Same-rank subproblems just processed:
{json.dumps(subproblems, ensure_ascii=False, indent=2)}

Unified query used for this rank:
{unified_query}

Decomposed result from this rank:
{json.dumps(decomposed_result, ensure_ascii=False, indent=2)}

Please analyze:
1. Can the original question now be answered completely? (Yes/No)
2. Summarize the current understanding.
3. State what information is still missing, if any.

Return ONLY a JSON object with these keys:
- "can_answer": boolean
- "current_understanding": string
- "missing_info": string
"""
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            parsed = fix_json_response(response)

            if not isinstance(parsed, dict):
                raise ValueError("Rank-aware analysis response is not a dict.")

            return {
                "can_answer": bool(parsed.get("can_answer", False)),
                "current_understanding": str(parsed.get("current_understanding", "") or ""),
                "missing_info": str(parsed.get("missing_info", "") or ""),
            }

        except Exception as e:
            logger.error(f"{Fore.RED}Error in rank_aware_rag: {e}{Style.RESET_ALL}")
            return {
                "can_answer": False,
                "current_understanding": f"Error during rank-level analysis: {str(e)}",
                "missing_info": "Rank-level analysis failed.",
            }

    # ==================================================================
    # Dynamic DAG Adaptation 관련 헬퍼
    # 논문 Algorithm 1 line 14–17, Section 3.2 ❸ 구현
    # ==================================================================

    def _match_node_id_by_text(
        self,
        dag: QueryLogicDAG,
        text: str,
    ) -> Optional[int]:
        """
        DAG의 V에서 주어진 text와 일치하는 node id 반환.
        대소문자/공백 무시 매칭.

        decompose_unified_context_by_subproblem이 반환한 subproblem text를
        DAG node id로 역매핑할 때 사용.

        Args:
            dag: QueryLogicDAG 객체.
            text: 매칭하려는 subproblem text.

        Returns:
            int: 매칭되는 node id.
            None: 매칭 실패 또는 text가 비어있음.
        """
        target = (text or "").strip().lower()
        if not target:
            return None

        for node_id, node in dag.V.items():
            if node.text.strip().lower() == target:
                return node_id
        return None

    def _maybe_add_subproblem(
        self,
        question: str,
        info_summary: str,
        dag: QueryLogicDAG,
        sub_answers: Dict[int, str],
        current_max_rank: int,
    ) -> Optional[Dict[str, Any]]:
        """
        논문 Algorithm 1 line 14–17 + Section 3.2 ❸ Dynamic Adaptation 구현.

        매 rank 처리 직후 호출되어, 현재까지의 정보로 원 질문에 답할 수 있는지
        LLM에게 판정 요청한다. 새 subproblem이 필요하면 DAG에 추가하고
        그 정보를 반환한다.

        새 노드는 항상 max(기존 id) + 1 위치에 추가되고, edge는
        기존 노드 → 새 노드 방향만 만들어지므로 cycle은 구조적으로 불가능.
        따라서 verify_non_cyclicity 재호출은 필요 없다.

        Args:
            question: 원래 질문 Q.
            info_summary: 현재까지 누적된 rolling memory.
            dag: QueryLogicDAG 객체. 내부 상태가 mutate된다.
            sub_answers: 지금까지 풀린 sub의 답 모음 {node_id: answer_string}.
            current_max_rank: 현재까지 사용된 최대 rank.

        Returns:
            None: 추가 sub 불필요 → 다음 rank로 자연스럽게 진행.
            Dict: 새 sub가 추가됨. 다음 키들을 포함한다.
                - "new_subproblem_id": int     - 새로 부여된 node id
                - "new_subproblem_text": str   - 새 sub 문장
                - "new_rank": int              - 새 sub의 rank
                - "depends_on": List[int]      - 의존하는 기존 node id 목록
                - "reason": str                - LLM이 제시한 추가 이유
        """
        # ── Step 1: LLM에게 보낼 요약 정보 구성 ──
        nodes_summary_lines = []
        for node_id in sorted(dag.V.keys()):
            node_text = dag.V[node_id].text
            answer = sub_answers.get(node_id, "(unresolved)")
            nodes_summary_lines.append(
                f"- id={node_id}, text=\"{node_text}\", answer=\"{answer}\""
            )
        nodes_summary = "\n".join(nodes_summary_lines)

        prompt = f"""You are deciding whether to extend a Query Logic Dependency Graph with ONE additional subproblem.

Original question:
{question}

Existing subproblems in the DAG (with their resolved answers, if any):
{nodes_summary}

Current rolling memory:
{info_summary}

Task:
Decide whether ONE additional subproblem is needed to fully answer the original question.

Rules:
- If the question can already be answered with the existing subproblem answers and rolling memory, output {{"need_new_subproblem": false}} and set other fields to null/empty.
- If a new subproblem is needed, output its text and the existing subproblem ids it depends on.
- The new subproblem MUST NOT duplicate any existing subproblem.
- depends_on must reference existing subproblem ids only (or be an empty list if independent).
- Add at most ONE subproblem per call.

Output format (JSON ONLY, no other text):
{{
  "need_new_subproblem": boolean,
  "new_subproblem_text": string or null,
  "depends_on": [list of integers],
  "reason": string
}}"""

        # ── Step 2: LLM 호출 + JSON 파싱 ──
        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            result = fix_json_response(response)
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error in _maybe_add_subproblem LLM call: {e}{Style.RESET_ALL}"
            )
            return None

        if not isinstance(result, dict):
            logger.warning(
                f"{Fore.YELLOW}Dynamic adaptation: LLM response was not a dict.{Style.RESET_ALL}"
            )
            return None

        if not bool(result.get("need_new_subproblem", False)):
            logger.info(
                f"{Fore.GREEN}Dynamic adaptation: LLM determined no new subproblem is needed.{Style.RESET_ALL}"
            )
            return None

        # ── Step 3: 응답 검증 ──
        new_text = result.get("new_subproblem_text") or ""
        if not isinstance(new_text, str):
            return None
        new_text = new_text.strip()
        if not new_text:
            logger.warning(
                f"{Fore.YELLOW}Dynamic adaptation: empty new_subproblem_text.{Style.RESET_ALL}"
            )
            return None

        if self._match_node_id_by_text(dag, new_text) is not None:
            logger.warning(
                f"{Fore.YELLOW}Dynamic adaptation: new subproblem duplicates "
                f"an existing one. Skip.{Style.RESET_ALL}"
            )
            return None

        existing_ids = set(dag.V.keys())
        depends_on_raw = result.get("depends_on", []) or []
        depends_on: List[int] = []
        for raw_id in depends_on_raw:
            if isinstance(raw_id, bool):
                continue
            if not isinstance(raw_id, (int, float)):
                continue
            try:
                pid = int(raw_id)
            except (TypeError, ValueError):
                continue
            if pid in existing_ids and pid not in depends_on:
                depends_on.append(pid)

        reason = str(result.get("reason", "") or "Dynamic adaptation.").strip()

        # ── Step 4: DAG mutation (node 추가) ──
        new_id = max(dag.V.keys()) + 1 if dag.V else 0

        try:
            dag.add_node(
                SubproblemNode(
                    id=new_id,
                    text=new_text,
                    metadata={
                        "source": "dynamic_adaptation",
                        "created_by": "maybe_add_subproblem",
                        "added_at_round_max_rank": current_max_rank,
                    },
                )
            )
        except ValueError as e:
            logger.error(
                f"{Fore.RED}Dynamic adaptation: failed to add node {new_id}: {e}{Style.RESET_ALL}"
            )
            return None

        # ── Step 5: DAG mutation (edge 추가) ──
        for parent_id in depends_on:
            try:
                dag.add_edge(
                    DependencyEdge(
                        prerequisite_id=parent_id,
                        dependent_id=new_id,
                        reason=reason,
                        metadata={
                            "source": "dynamic_adaptation",
                            "relation_type": "logical_precedence",
                            "created_by": "maybe_add_subproblem",
                        },
                    )
                )
            except ValueError as e:
                logger.warning(
                    f"{Fore.YELLOW}Dynamic adaptation: failed to add edge "
                    f"{parent_id}->{new_id}: {e}{Style.RESET_ALL}"
                )

        # ── Step 6: 새 노드의 rank 계산 ──
        # 부모가 있으면 max(parent_rank) + 1, 없으면 current_max_rank + 1
        # 논문 Algorithm 1 line 16: "Append p_{n+1} as a new rank after the current rank sequence"
        if depends_on:
            parent_ranks = [dag.ranks.get(p, 0) for p in depends_on]
            new_rank = max(parent_ranks) + 1
        else:
            new_rank = current_max_rank + 1

        dag.ranks[new_id] = new_rank

        logger.info(
            f"{Fore.YELLOW}Dynamic adaptation: added subproblem #{new_id} "
            f"\"{new_text}\" at rank {new_rank} "
            f"(depends_on={depends_on}). Reason: {reason}{Style.RESET_ALL}"
        )

        return {
            "new_subproblem_id": new_id,
            "new_subproblem_text": new_text,
            "new_rank": new_rank,
            "depends_on": depends_on,
            "reason": reason,
        }

    # ==================================================================
    # ranked subproblem groups 추출 헬퍼
    # ==================================================================

    @staticmethod
    def _ranked_subproblem_groups(
        topological_rank_result: Dict[str, Any],
        sorted_dependencies: List[str],
    ) -> List[Dict[str, Any]]:
        """Topological rank 결과를 unified retrieval loop에서 돌기 쉬운 형태로 변환."""
        ranked_dependencies = topological_rank_result.get("ranked_dependencies", {}) if topological_rank_result else {}
        groups: List[Dict[str, Any]] = []

        if isinstance(ranked_dependencies, dict) and ranked_dependencies:
            for raw_rank, raw_subproblems in ranked_dependencies.items():
                try:
                    rank = int(raw_rank)
                except (TypeError, ValueError):
                    continue

                if not isinstance(raw_subproblems, list):
                    continue

                subproblems = [
                    item.strip()
                    for item in raw_subproblems
                    if isinstance(item, str) and item.strip()
                ]

                if subproblems:
                    groups.append({
                        "rank": rank,
                        "subproblems": subproblems,
                    })

        if not groups:
            groups = [
                {
                    "rank": idx,
                    "subproblems": [dependency],
                }
                for idx, dependency in enumerate(sorted_dependencies or [])
                if isinstance(dependency, str) and dependency.strip()
            ]

        return sorted(groups, key=lambda item: item["rank"])

    def _retrieve_for_query(
        self,
        query: str,
        retrieved_chunks_set: Optional[set] = None,
    ) -> List[str]:
        """unified query 하나를 실제 retrieval에 넘기는 wrapper 함수."""
        if self.filter_repeats and retrieved_chunks_set is not None:
            contexts = self._retrieve_with_filter(query, retrieved_chunks_set)
            for chunk in contexts:
                retrieved_chunks_set.add(chunk)
            return contexts

        return self.retrieve(query)

    def generate_answer(self, question: str, info_summary: str) -> str:
        """Generate final answer based on the information summary (논문 line 19: Compose)."""
        try:
            prompt = f"""You must give ONLY the direct answer in the most concise way possible. DO NOT explain or provide any additional context.
If the answer is a simple yes/no, just say "Yes." or "No."
If the answer is a name, just give the name.
If the answer is a date, just give the date.
If the answer is a number, just give the number.
If the answer requires a brief phrase, make it as concise as possible.

Question: {question}

Information Summary:
{info_summary}

Remember: Be concise - give ONLY the essential answer, nothing more.
Ans: """
            return get_response_with_retry(prompt)
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating answer: {e}{Style.RESET_ALL}")
            return ""

    def _sort_dependencies(self, dependencies: List[str], query) -> List[Tuple]:
        """[Legacy] dependency sorting (현재 흐름에서는 _verify_sort_dependencies_with_repair로 대체)."""
        prompt = f"""
        Given the question:
        Question: {query}

        and its decomposed dependencies:
        Dependencies: {dependencies}

        Please output the dependency pairs that dependency A relies on dependency B, if any. If no dependency pairs are found, output an empty list.

        format your response as a JSON object with these keys:
        - "dependency_pairs": list of tuples of integers
        """
        response = get_response_with_retry(prompt)
        result = fix_json_response(response)
        dependency_pairs = result["dependency_pairs"]
        return self._topological_sort(dependencies, dependency_pairs)

    @staticmethod
    def _get_dag_node_edge_keys(dag_dict: Dict[str, Any]) -> Tuple[str, str]:
        if "nodes" in dag_dict and "edges" in dag_dict:
            return "nodes", "edges"
        if "V" in dag_dict and "E" in dag_dict:
            return "V", "E"
        raise ValueError("DAG dict must have either nodes/edges or V/E.")

    @staticmethod
    def _nodes_payload_from_dag_dict(dag_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        node_key, _ = LogicRAG._get_dag_node_edge_keys(dag_dict)
        raw_nodes = dag_dict[node_key]

        nodes_payload = []

        for raw_key, raw_node in raw_nodes.items():
            if not isinstance(raw_node, dict):
                continue

            try:
                node_id = int(raw_node.get("id", raw_key))
            except (TypeError, ValueError):
                continue

            text = raw_node.get("text", "")

            nodes_payload.append({
                "id": node_id,
                "text": text,
            })

        return sorted(nodes_payload, key=lambda item: item["id"])

    @staticmethod
    def _rebuild_dag_indexes_dict(dag_dict: Dict[str, Any]) -> Dict[str, Any]:
        node_key, edge_key = LogicRAG._get_dag_node_edge_keys(dag_dict)

        raw_nodes = dag_dict[node_key]
        raw_edges = dag_dict[edge_key]

        node_ids = []

        for raw_key, raw_node in raw_nodes.items():
            if not isinstance(raw_node, dict):
                continue

            try:
                node_id = int(raw_node.get("id", raw_key))
            except (TypeError, ValueError):
                continue

            node_ids.append(node_id)

        node_id_set = set(node_ids)

        parents = {node_id: set() for node_id in node_ids}
        children = {node_id: set() for node_id in node_ids}

        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue

            try:
                pre = int(edge["prerequisite_id"])
                dep = int(edge["dependent_id"])
            except (KeyError, TypeError, ValueError):
                continue

            if pre not in node_id_set or dep not in node_id_set:
                continue

            if pre == dep:
                continue

            children[pre].add(dep)
            parents[dep].add(pre)

        dag_dict["parents"] = {
            node_id: sorted(list(parent_ids))
            for node_id, parent_ids in parents.items()
        }

        dag_dict["children"] = {
            node_id: sorted(list(child_ids))
            for node_id, child_ids in children.items()
        }

        return dag_dict

    def _repair_cyclic_dag_with_llm(
        self,
        question: str,
        dag_dict: Dict[str, Any],
        dag_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """cycle 있는 DAG → LLM에게 고쳐달라고 요청."""
        _, edge_key = self._get_dag_node_edge_keys(dag_dict)

        nodes_payload = self._nodes_payload_from_dag_dict(dag_dict)
        current_edges = dag_result.get("valid_dependency_edges") or dag_dict.get(edge_key, [])

        cycle_node_ids = dag_result.get("cycle_node_ids", [])
        cycle_dependencies = dag_result.get("cycle_dependencies", [])
        blocked_node_ids = dag_result.get("blocked_node_ids", [])

        prompt = f"""
You are repairing the edge set of a Query Logic Dependency Graph.

Original question:
{question}

Subproblem nodes:
{json.dumps(nodes_payload, ensure_ascii=False, indent=2)}

Current directed edges:
{json.dumps(current_edges, ensure_ascii=False, indent=2)}

Detected cycle node ids:
{json.dumps(cycle_node_ids, ensure_ascii=False)}

Detected cycle subproblems:
{json.dumps(cycle_dependencies, ensure_ascii=False, indent=2)}

Blocked node ids:
{json.dumps(blocked_node_ids, ensure_ascii=False)}

Task:
Repair the edge set so that the graph becomes a valid DAG.

Rules:
- Use only node ids from the provided subproblem nodes.
- Edge direction must be prerequisite_id -> dependent_id.
- Do not create self-loops.
- Remove or modify the minimum number of edges needed to break cycles.
- Preserve necessary logical dependencies when possible.
- Do not add redundant transitive edges.
- Each edge must include a short reason.
- Return ONLY a JSON object.

Output schema:
{{
  "edges": [
    {{
      "prerequisite_id": integer,
      "dependent_id": integer,
      "reason": string
    }}
  ]
}}
"""

        try:
            response = get_response_with_retry(prompt)
            response = response.strip()
            response = response.replace("```json", "").replace("```", "")

            repaired = fix_json_response(response)

            if not isinstance(repaired, dict) or not isinstance(repaired.get("edges"), list):
                logger.error(
                    f"{Fore.RED}Failed to parse repaired DAG edges. "
                    f"Using original DAG for re-verification.{Style.RESET_ALL}"
                )
                return copy.deepcopy(dag_dict)

            repaired_edges = []

            for edge in repaired["edges"]:
                if not isinstance(edge, dict):
                    continue

                try:
                    prerequisite_id = edge["prerequisite_id"]
                    dependent_id = edge["dependent_id"]

                    if isinstance(prerequisite_id, bool) or isinstance(dependent_id, bool):
                        continue

                    repaired_edges.append({
                        "prerequisite_id": int(prerequisite_id),
                        "dependent_id": int(dependent_id),
                        "reason": edge.get("reason", ""),
                    })
                except (KeyError, TypeError, ValueError):
                    continue

            repaired_dag_dict = copy.deepcopy(dag_dict)
            repaired_dag_dict[edge_key] = repaired_edges
            repaired_dag_dict = self._rebuild_dag_indexes_dict(repaired_dag_dict)

            return repaired_dag_dict

        except Exception as e:
            logger.error(f"{Fore.RED}Error during DAG repair: {e}{Style.RESET_ALL}")
            return copy.deepcopy(dag_dict)

    def _verify_sort_dependencies_with_repair(
        self,
        question: str,
        dag_dict: Dict[str, Any],
        max_repair_attempts: int = 1,
        on_repair_failure: str = "raise",
    ) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """DAG 검증 + topological sort + cycle repair orchestration."""
        if on_repair_failure not in {"raise", "fallback"}:
            raise ValueError("on_repair_failure must be either 'raise' or 'fallback'.")

        verification_history = []
        current_dag_dict = copy.deepcopy(dag_dict)

        dag_result = verify_dag_and_topological_sort(current_dag_dict)

        verification_history.append({
            "attempt": 0,
            "type": "initial_verification",
            "dag": current_dag_dict,
            "dag_verification": dag_result,
        })

        if dag_result["is_dag"]:
            return dag_result["sorted_dependencies"], current_dag_dict, verification_history

        if not dag_result.get("has_cycle", False):
            if on_repair_failure == "fallback":
                fallback_dependencies = build_partial_order_fallback(dag_result)
                return fallback_dependencies, current_dag_dict, verification_history

            raise ValueError(
                f"Invalid DAG input. invalid_edges={dag_result.get('invalid_edges', [])}"
            )

        for attempt in range(1, max_repair_attempts + 1):
            logger.warning(
                f"{Fore.YELLOW}Cycle detected in Query Logic DAG. "
                f"Attempting LLM repair {attempt}/{max_repair_attempts}.{Style.RESET_ALL}"
            )

            repaired_dag_dict = self._repair_cyclic_dag_with_llm(
                question=question,
                dag_dict=current_dag_dict,
                dag_result=dag_result,
            )

            repaired_result = verify_dag_and_topological_sort(repaired_dag_dict)

            verification_history.append({
                "attempt": attempt,
                "type": "llm_repair_verification",
                "dag": repaired_dag_dict,
                "dag_verification": repaired_result,
            })

            current_dag_dict = repaired_dag_dict
            dag_result = repaired_result

            if dag_result["is_dag"]:
                logger.info(f"{Fore.GREEN}DAG repair succeeded.{Style.RESET_ALL}")
                return dag_result["sorted_dependencies"], current_dag_dict, verification_history

            if not dag_result.get("has_cycle", False):
                break

        if on_repair_failure == "fallback":
            logger.warning(
                f"{Fore.YELLOW}DAG repair failed. "
                f"Using partial-order fallback.{Style.RESET_ALL}"
            )
            fallback_dependencies = build_partial_order_fallback(dag_result)
            return fallback_dependencies, current_dag_dict, verification_history

        raise DependencyGraphCycleError(dag_result)

    def _retrieve_with_filter(self, query: str, retrieved_chunks_set: set) -> list:
        """context chunk에 대한 Sampling without Replacement."""
        if retrieved_chunks_set is None:
            retrieved_chunks_set = set()

        if self.corpus_embeddings is None or not self.corpus:
            return []

        target_k = min(int(self.top_k), len(self.corpus))
        if target_k <= 0:
            return []

        unique_results = []
        retrieval_window = target_k

        while len(unique_results) < target_k and retrieval_window <= len(self.corpus):
            all_results = self._retrieve_top_n(query, retrieval_window)
            unique_results = [
                chunk
                for chunk in all_results
                if chunk not in retrieved_chunks_set
            ]

            if len(unique_results) >= target_k:
                break

            retrieval_window += target_k

        return unique_results[:target_k]

    def _retrieve_top_n(self, query: str, n: int) -> list:
        """query에 대해 top-n 결과 검색."""
        old_top_k = self.top_k
        try:
            self.top_k = min(int(n), len(self.corpus))
            return self.retrieve(query)
        finally:
            self.top_k = old_top_k

    # ==================================================================
    # 메인 진입점 (논문 Algorithm 1 그대로 따름)
    # ==================================================================

    def answer_question(self, question: str) -> Tuple[str, List[str], int]:

        info_summary = ""
        round_count = 0
        retrieval_history = []
        last_contexts = []
        dependency_analysis_history = []
        retrieved_chunks_set = set() if self.filter_repeats else None

        print(f"\n\n{Fore.CYAN}{self.MODEL_NAME} answering: {question}{Style.RESET_ALL}\n\n")

        # ===============================================
        # == Stage 1: warm up retrieval + decompose_query ==
        # 논문 Algorithm 1 line 1: decompose Q into subproblems P
        # ===============================================
        if self.filter_repeats:
            new_contexts = self._retrieve_with_filter(question, retrieved_chunks_set)
            for chunk in new_contexts:
                retrieved_chunks_set.add(chunk)
        else:
            new_contexts = self.retrieve(question)
        last_contexts = new_contexts
        info_summary = self.refine_summary_with_context(
            question,
            new_contexts,
            info_summary
        )

        decomposition = self.decompose_query(question)

        if decomposition["is_simple"]:
            print("Query decomposition indicates a simple single-hop question. Answering directly.")
            answer = self.generate_answer(question, info_summary)
            self.last_dependency_analysis = []
            self.last_query_logic_dag = None
            return answer, last_contexts, round_count
        else:
            logger.info(f"Query decomposition result: {len(decomposition['subproblems'])} subproblems detected.")
            logger.info(f"Subproblems: {decomposition['subproblems']}")

            # ===============================================
            # == Stage 2: Build Query Logic DAG ==
            # 논문 Algorithm 1 line 2–3: Initialize DAG, populate edges
            # ===============================================
            dag = self.dag_builder.construct_from_subproblems(
                question=question,
                subproblems=decomposition["subproblems"],
            )

        self.last_query_logic_dag = dag
        dag_dict = dag.to_dict()

        dependency_analysis_history.append({
            "query_logic_dag": dag_dict,
        })
        logger.info(f"Constructed Query Logic DAG: {dag_dict}\n\n")

        # ===============================================
        # == Stage 3: DAG topological sort + cycle verification ==
        # ===============================================
        sorted_dependencies, verified_dag_dict, dag_verification_history = (
            self._verify_sort_dependencies_with_repair(
                question=question,
                dag_dict=dag_dict,
                max_repair_attempts=self.max_dag_repair_attempts,
                on_repair_failure=self.dag_cycle_policy,
            )
        )

        self.last_query_logic_dag_dict = verified_dag_dict

        # ===============================================
        # == Stage 4: Topological rank calculation ==
        # 논문 Algorithm 1 line 4: Topologically sort G to obtain ranks
        # ===============================================
        final_dag_result = dag_verification_history[-1]["dag_verification"]

        if final_dag_result.get("is_dag", False):
            try:
                topological_rank_result = compute_topological_ranks_from_verification(
                    final_dag_result
                )

                verified_dag_dict = attach_topological_ranks_to_dag_dict(
                    verified_dag_dict,
                    topological_rank_result,
                )

                # dag 객체 내부의 ranks dict도 함께 채워둔다.
                # _maybe_add_subproblem이 dag.ranks.get()으로 부모 rank를 조회하므로 필요.
                ranks_payload = topological_rank_result.get("ranks", {}) or {}
                for raw_id, raw_rank in ranks_payload.items():
                    try:
                        node_id = int(raw_id)
                        rank_val = int(raw_rank)
                    except (TypeError, ValueError):
                        continue
                    if node_id in dag.V:
                        dag.ranks[node_id] = rank_val

                logger.info(f"Topological rank result: {topological_rank_result}\n\n")

            except TopologicalRankError as e:
                logger.error(
                    f"{Fore.RED}Failed to compute topological ranks: {e}{Style.RESET_ALL}"
                )
                topological_rank_result = {}
        else:
            topological_rank_result = {}
            logger.warning(
                f"{Fore.YELLOW}Skip topological rank calculation because final DAG is not valid.{Style.RESET_ALL}"
            )

        self.last_query_logic_dag_dict = verified_dag_dict

        dependency_analysis_history.append({
            "query_logic_dag": dag_dict,
            "verified_query_logic_dag": verified_dag_dict,
            "dag_verification_history": dag_verification_history,
            "topological_rank": topological_rank_result,
            "sorted_dependencies": sorted_dependencies,
        })

        logger.info(f"Verified Query Logic DAG: {verified_dag_dict}\n\n")
        logger.info(f"Sorted dependencies: {sorted_dependencies}\n\n")

        # ===============================================
        # == Stage 5: rank-level unified retrieval +
        #             Dynamic DAG Adaptation
        # 논문 Algorithm 1 line 6–18: for each rank r in ascending topological order
        # ===============================================
        ranked_subproblem_groups = self._ranked_subproblem_groups(
            topological_rank_result=topological_rank_result,
            sorted_dependencies=sorted_dependencies,
        )

        dependency_analysis_history.append({
            "stage": "unified_subquery_generation",
            "ranked_subproblem_groups": ranked_subproblem_groups,
        })

        # rank-batch 단위의 Sampling without Replacement.
        # 한 번 pop된 rank group은 다시 방문하지 않는다.
        remaining_rank_groups = list(ranked_subproblem_groups)

        # Dynamic adaptation을 위한 상태 변수
        # sub_answers: 지금까지 풀린 sub의 답을 dag node id 기준으로 누적
        #              (논문 line 12의 a_i를 누적)
        # current_max_rank: 새 sub를 어디에 끼울지 결정할 때 사용
        # adaptation_count: max_dynamic_adaptations 안전장치
        sub_answers: Dict[int, str] = {}
        if ranked_subproblem_groups:
            current_max_rank = max(
                g["rank"] for g in ranked_subproblem_groups
            )
        else:
            current_max_rank = -1
        adaptation_count = 0

        # 논문 line 6: for each rank r in ascending topological order
        # remaining_rank_groups가 비면 자연스럽게 종료.
        # dynamic adaptation이 새 rank를 push하면 list가 다시 채워져 루프 연장.
        while remaining_rank_groups and round_count < self.max_rounds:

            rank_group = remaining_rank_groups.pop(0)
            round_count += 1

            rank = rank_group["rank"]
            same_rank_subproblems = rank_group["subproblems"]

            # 논문 line 7: Let S_r = {p_i | rank(p_i) = r}
            # 논문 line 8: Construct unified query q_r = Merge(S_r) using LLM
            unified_query = self.build_unified_query(
                question=question,
                rank=rank,
                subproblems=same_rank_subproblems,
            )

            # 논문 line 9: Retrieve documents C_r = R(q_r; θ, K)
            new_contexts = self._retrieve_for_query(
                unified_query,
                retrieved_chunks_set=retrieved_chunks_set,
            )
            last_contexts = new_contexts

            # 논문 line 11–12: for each p_i ∈ S_r, resolve p_i → a_i
            # decompose_unified_context_by_subproblem이 한 번의 LLM 호출로
            # S_r 전체의 a_i를 동시에 도출한다 (논문의 inner for를 batch 처리).
            decomposed_result = self.decompose_unified_context_by_subproblem(
                question=question,
                rank=rank,
                subproblems=same_rank_subproblems,
                unified_query=unified_query,
                contexts=new_contexts,
                current_summary=info_summary,
            )

            # 논문 line 10: Summarize C_r with respect to Q and Mem_{r-1} to get Mem_r
            info_summary = self.refine_summary_with_unified_rank_result(
                question=question,
                rank=rank,
                subproblems=same_rank_subproblems,
                unified_query=unified_query,
                contexts=new_contexts,
                decomposed_result=decomposed_result,
                current_summary=info_summary,
            )

            logger.info(f"Unified retrieval at round {round_count}, topological rank {rank}")
            logger.info(f"same-rank subproblems: {same_rank_subproblems}")
            logger.info(f"unified query: {unified_query}")

            # sub_answers 누적 (논문 line 12의 a_i 저장에 해당)
            for sub_ans in decomposed_result.get("subproblem_answers", []):
                sub_text = sub_ans.get("subproblem", "")
                answer_text = sub_ans.get("answer", "") or ""
                if not isinstance(sub_text, str) or not isinstance(answer_text, str):
                    continue
                node_id = self._match_node_id_by_text(dag, sub_text)
                if node_id is not None and answer_text.strip():
                    sub_answers[node_id] = answer_text.strip()

            retrieval_history.append({
                "round": round_count,
                "rank": rank,
                "subproblems": same_rank_subproblems,
                "unified_query": unified_query,
                "contexts": new_contexts,
                "decomposed_result": decomposed_result,
            })

            # ── 논문 line 14–17: Dynamic DAG Adaptation ──
            # 매 rank 처리 직후 LLM에게 새 unresolved subproblem이 있는지 묻고,
            # 있으면 DAG에 새 노드/엣지를 추가하여 다음 rank로 처리되도록 한다.
            # 새 rank가 추가되면 remaining_rank_groups에 push되어
            # 외곽 while 루프가 자연스럽게 한 번 더 돈다.
            adaptation_event: Optional[Dict[str, Any]] = None
            if adaptation_count < self.max_dynamic_adaptations:
                new_sub_info = self._maybe_add_subproblem(
                    question=question,
                    info_summary=info_summary,
                    dag=dag,
                    sub_answers=sub_answers,
                    current_max_rank=current_max_rank,
                )

                if new_sub_info is not None:
                    new_id = new_sub_info["new_subproblem_id"]
                    new_rank = new_sub_info["new_rank"]
                    new_text = new_sub_info["new_subproblem_text"]

                    remaining_rank_groups.append({
                        "rank": new_rank,
                        "subproblems": [new_text],
                    })
                    current_max_rank = max(current_max_rank, new_rank)
                    adaptation_count += 1
                    adaptation_event = new_sub_info

            dependency_analysis_history.append({
                "round": round_count,
                "rank": rank,
                "subproblems": same_rank_subproblems,
                "unified_query": unified_query,
                "decomposed_result": decomposed_result,
                "adaptation": adaptation_event,
            })

            # 논문 line 6의 for 루프는 모든 rank를 끝까지 돌고,
            # 중간에 종료하지 않는다. 따라서 can_answer 기반 즉시 종료 분기는 없다.

        # ─── 논문 line 19: 모든 rank 처리 완료 → final answer = Compose({a_i}) ───
        logger.info(
            f"Completed rank-level retrieval "
            f"(rounds={round_count}/{self.max_rounds}, "
            f"adaptations={adaptation_count}). Generating final answer..."
        )
        answer = self.generate_answer(question, info_summary)
        self.last_dependency_analysis = dependency_analysis_history
        self.last_retrieval_history = retrieval_history
        return answer, last_contexts, round_count