"""
decompose_query() 프롬프트 실험용 LogicRAG 서브클래스.

base 코드(logic_rag.py)를 수정하지 않고 decompose_query()만 override한다.

실험 목록:
  LogicRAGExpEntityCoT   — 실험 1: pivot entity 먼저 식별 후 분해 (CoT)
  LogicRAGExpSelfVerify  — 실험 2: 분해 후 각 step 자기검증 + 분할
  LogicRAGExpHopCount    — 실험 3: hop count 사전 추정 후 분해

실행은 run_decomp_experiments.py 참고.
"""

import logging
from typing import Any, Dict, List

from src.models.logic_rag import LogicRAG, QUERY_DECOMPOSITION_FEW_SHOT_EXAMPLES
from src.utils.utils import get_response_with_retry, fix_json_response

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 실험 1: Entity-first CoT
# ──────────────────────────────────────────────────────────────────────────────

class LogicRAGExpEntityCoT(LogicRAG):
    """
    실험 1 — pivot entity를 먼저 식별한 뒤 분해하는 CoT 방식.

    기존 방식: 질문 → 한 번에 subproblem 리스트 출력
    이 방식 : 질문 → ① pivot entity 목록 나열 (thought) → ② 순서대로 subproblem 변환

    타깃 오류: 패턴 2(소유 속성 체인 축약), 패턴 1(과소분해)
    추가 API 호출: 없음 (single-prompt CoT)
    """

    def decompose_query(self, question: str) -> Dict[str, Any]:
        try:
            prompt = f"""You are an expert at decomposing complex questions into smaller, logically ordered subproblems.

Given a question, first identify the key intermediate entities, then decompose into subproblems.

Step 1 — Identify "pivot entities": intermediate entities that must be explicitly looked up
  because their result feeds into the next lookup. Include ONLY:
  - Possessive properties used as input: e.g., "X's religion", "Y's record label"
  - Unnamed entities described by a relative clause: e.g., "the person who did X"
  Do NOT include:
  - Entities that are already explicitly and unambiguously named in the question
  - Attributes that can be looked up in a single step from a named entity
  Keep the pivot entity list as short as possible.

Step 2 — Order pivot entities by dependency chain (what must come first?)

Step 3 — Create one subproblem per lookup, in dependency order

Rules:
1. Each subproblem must ask for exactly one fact that can be looked up independently.
2. Create a new subproblem only when its answer is needed as input for a later subproblem or for the final answer.
3. Do not add background, context, explanation, or verification steps.
4. If "X's Y" is used as input for a later lookup, "What is X's Y?" must be its own subproblem.
5. If an entity is already named and only one direct attribute is needed, that is ONE subproblem.
6. Preserve the original question's intent, key terms, and expected answer type exactly.
7. If a subproblem depends on a previous answer, refer to it as "this person", "this city", etc.
8. If the question involves comparison or multiple targets, resolve each independently first, then compare.
9. If the question can be answered with a single lookup, output one subproblem and mark is_simple=true.
10. If two or more anchor entities determine a final relational lookup, create a separate lookup for each anchor first.
11. Minimize the total number of subproblems — do not add steps that are not strictly required.

Here are some examples:
{QUERY_DECOMPOSITION_FEW_SHOT_EXAMPLES}

Now decompose the following question.

Question: "{question}"

Respond ONLY with a JSON object with these keys:
  "thought"      : brief list of pivot entities and their order (1-2 sentences)
  "subproblems"  : list of objects, each with "id" (int) and "text" (string)
  "is_simple"    : boolean"""

            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            result = fix_json_response(response)

            if result is None:
                return {"subproblems": [{"id": 0, "text": question}], "is_simple": True}

            if (
                "subproblems" not in result
                or not isinstance(result["subproblems"], list)
                or len(result["subproblems"]) == 0
            ):
                result["subproblems"] = [{"id": 0, "text": question}]

            if "is_simple" not in result:
                result["is_simple"] = len(result["subproblems"]) <= 1

            # thought 필드는 downstream에서 사용하지 않음
            return {
                "subproblems": result["subproblems"],
                "is_simple": result["is_simple"],
            }

        except Exception as e:
            logger.error(f"LogicRAGExpEntityCoT.decompose_query error: {e}")
            return {"subproblems": [{"id": 0, "text": question}], "is_simple": True}


# ──────────────────────────────────────────────────────────────────────────────
# 실험 2: Self-verification loop
# ──────────────────────────────────────────────────────────────────────────────

class LogicRAGExpSelfVerify(LogicRAG):
    """
    실험 2 — 초기 분해 후 각 step이 단일 직접 lookup인지 자기검증.

    "이 step은 단일 사실 하나로 직접 답할 수 있는가?
     아니면 먼저 중간 entity를 찾아야 하는가?"
    → 중간 entity가 필요하면 두 step으로 분할.

    타깃 오류: 패턴 1(과소분해 — 직접 답 가능처럼 보이는 2-hop)
    추가 API 호출: subproblem 수만큼 (+N 호출)
    검증은 단일 pass (무한 루프 방지)
    """

    def decompose_query(self, question: str) -> Dict[str, Any]:
        # 1단계: base 클래스로 초기 분해
        initial = super().decompose_query(question)

        if initial.get("is_simple") or not initial.get("subproblems"):
            return initial

        # 2단계: 각 step 검증 및 필요시 분할
        verified: List[Dict] = []
        for step in initial["subproblems"]:
            check = self._verify_step(question, step["text"])
            if check.get("needs_split") and len(check.get("steps", [])) == 2:
                verified.append({"text": check["steps"][0]})
                verified.append({"text": check["steps"][1]})
            else:
                verified.append({"text": step["text"]})

        # id 재부여
        for i, s in enumerate(verified):
            s["id"] = i

        return {
            "subproblems": verified,
            "is_simple": len(verified) <= 1,
        }

    def _verify_step(self, question: str, step_text: str) -> Dict[str, Any]:
        """단일 step이 직접 lookup인지, 아니면 중간 entity 탐색이 필요한지 판단한다."""
        prompt = f"""You are checking whether a decomposition step is granular enough for single-fact retrieval.

Original question: "{question}"
Step to check: "{step_text}"

Determine: can this step be answered by a SINGLE direct fact lookup (one Wikipedia-style property)?
Or does it first require finding an intermediate entity, and then looking up a property of that entity?

Examples of steps that NEED splitting:
- "In which district was Ernie Watts born?"
  → First find Ernie Watts's birthplace (city), THEN find which district that city is in.
- "Who wanted to reform John Kodwo Amissah's religion?"
  → First find Amissah's religion, THEN find who wanted to reform that religion.

Examples of steps that do NOT need splitting:
- "What is the capital of France?"  (single lookup)
- "Who is the mayor of Paris?"       (single lookup given Paris is already known)

Return ONLY a JSON object:
{{
  "needs_split": true or false,
  "steps": ["step A text", "step B text"]
}}
Note: "steps" is required only when needs_split=true. Keep step texts concise."""

        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            result = fix_json_response(response)
            if not isinstance(result, dict):
                return {"needs_split": False}
            return result
        except Exception as e:
            logger.error(f"_verify_step error: {e}")
            return {"needs_split": False}


# ──────────────────────────────────────────────────────────────────────────────
# 실험 3: Hop count 사전 추정
# ──────────────────────────────────────────────────────────────────────────────

class LogicRAGExpHopCount(LogicRAG):
    """
    실험 3 — hop count를 먼저 추정한 뒤 그 수에 맞게 분해.

    "이 질문은 독립적인 사실 조회가 몇 번 필요한가?"를 먼저 판단하고,
    그 수를 분해 프롬프트에 hint로 제공한다.

    타깃 오류: step count mismatch (현재 30%) — over/under-decomposition 둘 다
    추가 API 호출: 1회 (hop count 추정)
    """

    def decompose_query(self, question: str) -> Dict[str, Any]:
        # 1단계: hop count 추정
        hop_count = self._estimate_hop_count(question)

        # 2단계: hop count hint를 포함한 분해
        try:
            prompt = f"""You are an expert at decomposing complex questions into smaller, logically ordered subproblems.

Given a question, decompose it into the minimum number of subproblems needed to answer it.

Hop count estimate: this question likely requires approximately {hop_count} independent fact lookup(s).
Use this as a guide — if the question clearly needs more or fewer steps, adjust accordingly.

Rules:
1. Each subproblem must ask for exactly one fact that can be looked up independently.
2. Create a new subproblem only when its answer is needed as input for a later subproblem or for the final answer.
3. Do not add background, context, explanation, or verification steps.
4. If "X's Y" is used as input for a later lookup, "What is X's Y?" must be its own subproblem.
5. If an entity is already named and only one direct attribute is needed, that is ONE subproblem.
6. Preserve the original question's intent, key terms, and expected answer type exactly.
7. If a subproblem depends on a previous answer, refer to it as "this person", "this city", etc.
8. If the question involves comparison or multiple targets, resolve each independently first, then compare.
9. If the question can be answered with a single lookup, output one subproblem and mark is_simple=true.
10. If two or more anchor entities determine a final relational lookup, create a separate lookup for each anchor first.

Here are some examples:
{QUERY_DECOMPOSITION_FEW_SHOT_EXAMPLES}

Now decompose the following question:
Question: "{question}"

Respond ONLY with a JSON object with these keys:
  "subproblems" : list of objects, each with "id" (int) and "text" (string)
  "is_simple"   : boolean"""

            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            result = fix_json_response(response)

            if result is None:
                return {"subproblems": [{"id": 0, "text": question}], "is_simple": True}

            if (
                "subproblems" not in result
                or not isinstance(result["subproblems"], list)
                or len(result["subproblems"]) == 0
            ):
                result["subproblems"] = [{"id": 0, "text": question}]

            if "is_simple" not in result:
                result["is_simple"] = len(result["subproblems"]) <= 1

            return result

        except Exception as e:
            logger.error(f"LogicRAGExpHopCount.decompose_query error: {e}")
            return {"subproblems": [{"id": 0, "text": question}], "is_simple": True}

    def _estimate_hop_count(self, question: str) -> int:
        """질문이 필요로 하는 독립적 사실 조회 횟수(hop)를 추정한다."""
        prompt = f"""How many independent fact lookups does this question require?

Count each lookup whose answer depends on a previous lookup result as a separate hop.

Examples:
- "What is the capital of France?"  → 1 hop
- "Who is the mayor of the capital of France?"  → 2 hops (capital first, then mayor)
- "What is the population of the country where the telephone inventor was born?"  → 3 hops
- "Who is the child of the navigator who explored the coast of the continent where César Gaytan was born?"  → 4 hops

Question: "{question}"

Return ONLY a JSON object: {{"hops": <integer between 1 and 5>}}"""

        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            result = fix_json_response(response)
            if isinstance(result, dict) and "hops" in result:
                return max(1, min(5, int(result["hops"])))
        except Exception as e:
            logger.error(f"_estimate_hop_count error: {e}")

        return 2 
