"""
decompose_query() 프롬프트 실험용 LogicRAG 서브클래스 (exp4).

실험:
  LogicRAGExpQueryTypeClassifier — 실험 4: 질문 구조 유형 분류 후 유형별 최적화된 분해 전략 적용
  chain      — A→B→...→Z 순차 체인, ~87% (fallback)
  branching  — 독립 root 복수 개 → 병합 (+ 병합 후 체인 가능), ~13%

실행:
  python -m src.experiments.run_decomp_experiments_v2
"""

import logging
from typing import Any, Dict

from src.models.logic_rag import LogicRAG, QUERY_DECOMPOSITION_FEW_SHOT_EXAMPLES
from src.utils.utils import get_response_with_retry, fix_json_response

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 분류 few-shot (실제 MuSiQue 샘플 기반)
# ──────────────────────────────────────────────────────────────────────────────

_CLASSIFY_FEW_SHOT = """
Examples:

- "Who is the mayor of the capital of France?"
  → "chain"       (capital found first, then mayor — each step needs only the previous answer)

- "What is the religion of the person who composed the national anthem of Pakistan?"
  → "chain"       (composer found first, then religion)

- "What was the form of the language Auctor is in, used in the era of the Frankish king
   who created the Holy Roman Empire, later known as?"
  → "branching"   (two independent roots: the language of Auctor, AND the Frankish king
                   who created the Holy Roman Empire — both are needed before the final
                   "what was this language later known as in that era" step)

- "When was the region immediately north of the region where Israel is located and the
   location of the Battle of Qurah and Umm al Maradim created?"
  → "branching"   (two independent roots: Israel's region, AND the battle's location —
                   both feed the "region north of both" step, then one more step follows)

- "How were the people from whom new coins were a proclamation of independence by the
   Somali Muslim Ajuran Empire expelled from the country between Thailand and A Lim's
   country?"
  → "branching"   (one root is itself a 2-step chain — A Lim's country, then the boundary
                   with Thailand — merged with an independent root about who issued the
                   coins; the final step needs both)

- "What is the capital of France?"
  → "chain"       (single-hop — always classified as chain)
"""

# ──────────────────────────────────────────────────────────────────────────────
# 유형별 분해 프롬프트
# ──────────────────────────────────────────────────────────────────────────────

# baseline(logic_rag.py decompose_query)의 12-rule 프롬프트를 그대로 베이스로 삼는다.
# chain/branching 두 변형 모두 이 rule/예제 세트에서 출발 — 처음부터 얇은 프롬프트를
# 새로 쓰면 baseline이 다듬어온 3~4hop 대응력을 잃는다 (Exp4 1차 결과에서 확인:
# 3hop1 step_match 92.6%→59.3%, 4hop1 55.0%→25.0%로 급락했던 원인).
_BASE_RULES = """Rules:
    1. Each subproblem must ask for exactly one factual lookup target: one entity, attribute, location, date, number, or other value.
    2. Create a new subproblem only when its result is needed as input for another subproblem or directly needed for the final answer.
    3. Do not add background, context, explanation, or verification steps that are not strictly necessary to reach the final answer.
    4. If an entity or value must be found before it can be used in another lookup, create a separate subproblem for that intermediate entity or value. This includes possessive property chains: if "X's Y" is used as input for a later lookup, then "What is X's Y?" must be its own subproblem. Never combine the intermediate lookup with the step that uses it.
    5. If an entity is already explicitly named in the question and the question only requires one direct attribute of that entity, treat it as a single subproblem. Do not split a direct attribute lookup into multiple subproblems. For example, "Where was X born?" should not be split into "Who is X?" and "Where was X born?"
        Note: Rule 5 applies only when the direct attribute is the final answer. If an unknown property or relation of an explicitly named entity is needed as input for a later lookup, create one separate subproblem for that property or relation before the subproblem that uses it.
    6. Do not decompose descriptive modifiers unless they are required to identify the target entity. Keep modifiers such as "recently abdicated," "famous," "largest," or "first" as constraints only when they are necessary to find the correct entity.
    7. Each subproblem must preserve the original question's intent, key terms, constraints, and expected answer type. Do not remove or change dates, places, titles, organizations, relationships, or other constraints. If the original question asks "who," "when," "where," or "what," the final subproblem must preserve that answer type.
    8. If a subproblem uses an entity or value introduced by another subproblem, refer to that unknown result symbolically and unambiguously. Do not assume the actual answer during decomposition. For a single referenced result, use a typed phrase such as "this person", "this city", "this country", "this date", or "this entity". For multiple referenced results used together, use clear plural phrases such as "these two countries", "these two dates", "these locations", or "these entities". Avoid ambiguous singular references such as "this country" or "this entity" when more than one result of the same type is involved. Do not include formal dependency labels or step-id references in the subproblem text. The subproblem text should remain a natural-language question.
    9. If the question involves comparison, aggregation, judgment, or a final relation that depends on multiple independent entities or values, first create subproblems for the necessary entities or values, then add a final subproblem that performs the comparison, aggregation, judgment, or relation.
    10. If the question can be answered with a single independent lookup, output exactly one subproblem identical to the original question and mark "is_simple" as true.
    11. If the question asks about a location, relationship, or comparison involving two or more independently resolvable entities or values, create a separate lookup subproblem for each required entity or value before the final resolution step. Do not merge multiple independent lookups into one step. The final step must use all required entities or values together.
    12. Read the full question before decomposing. If the question contains a compound structure where the result of the first lookup is used in a different context for the second lookup, both steps are required. Do not stop at the first entity lookup and omit the contextual second step."""

_CHAIN_PROMPT = f"""You are an expert at decomposing complex questions into smaller, logically ordered subproblems.

    This question follows a single sequential dependency chain: each step's answer feeds
    directly into the next lookup, one fact at a time, with no independent branches to merge.

    Given a question, decompose it into the minimum number of subproblems needed to answer it.

    {_BASE_RULES}
    Here are some examples:
    {QUERY_DECOMPOSITION_FEW_SHOT_EXAMPLES}

    Now decompose the following question:
    Question: "{{question}}"

    Please format your response as a JSON object with these keys:

    * "subproblems": list of objects, each with "id" (int) and "text" (string)
    * "is_simple": boolean

    Respond ONLY with the JSON object, no additional text."""

_BRANCHING_PROMPT = f"""You are an expert at decomposing complex questions into smaller, logically ordered subproblems.

    This question has two or more independent "roots" — facts that can each be looked up
    without depending on each other — which then converge at a merge step that needs BOTH
    (or all) of their answers together. A root can itself be a short chain of 1-2 steps if
    it has its own internal dependency. If the merge step's result is itself needed for a
    further lookup, the decomposition must continue with additional steps after the merge —
    do not stop at the merge step if more information is still required for the final answer.

    Given a question, decompose it into the minimum number of subproblems needed to answer it.

    {_BASE_RULES}
    13. Identify every independent root first, and resolve each one fully (as its own mini-chain if it has internal dependencies) before the merge step. Do not merge two roots' lookups into one subproblem — keep them separate until the step that actually needs both.
    14. If the merge step's result feeds into a further lookup, continue decomposing after the merge step, one fact per subproblem, instead of stopping once the roots are combined.

    Here are some examples:
    {QUERY_DECOMPOSITION_FEW_SHOT_EXAMPLES}

    Example 9 (branching root that is itself a 2-step chain, merged with an independent second root):
    Question: "How were the people from whom new coins were a proclamation of independence by the Somali Muslim Ajuran Empire expelled from the country between Thailand and A Lim's country?"
    Subproblems:
    [
      {{"id": 0, "text": "What is A Lim's country?"}},
      {{"id": 1, "text": "What natural boundary lies between Thailand and this country?"}},
      {{"id": 2, "text": "From whom did new coins represent a proclamation of independence by the Ajuran Empire?"}},
      {{"id": 3, "text": "How were these people expelled from this boundary?"}}
    ]

    Example 10 (merge step followed by an additional chain step):
    Question: "When was the region immediately north of the region where Israel is located and the location of the Battle of Qurah and Umm al Maradim created?"
    Subproblems:
    [
      {{"id": 0, "text": "What region is Israel located in?"}},
      {{"id": 1, "text": "Where was the Battle of Qurah and Umm al Maradim?"}},
      {{"id": 2, "text": "What region lies immediately north of these two regions?"}},
      {{"id": 3, "text": "When was this region created?"}}
    ]

    Now decompose the following question:
    Question: "{{question}}"

    Please format your response as a JSON object with these keys:

    * "subproblems": list of objects, each with "id" (int) and "text" (string)
    * "is_simple": boolean

    Respond ONLY with the JSON object, no additional text."""

_TYPE_PROMPTS: Dict[str, str] = {
    "chain":     _CHAIN_PROMPT,
    "branching": _BRANCHING_PROMPT,
}

# ──────────────────────────────────────────────────────────────────────────────
# 실험 4: Query Structure Classifier
# ──────────────────────────────────────────────────────────────────────────────

class LogicRAGExpQueryTypeClassifier(LogicRAG):

    def decompose_query(self, question: str) -> Dict[str, Any]:
        q_type = self._classify_structure_type(question)
        result = self._decompose_by_type(question, q_type)
        return result

    # ── Step 1: 구조 유형 분류 ────────────────────────────────────────────────

    def _classify_structure_type(self, question: str) -> str:
        prompt = f"""Classify the following question into exactly one structural type,
based on its dependency graph shape — NOT its topic or phrasing.

- "chain"      : A single sequential dependency. Each step's answer feeds the next
                 lookup, one fact at a time, with no independent branches.
- "branching"  : Two or more independent facts ("roots") must each be looked up on
                 their own, then combined at a merge step that needs all of them.
                 A root may itself be a short 1-2 step chain. The chain may continue
                 after the merge step.

{_CLASSIFY_FEW_SHOT}

Question: "{question}"

Return ONLY a JSON object: {{"type": "<chain|branching>", "reason": "one sentence"}}"""

        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            result = fix_json_response(response)
            if isinstance(result, dict) and result.get("type") in _TYPE_PROMPTS:
                return result["type"]
        except Exception as e:
            logger.error(f"_classify_structure_type error: {e}")

        return "chain"

    # ── Step 2: 유형별 분해 ───────────────────────────────────────────────────

    def _decompose_by_type(self, question: str, q_type: str) -> Dict[str, Any]:
        # 프롬프트 안에 few-shot JSON 예제의 리터럴 중괄호가 섞여 있어 .format()은
        # 쓸 수 없다 (모든 "{...}"를 필드로 해석해 KeyError 발생) — 단순 치환 사용.
        prompt = _TYPE_PROMPTS.get(q_type, _CHAIN_PROMPT).replace("{question}", question)

        try:
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

            return {
                "subproblems": result["subproblems"],
                "is_simple": result["is_simple"],
            }

        except Exception as e:
            logger.error(f"_decompose_by_type error (type={q_type}): {e}")
            return {"subproblems": [{"id": 0, "text": question}], "is_simple": True}
