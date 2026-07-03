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

from src.models.logic_rag import LogicRAG
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

_CHAIN_PROMPT = """You are an expert at decomposing chain questions.

A chain question follows a single sequential dependency: the answer to each step feeds
directly into the next, and there is exactly one lookup at each point in the chain.

Rules:
1. Each subproblem must ask for exactly one fact.
2. If a subproblem depends on the previous answer, refer to it as "this person", "this city", etc.
3. Do not add steps that are not strictly necessary.
4. If "X's Y" is used as input for a later lookup, "What is X's Y?" must be its own subproblem.
5. Minimize total number of subproblems.

Examples:
- "Who is the mayor of the capital of France?"
  1. What is the capital of France?
  2. Who is the mayor of this city?

- "What is the religion of the person who composed the national anthem of Pakistan?"
  1. Who composed the national anthem of Pakistan?
  2. What is the religion of this person?

Now decompose: "{question}"

Respond ONLY with a JSON object:
  "subproblems": list of objects with "id" (int) and "text" (string)
  "is_simple": boolean"""

_BRANCHING_PROMPT = """You are an expert at decomposing branching questions.

A branching question has two or more independent "roots" — facts that can each be looked
up without depending on the other — which then converge at a merge step that needs BOTH
(or all) of their answers together. A root can itself be a short chain of 1-2 steps if it
has its own internal dependency. After the merge step, the chain may continue with one or
more further sequential steps.

Rules:
1. Identify every independent root first. Resolve each root fully (as its own mini-chain
   if needed) before the merge step.
2. Do not merge two roots into one subproblem — each root's own lookups stay separate
   until the step that actually needs both.
3. The merge step must explicitly combine the answers of the roots it depends on
   (e.g., "What lies between this country and this boundary?", "Which of this university
   and this university has more national championships?").
4. If the merge step's result is itself an input to further lookups, continue the chain
   with additional subproblems after the merge, one fact per step.
5. Each subproblem asks for exactly one fact or one merge/comparison.
6. Minimize total number of subproblems — do not add roots or steps that are not
   strictly necessary.

Examples:
- "What was the form of the language Auctor is in, used in the era of the Frankish king
   who created the Holy Roman Empire, later known as?"
  1. In what language is Auctor?
  2. Who was the Frankish king who created the Holy Roman Empire?
  3. What was the form of this language, in this king's era, later known as?

- "When was the region immediately north of the region where Israel is located and the
   location of the Battle of Qurah and Umm al Maradim created?"
  1. What region is Israel located in?
  2. Where was the Battle of Qurah and Umm al Maradim?
  3. What region lies immediately north of these two regions?
  4. When was this region created?

- "How were the people from whom new coins were a proclamation of independence by the
   Somali Muslim Ajuran Empire expelled from the country between Thailand and A Lim's
   country?"
  1. What is A Lim's country?
  2. What natural boundary lies between Thailand and this country?
  3. From whom did new coins represent a proclamation of independence by the Ajuran Empire?
  4. How were these people expelled from this boundary?

Now decompose: "{question}"

Respond ONLY with a JSON object:
  "subproblems": list of objects with "id" (int) and "text" (string)
  "is_simple": boolean"""

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
        prompt = _TYPE_PROMPTS.get(q_type, _CHAIN_PROMPT).format(question=question)

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
