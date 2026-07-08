"""
decompose_query() 프롬프트 실험용 LogicRAG 서브클래스.

base 코드(logic_rag.py)를 수정하지 않고 decompose_query()만 override한다.

실험 목록:
  LogicRAGExpEntityCoT   — 실험 1: pivot entity 먼저 식별 후 분해 (CoT)
  LogicRAGExpSelfVerify  — 실험 2: 분해 후 각 step 자기검증 + 분할
  LogicRAGExpHopCount    — 실험 3: hop count 사전 추정 후 분해
  LogicRAGExpDependencyAwareDecomp
                         — 실험 4: decomposition 단계에서 draft dependency 함께 예측

실행은 run_decomp_experiments.py 참고.
"""

import logging
import re
from typing import Any, Dict, List

from src.models.logic_rag import LogicRAG, QUERY_DECOMPOSITION_FEW_SHOT_EXAMPLES
from src.utils.utils import get_response_with_retry, fix_json_response

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 실험 4: Dependency-aware decomposition (B1)
# ──────────────────────────────────────────────────────────────────────────────

class LogicRAGExpDependencyAwareDecomp(LogicRAG):
    """
    실험 4 — subproblem과 draft dependency를 함께 생성한다.

    B1 범위:
    - dependencies는 저장/분석용으로만 반환한다.
    - DAG builder, edge inference, rank 계산에는 사용하지 않는다.
    - 기존 decomposition metric은 subproblems만으로 계산한다.
    """

    def decompose_query(self, question: str) -> Dict[str, Any]:
        try:
            prompt = f"""You are an expert at decomposing complex questions into smaller, logically ordered subproblems.

    Given a question, decompose it into the minimum number of subproblems needed to answer it.

    Rules:
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
        12. Read the full question before decomposing. If the question contains a compound structure where the result of the first lookup is used in a different context for the second lookup, both steps are required. Do not stop at the first entity lookup and omit the contextual second step.

    After creating the subproblems, also draft direct dependency edges among them for analysis.
    A dependency edge A -> B means B cannot be answered without the result of A.
    Do not create dependencies from subproblem order or id order alone.
    Do not connect independent subproblems, even if both are needed for a later comparison, aggregation, judgment, or relation.
    For comparison, aggregation, judgment, or relation subproblems, connect each required input subproblem directly to that final subproblem.
    Do not add redundant transitive dependencies.

    Here are some examples:
    {QUERY_DECOMPOSITION_FEW_SHOT_EXAMPLES}

    Now decompose the following question:
    Question: "{question}"

    Please format your response as a JSON object with these keys:

    * "subproblems": list of objects, each with "id" (int) and "text" (string)
    * "dependencies": list of objects, each with "prerequisite_id" (int), "dependent_id" (int), and "reason" (string)
    * "is_simple": boolean

    Respond ONLY with the JSON object, no additional text."""

            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            result = fix_json_response(response)

            if result is None:
                return {
                    "subproblems": [{"id": 0, "text": question}],
                    "dependencies": [],
                    "is_simple": True,
                }

            if (
                "subproblems" not in result
                or not isinstance(result["subproblems"], list)
                or len(result["subproblems"]) == 0
            ):
                result["subproblems"] = [{"id": 0, "text": question}]

            if "is_simple" not in result:
                result["is_simple"] = len(result["subproblems"]) <= 1

            dependencies = self._normalize_dependencies(
                result.get("dependencies", []),
                result["subproblems"],
            )

            return {
                "subproblems": result["subproblems"],
                "dependencies": dependencies,
                "is_simple": result["is_simple"],
            }

        except Exception as e:
            logger.error(f"LogicRAGExpDependencyAwareDecomp.decompose_query error: {e}")
            return {
                "subproblems": [{"id": 0, "text": question}],
                "dependencies": [],
                "is_simple": True,
            }

    @staticmethod
    def _normalize_dependencies(
        raw_dependencies: Any,
        subproblems: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not isinstance(raw_dependencies, list):
            return []

        valid_ids = set()
        for subproblem in subproblems:
            if not isinstance(subproblem, dict):
                continue
            try:
                node_id = int(subproblem.get("id"))
            except (TypeError, ValueError):
                continue
            valid_ids.add(node_id)

        normalized: List[Dict[str, Any]] = []
        seen_edges = set()

        for raw_dependency in raw_dependencies:
            if not isinstance(raw_dependency, dict):
                continue

            try:
                prerequisite_id = int(raw_dependency.get("prerequisite_id"))
                dependent_id = int(raw_dependency.get("dependent_id"))
            except (TypeError, ValueError):
                continue

            if prerequisite_id == dependent_id:
                continue

            if prerequisite_id not in valid_ids or dependent_id not in valid_ids:
                continue

            edge_key = (prerequisite_id, dependent_id)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            normalized.append({
                "prerequisite_id": prerequisite_id,
                "dependent_id": dependent_id,
                "reason": str(raw_dependency.get("reason", "")).strip(),
            })

        return normalized


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
    추가 API 호출: 검증 대상 step 수만큼 (소유 속성 패턴 step에만 적용)
    검증은 단일 pass (무한 루프 방지)

    수정 (v2):
    - Fix 1: "X's Y" 소유 속성 패턴이 있는 step에만 검증 적용 (전체 적용 금지)
    - Fix 2: 검증 프롬프트 재설계 — split 불필요 예시 강화, 보수적 판단 유도
    - Fix 3: confidence 필드 추가 — high일 때만 실제 분할
    """

    # "X's Y"가 직접 답의 대상이 아니라 다른 lookup의 입력으로 쓰이는 step만 검증 대상
    # "What is X's Y?" 형태는 이미 단일 lookup이므로 제외
    _POSSESSIVE_AS_INPUT = re.compile(r"\b\w{2,}'s\s+\w+")
    _DIRECT_PROPERTY_LOOKUP = re.compile(
        r"^(What|Who|Which)\s+(is|was|are|were)\s+[\w\s]+'s\s+\w+[\w\s]*\??$",
        re.IGNORECASE,
    )

    def decompose_query(self, question: str) -> Dict[str, Any]:
        # 1단계: base 클래스로 초기 분해
        initial = super().decompose_query(question)

        if initial.get("is_simple") or not initial.get("subproblems"):
            return initial

        # 2단계: 소유 속성이 다른 lookup의 입력으로 쓰이는 step만 선별 검증
        # "What is X's Y?" 형태는 이미 단일 lookup이므로 제외
        verified: List[Dict] = []
        for step in initial["subproblems"]:
            has_possessive = self._POSSESSIVE_AS_INPUT.search(step["text"])
            is_direct_lookup = self._DIRECT_PROPERTY_LOOKUP.match(step["text"])
            if has_possessive and not is_direct_lookup:
                check = self._verify_step(question, step["text"])
                # confidence == "high" 일 때만 분할
                if (
                    check.get("needs_split")
                    and check.get("confidence") == "high"
                    and len(check.get("steps", [])) == 2
                ):
                    verified.append({"text": check["steps"][0]})
                    verified.append({"text": check["steps"][1]})
                else:
                    verified.append({"text": step["text"]})
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
        """소유 속성 체인이 포함된 step이 실제로 분할이 필요한지 보수적으로 판단한다."""
        prompt = f"""You are checking whether a decomposition step contains a possessive property chain that requires splitting.

Original question: "{question}"
Step to check: "{step_text}"

A step needs splitting ONLY when it contains "X's Y" where Y is itself used as input to the next lookup —
meaning you must first find Y's value, and then use that value to answer another question.

This is a HIGH bar. Most steps do NOT need splitting. Default to "needs_split": false.

Examples of steps that do NOT need splitting (the vast majority):
- "What is the capital of France?"              → single lookup, NO SPLIT
- "Who is the mayor of Paris?"                  → single lookup, NO SPLIT
- "In which country was the inventor born?"     → single lookup, NO SPLIT
- "What is the population of this country?"     → single lookup, NO SPLIT
- "When did the explorer reach this city?"      → single lookup, NO SPLIT
- "Who directed this film?"                     → single lookup, NO SPLIT

Examples of steps that DO need splitting (rare — only when X's Y feeds another lookup):
- "Who wanted to reform John Kodwo Amissah's religion?"
  → Must find Amissah's religion first, THEN find who wanted to reform it.
  → SPLIT into: "What is Amissah's religion?" + "Who wanted to reform this religion?"
- "What is the only group larger than Mankatha's record label?"
  → Must find Mankatha's record label first, THEN find the larger group.
  → SPLIT into: "What is Mankatha's record label?" + "What is the only group larger than this label?"

Return ONLY a JSON object:
{{
  "needs_split": true or false,
  "confidence": "high" or "low",
  "steps": ["step A text", "step B text"]
}}
"steps" is required only when needs_split=true. "confidence" must be "high" only when you are certain splitting is needed."""

        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            result = fix_json_response(response)
            if not isinstance(result, dict):
                return {"needs_split": False, "confidence": "low"}
            return result
        except Exception as e:
            logger.error(f"_verify_step error: {e}")
            return {"needs_split": False, "confidence": "low"}


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
