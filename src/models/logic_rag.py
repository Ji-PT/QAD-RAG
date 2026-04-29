import copy
import json
import logging
from typing import List, Dict, Tuple, Any
from src.models.base_rag import BaseRAG
from src.utils.utils import get_response_with_retry, fix_json_response
from colorama import Fore, Style, init

# Initialize colorama
init()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# [추가] Few-shot 예시 상수 (논문 Section 3.2)
# decompose_query()의 프롬프트에서 참조
# subproblems 형식: {"id", "text"} → QueryLogicDAGBuilder 입력 형식에 맞춤
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

        self.max_rounds = 3  # agentic iterative retrieval의 최대 round 수
        self.MODEL_NAME = "LogicRAG"
        self.filter_repeats = filter_repeats  # Option to filter repeated chunks across rounds

        # Query decomposition 담당자가 만든 decompose_query()의 결과인 subproblems를 Query Logic DAG G=(V,E)로 변환하기 위한 Builder.
        # 여기서는 decomposition["subproblems"]를 DAG Builder에 연결하는 역할만 한다.     
        self.dag_builder = QueryLogicDAGBuilder()

        # 마지막으로 생성된 Query Logic DAG를 평가/디버깅용으로 저장한다.
        self.last_query_logic_dag = None    

        # DAG verification / repair settings
        self.max_dag_repair_attempts = 1
        self.dag_cycle_policy = "raise"  # "raise" or "fallback" 

    def set_max_rounds(self, max_rounds: int):
        """Set the maximum number of retrieval rounds."""
        self.max_rounds = max_rounds
    
    def refine_summary_with_context(self, question: str, new_contexts: List[str], 
                                  current_summary: str = "") -> str:
        """
        Generate a new summary or refine an existing one based on newly retrieved contexts.
        
        Args:
            question: The original question
            new_contexts: Newly retrieved context chunks
            current_summary: Current information summary (if any)
            
        Returns:
            A concise summary of all relevant information so far
        """
        try:
            context_text = "\n".join(new_contexts)
            
            if not current_summary:
                # Generate initial summary
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
                # Refine existing summary with new information
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
            
            summary = get_response_with_retry(prompt)
            return summary
            
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating/refining summary: {e}{Style.RESET_ALL}")
            # If error occurs, concatenate current summary with new contexts as fallback
            if current_summary:
                return f"{current_summary}\n\nNew information:\n{context_text}"
            return context_text
    
    def warm_up_analysis(self, question: str, info_summary: str) -> Dict:
        """
        This is a warm-up analysis, which is used to analyze if the question can be answered with simple fact retrieval, without any dependency analysis.
        
        Args:
            question: The original question
            info_summary: Current information summary
            
        Returns:
            Dictionary with analysis results
        """
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
            
            # Clean up response to ensure it's valid JSON
            response = response.strip()
            
            # Remove any markdown code block markers
            response = response.replace('```json', '').replace('```', '')
            
            # Parse the cleaned response using fix_json_response
            result = fix_json_response(response)
            if result is None:
                return {
                    "can_answer": True,
                    "missing_info": "",
                    "subquery": question,
                    "current_understanding": "Failed to parse reflection response.",
                    "dependencies": ["Information relevant to the question"],
                    "missing_reason": "Parse error occurred"
                }
            
            # Validate required fields
            required_fields = ["can_answer", "missing_info", "subquery", "current_understanding"]
            if not all(field in result for field in required_fields):
                logger.error(f"{Fore.RED}Missing required fields in response: {response}{Style.RESET_ALL}")
                raise ValueError("Missing required fields")
            
            # Add default values for new interpretability fields if missing
            if "dependencies" not in result:
                result["dependencies"] = ["Information relevant to the question"]
            if "missing_reason" not in result:
                result["missing_reason"] = "Additional context needed" if not result["can_answer"] else "No missing information"
            
            # Ensure boolean type for can_answer
            result["can_answer"] = bool(result["can_answer"])
            
            # Ensure non-empty subquery
            if not result["subquery"]:
                result["subquery"] = question
            
            return result
                
        except Exception as e:
            logger.error(f"{Fore.RED}Error in analyze_dependency_graph: {e}{Style.RESET_ALL}")
            return {
                "can_answer": True,
                "missing_info": "",
                "subquery": question,
                "current_understanding": f"Error during analysis: {str(e)}",
                "dependencies": ["Information relevant to the question"],
                "missing_reason": "Analysis error occurred"
            }

    # [추가] 논문 Section 3.2 - Query Decomposition Prompting
    # subproblem 분해 + Few-shot prompting을 하나의 Task로 합침
    # 출력된 subproblems는 QueryLogicDAGBuilder.construct_from_subproblems()의 input으로 전달됨
    def decompose_query(self, question: str) -> Dict:
        """
        Decompose the input query into subproblems using few-shot prompting.
        Output subproblems are passed to QueryLogicDAGBuilder.construct_from_subproblems().

        Args:
            question: The original question

        Returns:
            Dictionary with:
                - "subproblems": List[Dict]  # [{"id": int, "text": str}, ...]
                - "is_simple": bool
        """
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

            # Remove any markdown code block markers
            response = response.strip().replace('```json', '').replace('```', '')

            # Parse the cleaned response using fix_json_response
            result = fix_json_response(response)

            if result is None:
                return {"subproblems": [{"id": 0, "text": question}], "is_simple": True}

            # Validate required fields
            if "subproblems" not in result or not isinstance(result["subproblems"], list) or len(result["subproblems"]) == 0:
                result["subproblems"] = [{"id": 0, "text": question}]
            if "is_simple" not in result:
                result["is_simple"] = len(result["subproblems"]) <= 1

            return result

        except Exception as e:
            logger.error(f"{Fore.RED}Error in decompose_query: {e}{Style.RESET_ALL}")
            return {"subproblems": [{"id": 0, "text": question}], "is_simple": True}

    def dependency_aware_rag(self, question: str, info_summary: str, dependencies: List[str], idx: int) -> str:
        """
        similar to "self.analyze_dependency_graph" that analyzes whether the current information summary is sufficient to answer the question,
        this function analyzes whether the current information summary is sufficient to answer the question with the decomposed dependencies as references.

        And the function will answer whether the question can be answered, and if not, it will update the current query with dependencies as references.

        Args:
            question: str
            info_summary: str
            dependencies: List[str]
            idx: int
        """

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
            result = fix_json_response(response)
            return result
        except Exception as e:
            logger.error(f"{Fore.RED}Error in dependency_aware_rag: {e}{Style.RESET_ALL}")
            return {
                "can_answer": True,
                "current_understanding": f"Error during analysis: {str(e)}",
            }

    def generate_answer(self, question: str, info_summary: str) -> str:
        """Generate final answer based on the information summary."""
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
        """
        Legacy dependency sorting method.

        Given a list of dependencies and the original query, this method asks the LLM
        to infer dependency pairs, then applies graph-based topological sorting.

        This method is kept for backward compatibility and ablation testing.
        The main path after merge should construct an explicit Query Logic DAG
        from decomposition["subproblems"] before sorting.
        """

        # Step 1: generate the dependency pairs by prompting LLMs
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

        # Step 2: use graph-based algorithm to sort the dependencies in a topological order
        sorted_dependencies = self._topological_sort(dependencies, dependency_pairs)
        return sorted_dependencies
    
    @staticmethod
    def _get_dag_node_edge_keys(dag_dict: Dict[str, Any]) -> Tuple[str, str]:
        """
        DAG dict에서 node field와 edge field 이름 찾기
        지원 형식:
        1. {"nodes": ..., "edges": ...}
        2. {"V": ..., "E": ...}
        """
        if "nodes" in dag_dict and "edges" in dag_dict:
            return "nodes", "edges"

        if "V" in dag_dict and "E" in dag_dict:
            return "V", "E"

        raise ValueError("DAG dict must have either nodes/edges or V/E.")


    @staticmethod
    def _nodes_payload_from_dag_dict(dag_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        LLM repair prompt용 node 리스트 생성
        """
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
        """
        LLM이 edge를 수정하면 parents / children 다시 계산

        verify_non_cyclicity.py는 parents / children을 직접 쓰지 않지만,
        logging/history의 DAG 일관성을 위해 갱신
        """
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
        """
        cycle 있는 DAG → LLM에게 고쳐달라고 요청

        - 이 함수에서는 DAG 검증을 하지 않고, repaired edge set을 포함한 DAG dict만 반환한다.
        - 검증은 _verify_sort_dependencies_with_repair()에서 다시 수행
        """
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
        """
        DAG 검증 + topological sort + cycle repair orchestration

        1. verify_non_cyclicity로 DAG 검증
        2. cycle 없으면 sorted_dependencies 반환
        3. cycle 있으면 LLM에게 targeted repair 요청
        4. repaired DAG를 다시 verify_non_cyclicity로 검증
        5. 그래도 cycle이면 policy에 따라 raise or fallback
        """
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
        """
        Retrieve top_k unique chunks not in retrieved_chunks_set. If not enough unique chunks, return as many as possible.
        """
        all_results = self.retrieve(query)
        unique_results = []
        idx = self.top_k
        # If not enough unique in top_k, keep expanding
        while len(unique_results) < self.top_k and idx <= len(self.corpus):
            # Expand retrieval window
            all_results = self.retrieve(query) if idx == self.top_k else self._retrieve_top_n(query, idx)
            unique_results = [chunk for chunk in all_results if chunk not in retrieved_chunks_set]
            idx += self.top_k
        return unique_results[:self.top_k]

    def _retrieve_top_n(self, query: str, n: int) -> list:
        """Retrieve top-n results for a query (helper for filtering)."""
        # Temporarily override top_k
        old_top_k = self.top_k
        self.top_k = n
        results = self.retrieve(query)
        self.top_k = old_top_k
        return results

    def answer_question(self, question: str) -> Tuple[str, List[str], int]:

        info_summary = "" 
        round_count = 0
        current_query = question
        retrieval_history = []
        last_contexts = []  
        dependency_analysis_history = []  
        retrieved_chunks_set = set() if self.filter_repeats else None  # Track retrieved chunks if filtering
        
        print(f"\n\n{Fore.CYAN}{self.MODEL_NAME} answering: {question}{Style.RESET_ALL}\n\n")
        
        #===============================================
        #== Stage 1: warm up retrieval ==
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

        # [수정] warm_up_analysis() 대신 decompose_query()를 사용 (논문 Section 3.2)
        # subproblems는 이후 QueryLogicDAGBuilder.construct_from_subproblems()의 input으로 전달됨
        decomposition = self.decompose_query(question)
        # Query decomposition 담당자가 dev 브랜치에 추가한 decompose_query()를 사용한다.
       
        if decomposition["is_simple"]:
            # In this case, the question can be answered with simple fact retrieval, without any dependency analysis
            print("Query decomposition indicates a simple single-hop question. Answering directly.")
            answer = self.generate_answer(question, info_summary)
            # Reset dependency analysis history for simple questions
            self.last_dependency_analysis = []
            self.last_query_logic_dag = None
            return answer, last_contexts, round_count
        else:
            logger.info(f"Query decomposition result: {len(decomposition['subproblems'])} subproblems detected.")
            logger.info(f"Subproblems: {decomposition['subproblems']}")

            # Query decompif decomposition["is_simple"]:Query decomposition  결과 P를 Query Logic DAG G=(V,E)로 변환한다.
            #
            # 논문 대응:
            #   - 입력 query Q를 subproblem 집합 P로 분해한다.
            #   - 각 subproblem p_i는 DAG의 node v_i가 된다.
            #   - subproblem 사이의 logical dependency는 DAG의 edge E가 된다.
            #   - edge는 QueryLogicDAGBuilder 내부에서 logical precedence 기준으로 추론한다.
            dag = self.dag_builder.construct_from_subproblems(
                question=question,
                subproblems=decomposition["subproblems"],
            )

            # 생성된 DAG를 평가/디버깅용으로 저장한다.
            self.last_query_logic_dag = dag

            dependency_analysis_history.append({
                "query_logic_dag": dag.to_dict()
            })
            logger.info(f"Constructed Query Logic DAG: {dag.to_dict()}\n\n")

            # 기존 retrieval loop와 연결하기 위한 임시 호환 경로.
            #
            # QueryLogicDAG edge 방향:
            #   prerequisite_id -> dependent_id
            #
            # 기존 _topological_sort() 입력 형식:
            #   (dependent_idx, dependency_idx)
            #
            # 따라서 DAG edge를 기존 pair 형식으로 변환한 뒤 topological sort를 수행한다.
            sorted_dependencies = self._topological_sort(
                dag.node_texts_in_id_order(),
                dag.to_legacy_dependency_pairs(),
            )

            dependency_analysis_history[-1]["sorted_dependencies"] = sorted_dependencies
            logger.info(f"Sorted dependencies: {sorted_dependencies}\n\n")

        #===============================================
        #== Stage 2: agentic iterative retrieval ==
        idx = 0 # used to track the current dependency index

        while round_count < self.max_rounds and idx < len(sorted_dependencies):
            round_count += 1
            
            current_query = sorted_dependencies[idx]
            if self.filter_repeats:
                new_contexts = self._retrieve_with_filter(current_query, retrieved_chunks_set)
                for chunk in new_contexts:
                    retrieved_chunks_set.add(chunk)
            else:
                new_contexts = self.retrieve(current_query)
            last_contexts = new_contexts  # Save current contexts
            
            
            # Generate or refine information summary with new contexts
            info_summary = self.refine_summary_with_context(
                question, 
                new_contexts, 
                info_summary
            )
            
            logger.info(f"Agentic retrieval at round {round_count}")
            logger.info(f"current query: {current_query}")
            
            analysis = self.dependency_aware_rag(question, info_summary, sorted_dependencies, idx)

            retrieval_history.append({
                "round": round_count,
                "query": current_query,
                "contexts": new_contexts,
            }) 

            dependency_analysis_history.append({
                "round": round_count,
                "query": current_query,
                "analysis": analysis
            })

            if analysis["can_answer"]:
                # Generate and return final answer
                answer = self.generate_answer(question, info_summary)
                # Store dependency analysis history for evaluation access
                self.last_dependency_analysis = dependency_analysis_history
                # We return the last retrieved contexts for evaluation purposes
                return answer, last_contexts, round_count
            else:
                idx += 1
        
        # If max rounds reached, generate best possible answer
        logger.info(f"Reached maximum rounds ({self.max_rounds}). Generating final answer...")
        answer = self.generate_answer(question, info_summary)
        # Store dependency analysis history for evaluation access
        self.last_dependency_analysis = dependency_analysis_history
        return answer, last_contexts, round_count