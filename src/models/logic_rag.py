import json
import logging
import pdb
import time
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
QUERY_DECOMPOSITION_FEW_SHOT_EXAMPLES = """
Example 1:
Question: "Who is the mayor of the capital of France?"
Subproblems:
[
  "What is the capital of France?",
  "Who is the mayor of this capital city?"
]
Dependency pairs: [[1, 0]]

Example 2:
Question: "When was the director of 'Inception' born, and what award did the film win at the Oscars?"
Subproblems:
[
  "Who directed the film 'Inception'?",
  "When was this director born?",
  "What award did 'Inception' win at the Oscars?"
]
Dependency pairs: [[1, 0]]

Example 3:
Question: "What is the population of the country where the inventor of the telephone was born?"
Subproblems:
[
  "Who invented the telephone?",
  "In which country was this inventor born?",
  "What is the population of this country?"
]
Dependency pairs: [[1, 0], [2, 1]]

Example 4:
Question: "What is the tallest building in Tokyo?"
Subproblems:
[
  "What is the tallest building in Tokyo?"
]
Dependency pairs: []
"""


class LogicRAG(BaseRAG):
    
    def __init__(self, corpus_path: str = None, cache_dir: str = "./cache", filter_repeats: bool = False):
        """Initialize the LogicRAG system."""
        super().__init__(corpus_path, cache_dir)
        self.max_rounds = 3  # Default max rounds for iterative retrieval
        self.MODEL_NAME = "LogicRAG"
        self.filter_repeats = filter_repeats  # Option to filter repeated chunks across rounds
    
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
    # warm_up_analysis()를 대체하며, subproblems와 dependency_pairs를 한 번의 LLM 호출로 추출
    def decompose_query(self, question: str) -> Dict:
        """
        Decompose the input query into subproblems and extract dependency pairs using few-shot prompting.
        subproblems and dependency_pairs are extracted in a single LLM call.

        Args:
            question: The original question

        Returns:
            Dictionary with:
                - "subproblems": List[str]
                - "dependency_pairs": List[List[int]]
                - "is_simple": bool
        """
        try:
            prompt = f"""You are an expert at decomposing complex questions into smaller, logically ordered subproblems.

Given a question, you must:
1. Decompose the question into a minimal set of subproblems.
2. Identify dependency pairs: if answering subproblem i requires the answer of subproblem j, output [i, j].
3. If the question is simple (single-hop, no decomposition needed), output a single subproblem identical to the original question with no dependency pairs.

Here are some examples:
{QUERY_DECOMPOSITION_FEW_SHOT_EXAMPLES}

Now decompose the following question:
Question: "{question}"

Please format your response as a JSON object with these keys:
- "subproblems": list of strings
- "dependency_pairs": list of [int, int] pairs
- "is_simple": boolean

Respond ONLY with the JSON object, no additional text."""

            response = get_response_with_retry(prompt)

            # Remove any markdown code block markers
            response = response.strip().replace('```json', '').replace('```', '')

            # Parse the cleaned response using fix_json_response
            result = fix_json_response(response)

            if result is None:
                return {"subproblems": [question], "dependency_pairs": [], "is_simple": True}

            # Validate required fields
            if "subproblems" not in result or not isinstance(result["subproblems"], list) or len(result["subproblems"]) == 0:
                result["subproblems"] = [question]
            if "dependency_pairs" not in result or not isinstance(result["dependency_pairs"], list):
                result["dependency_pairs"] = []
            if "is_simple" not in result:
                result["is_simple"] = len(result["subproblems"]) <= 1

            return result

        except Exception as e:
            logger.error(f"{Fore.RED}Error in decompose_query: {e}{Style.RESET_ALL}")
            return {"subproblems": [question], "dependency_pairs": [], "is_simple": True}

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

    def _sort_dependencies(self, dependencies: List[str], query,
                           dependency_pairs: List[List[int]] = None) -> List[Tuple]:  # [수정] dependency_pairs 파라미터 추가
        """
        given a list of dependencies and the original query,
        sort the dependencies in a topological order, that is solving a dependency A relies on the solution of the dependent dependency B,
        then B should be before A in the sorted string.

        Args:
            dependencies: List[str]
            query: str

            
        For example, if the question is "What is the mayor of the capital of France?",
        the input dependencies for this question are:
        - The capital of France
        - The mayor of this capital

        Then the output should be:
        - The capital of France
        - The mayor of this capital

        there are two steps to solve this problem:
        1. generate the dependency pairs that dependency A relies on dependency B
        2. use graph-based algorithm to sort the dependencies in a topological order

        For example, answering the question "What is the mayor of the capital of France?"
        the input dependencies are:
        - The capital of France
        - The mayor of this capital

        Then the dependency pairs are:
        - [(1, 0)]
        because the mayor of the capital of France relies on the capital of France

        Then the topological order is computed by the self._topological_sort function, which is a graph-based algorithm. The output is a list of indices of the dependencies in the topological order.
        In this case, the output is:
        [0, 1]

        The sorted dependencies are thus:
        - The capital of France
        - The mayor of this capital
        """

        # [수정] decompose_query()에서 이미 추출한 dependency_pairs가 있으면 LLM 호출을 건너뜀
        if dependency_pairs is None:
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
    def _topological_sort(dependencies: List[str], dependencies_pairs: List[Tuple[int, int]]) -> List[str]:
        """
        Use graph-based algorithm to sort the dependencies in a topological order.
        Args:
            dependencies: List[str]
            dependencies_pairs: List[Tuple[int, int]]
        Returns:
            List[str]
        """
        graph = {dep: [] for dep in dependencies}
        
        for dependent_idx, dependency_idx in dependencies_pairs:
            if dependent_idx < len(dependencies) and dependency_idx < len(dependencies):
                dependent = dependencies[dependent_idx]
                dependency = dependencies[dependency_idx]
                graph[dependency].append(dependent)  # dependency -> dependent
        
        visited = set()
        stack = []
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph[node]:
                dfs(neighbor)
            stack.append(node)

        for node in graph:
            if node not in visited:
                dfs(node)
        
        return stack[::-1]

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

        # [수정] warm_up_analysis() 대신 decompose_query()를 사용
        # subproblem 분해 + dependency_pairs 추출을 하나의 LLM 호출로 처리 (논문 Section 3.2)
        decomposition = self.decompose_query(question)

        if decomposition["is_simple"]:
            # In this case, the question can be answered with simple fact retrieval, without any dependency analysis
            print(f"Query decomposition indicates a simple single-hop question. Answering directly.")
            answer = self.generate_answer(question, info_summary)
            # Reset dependency analysis history for simple questions
            self.last_dependency_analysis = []
            return answer, last_contexts, round_count
        else:
            logger.info(f"Query decomposition result: {len(decomposition['subproblems'])} subproblems detected.")
            logger.info(f"Subproblems: {decomposition['subproblems']}")

            # sort the dependencies, by first constructing the dependency graphs, then use topological sort to get the sorted dependencies
            # [수정] decompose_query()에서 받은 dependency_pairs를 직접 전달하여 LLM 중복 호출 방지
            sorted_dependencies = self._sort_dependencies(
                decomposition["subproblems"],
                question,
                dependency_pairs=decomposition["dependency_pairs"]
            )
            dependency_analysis_history.append({"sorted_dependencies": sorted_dependencies})
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
