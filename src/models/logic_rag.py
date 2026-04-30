import json
import logging
import pdb
import time
from typing import List, Dict, Tuple, Any

from src.models.base_rag import BaseRAG
from src.models.query_logic_dag import QueryLogicDAGBuilder
from src.utils.utils import get_response_with_retry, fix_json_response
from colorama import Fore, Style, init

# Initialize colorama
init()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class LogicRAG(BaseRAG):
    
    def __init__(self, corpus_path: str = None, cache_dir: str = "./cache", filter_repeats: bool = False):
        """Initialize the LogicRAG system."""
        super().__init__(corpus_path, cache_dir)

        self.max_rounds = 3  # Default max rounds for iterative retrieval
        self.MODEL_NAME = "LogicRAG"
        self.filter_repeats = filter_repeats  # Option to filter repeated chunks across rounds

        # Builder for constructing an explicit Query Logic DAG G=(V,E)
        # from decomposed dependencies.
        self.dag_builder = QueryLogicDAGBuilder()

        # Store the most recently constructed Query Logic DAG for debugging,
        # evaluation, and downstream graph-based reasoning modules.
        self.last_query_logic_dag = None
    
    def set_max_rounds(self, max_rounds: int):
        """Set the maximum number of retrieval rounds."""
        self.max_rounds = max_rounds
    
    def refine_summary_with_context(
        self,
        question: str,
        new_contexts: List[str],
        current_summary: str = ""
    ) -> str:
        """
        Generate a new summary or refine an existing one based on newly retrieved contexts.
        
        Args:
            question: The original question
            new_contexts: Newly retrieved context chunks
            current_summary: Current information summary, if any
            
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
        Analyze whether the question can be answered with simple fact retrieval,
        before performing deeper dependency-aware reasoning.
        
        Args:
            question: The original question
            info_summary: Current information summary
            
        Returns:
            Dictionary with answerability analysis, missing information,
            current understanding, and logical dependencies
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

            # Clean up response to ensure it is valid JSON
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
            
            # Add default values for interpretability fields if missing
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

    def dependency_aware_rag(
        self,
        question: str,
        info_summary: str,
        dependencies: List[str],
        idx: int
    ) -> Dict:
        """
        Analyze whether the current information summary is sufficient to answer
        the original question, using the sorted dependencies as reasoning references.

        Args:
            question: The original question
            info_summary: Current accumulated information summary
            dependencies: Topologically sorted dependencies
            idx: Current dependency index

        Returns:
            Dictionary with can_answer and current_understanding
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

    def _sort_dependencies(self, dependencies: List[str], query) -> List[str]:
        """
        Legacy dependency sorting method.

        Given a list of dependencies and the original query, this method asks the LLM
        to infer dependency pairs, then applies graph-based topological sorting.

        This method is currently kept for backward compatibility and ablation testing.
        The main dependency flow now constructs an explicit Query Logic DAG before sorting.
        """
        # Step 1: generate dependency pairs by prompting the LLM
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

        # Step 2: use graph-based algorithm to sort the dependencies in topological order
        sorted_dependencies = self._topological_sort(dependencies, dependency_pairs)
        return sorted_dependencies

    @staticmethod
    def _topological_sort(dependencies: List[str], dependencies_pairs: List[Tuple[int, int]]) -> List[str]:
        """
        Use graph-based algorithm to sort dependencies in topological order.

        Args:
            dependencies: List of dependency texts
            dependencies_pairs: Legacy dependency pairs in the format
                (dependent_idx, dependency_idx)

        Returns:
            List of dependency texts sorted in topological order

        Note:
            This is a legacy compatibility sorter. The new Query Logic DAG uses
            edges in the direction prerequisite_id -> dependent_id, so DAG edges
            must be converted to legacy pairs before calling this method.
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
        Retrieve top_k unique chunks not in retrieved_chunks_set.
        If not enough unique chunks are found, expand the retrieval window.
        """
        all_results = self.retrieve(query)
        unique_results = []
        idx = self.top_k

        # If not enough unique chunks are found in top_k, keep expanding
        while len(unique_results) < self.top_k and idx <= len(self.corpus):
            # Expand retrieval window
            all_results = self.retrieve(query) if idx == self.top_k else self._retrieve_top_n(query, idx)
            unique_results = [chunk for chunk in all_results if chunk not in retrieved_chunks_set]
            idx += self.top_k

        return unique_results[:self.top_k]

    def _retrieve_top_n(self, query: str, n: int) -> list:
        """Retrieve top-n results for a query. Helper for filtering repeated chunks."""
        # Temporarily override top_k
        old_top_k = self.top_k
        self.top_k = n
        results = self.retrieve(query)
        self.top_k = old_top_k
        return results

    def answer_question(self, question: str) -> Tuple[str, List[str], int]:
        """
        Answer a question using LogicRAG.

        The process consists of:
        1. Warm-up retrieval and summarization
        2. Answerability analysis
        3. Query Logic DAG construction if deeper reasoning is required
        4. Dependency-aware iterative retrieval
        5. Final answer generation

        Args:
            question: The input question

        Returns:
            answer: Generated final answer
            last_contexts: Last retrieved contexts for evaluation
            round_count: Number of iterative retrieval rounds used
        """
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

        last_contexts = new_contexts  # Save current contexts for evaluation
        
        # Generate or refine information summary with warm-up contexts
        info_summary = self.refine_summary_with_context(
            question, 
            new_contexts, 
            info_summary
        )

        analysis = self.warm_up_analysis(question, info_summary)

        if analysis["can_answer"]:
            # In this case, the question can be answered with simple fact retrieval,
            # without dependency-aware reasoning.
            print("Warm-up analysis indicate the question can be answered with simple fact retrieval, without any dependency analysis.")
            answer = self.generate_answer(question, info_summary)

            # Reset dependency analysis history for simple questions
            self.last_dependency_analysis = []
            return answer, last_contexts, round_count

        else:
            logger.info("Warm-up analysis indicate the requirement of deeper reasoning-enhanced RAG. Now perform analysis with logical dependency graph.")
            logger.info(f"Dependencies: {', '.join(analysis.get('dependencies', []))}")

            # Construct an explicit Query Logic DAG G=(V,E) from the dependencies.
            # Each dependency becomes a node, and LLM-inferred logical prerequisites
            # become directed edges.
            #
            # Edge direction:
            #   prerequisite_id -> dependent_id
            dag = self.dag_builder.construct_from_dependency_texts(
                question=question,
                dependencies=analysis["dependencies"],
            )

            # Keep the DAG for later inspection, debugging, and evaluation.
            self.last_query_logic_dag = dag

            dependency_analysis_history.append({
                "query_logic_dag": dag.to_dict()
            })
            logger.info(f"Constructed Query Logic DAG: {dag.to_dict()}\n\n")

            # Temporary compatibility path for the existing retrieval loop.
            #
            # The new DAG stores edges as:
            #   prerequisite_id -> dependent_id
            #
            # The legacy _topological_sort() expects pairs as:
            #   (dependent_idx, dependency_idx)
            #
            # Therefore, we convert the DAG back into legacy dependency pairs.
            # This block should later be replaced by a cycle-safe DAG topological
            # sorter once the graph validation/sorting module is integrated.
            sorted_dependencies = self._topological_sort(
                dag.node_texts_in_id_order(),
                dag.to_legacy_dependency_pairs(),
            )

            dependency_analysis_history[-1]["sorted_dependencies"] = sorted_dependencies
            logger.info(f"Sorted dependencies: {sorted_dependencies}\n\n")

        #===============================================
        #== Stage 2: agentic iterative retrieval ==
        idx = 0  # Track the current dependency index

        while round_count < self.max_rounds and idx < len(sorted_dependencies):
            round_count += 1
            
            current_query = sorted_dependencies[idx]

            if self.filter_repeats:
                new_contexts = self._retrieve_with_filter(current_query, retrieved_chunks_set)
                for chunk in new_contexts:
                    retrieved_chunks_set.add(chunk)
            else:
                new_contexts = self.retrieve(current_query)

            last_contexts = new_contexts  # Save current contexts for evaluation
            
            # Generate or refine information summary with new contexts
            info_summary = self.refine_summary_with_context(
                question, 
                new_contexts, 
                info_summary
            )
            
            logger.info(f"Agentic retrieval at round {round_count}")
            logger.info(f"current query: {current_query}")
            
            analysis = self.dependency_aware_rag(
                question,
                info_summary,
                sorted_dependencies,
                idx
            )

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

                # Return the last retrieved contexts for evaluation purposes
                return answer, last_contexts, round_count

            else:
                idx += 1
        
        # If max rounds reached, generate best possible answer
        logger.info(f"Reached maximum rounds ({self.max_rounds}). Generating final answer...")
        answer = self.generate_answer(question, info_summary)

        # Store dependency analysis history for evaluation access
        self.last_dependency_analysis = dependency_analysis_history
        return answer, last_contexts, round_count