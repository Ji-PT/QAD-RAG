"""
Fixed-DAG LogicRAG variants for execution-only edge-method comparison.

The classes in this module keep decomposition subproblems and DAG edges fixed
from a prior analysis JSON. They do not call the decomposition LLM or the DAG
edge-inference LLM.
"""

from copy import deepcopy
from typing import Any, Dict, List

from src.models.logic_rag import LogicRAG
from src.models.query_logic_dag import (
    DependencyEdge,
    QueryLogicDAG,
    QueryLogicDAGBuilder,
)


class FixedEdgeDAGBuilder(QueryLogicDAGBuilder):
    def __init__(self, fixed_edges: List[Dict[str, Any]]) -> None:
        super().__init__()
        self._fixed_edges = deepcopy(fixed_edges)

    def construct_from_subproblems(
        self,
        question: str,
        subproblems: List[Dict[str, Any]],
    ) -> QueryLogicDAG:
        nodes = self._build_nodes_from_subproblems(subproblems)
        raw_edges = [
            DependencyEdge(
                prerequisite_id=int(edge["prerequisite_id"]),
                dependent_id=int(edge["dependent_id"]),
                reason=str(edge.get("reason", "")),
                metadata={"source": "comparison_json_injection"},
            )
            for edge in self._fixed_edges
        ]
        edges = self._validate_dependency_edges(nodes=nodes, edges=raw_edges)
        return self._build_dag(
            question=question,
            nodes=nodes,
            edges=edges,
            construction_source="fixed_edge_injection",
            edge_inference_status="fixed_from_comparison_json",
        )


class LogicRAGFixedDAG(LogicRAG):
    def __init__(
        self,
        corpus_path: str,
        fixed_subproblems: List[Dict[str, Any]],
        fixed_edges: List[Dict[str, Any]],
        cache_dir: str = "./cache",
    ) -> None:
        super().__init__(corpus_path, cache_dir)
        self._fixed_subproblems = deepcopy(fixed_subproblems)
        self.dag_builder = FixedEdgeDAGBuilder(fixed_edges)
        self.max_dag_repair_attempts = 0
        self.dag_cycle_policy = "raise"

    def decompose_query(self, question: str) -> Dict[str, Any]:
        return {
            "subproblems": deepcopy(self._fixed_subproblems),
            "is_simple": len(self._fixed_subproblems) <= 1,
        }
