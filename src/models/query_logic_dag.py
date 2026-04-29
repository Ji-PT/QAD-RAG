import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from src.utils.utils import get_response_with_retry, fix_json_response


@dataclass
class SubproblemNode:
    """
    Node in Query Logic Dependency Graph.

    id:
        Node id.
    text:
        Natural-language subproblem text.
    """
    id: int
    text: str


@dataclass
class DependencyEdge:
    """
    Directed logical dependency edge.

    prerequisite_id -> dependent_id

    Meaning:
        prerequisite_id must be solved before dependent_id.
    """
    prerequisite_id: int
    dependent_id: int
    reason: str = ""


@dataclass
class QueryLogicDAG:
    """
    Explicit Query Logic Dependency Graph G=(V,E).

    V:
        Node set.
    E:
        Directed edge set.
    """
    V: Dict[int, SubproblemNode]
    E: List[DependencyEdge]

    parents: Dict[int, Set[int]] = field(default_factory=dict)
    children: Dict[int, Set[int]] = field(default_factory=dict)

    # Filled later by topological-sort / rank module.
    topological_order: List[int] = field(default_factory=list)
    ranks: Dict[int, int] = field(default_factory=dict)

    def rebuild_indexes(self) -> None:
        """Build parent/child lookup tables from edges."""
        self.parents = {node_id: set() for node_id in self.V}
        self.children = {node_id: set() for node_id in self.V}

        for edge in self.E:
            self.children[edge.prerequisite_id].add(edge.dependent_id)
            self.parents[edge.dependent_id].add(edge.prerequisite_id)

    def get_parents(self, node_id: int) -> List[SubproblemNode]:
        """Return parent nodes of a given node."""
        return [
            self.V[parent_id]
            for parent_id in self.parents.get(node_id, set())
        ]

    def get_children(self, node_id: int) -> List[SubproblemNode]:
        """Return child nodes of a given node."""
        return [
            self.V[child_id]
            for child_id in self.children.get(node_id, set())
        ]

    def node_texts_in_id_order(self) -> List[str]:
        """Temporary compatibility helper for old _topological_sort()."""
        return [
            self.V[node_id].text
            for node_id in sorted(self.V.keys())
        ]

    def to_legacy_dependency_pairs(self) -> List[Tuple[int, int]]:
        """
        Convert new edge format back to old repo format.

        New format:
            prerequisite_id -> dependent_id

        Old repo format:
            (dependent_idx, dependency_idx)

        Example:
            new: 0 -> 2
            old: (2, 0)
        """
        return [
            (edge.dependent_id, edge.prerequisite_id)
            for edge in self.E
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serializable format for logging/evaluation."""
        return {
            "nodes": {
                node_id: {
                    "id": node.id,
                    "text": node.text,
                }
                for node_id, node in self.V.items()
            },
            "edges": [
                {
                    "prerequisite_id": edge.prerequisite_id,
                    "dependent_id": edge.dependent_id,
                    "reason": edge.reason,
                }
                for edge in self.E
            ],
            "parents": {
                node_id: sorted(list(parent_ids))
                for node_id, parent_ids in self.parents.items()
            },
            "children": {
                node_id: sorted(list(child_ids))
                for node_id, child_ids in self.children.items()
            },
            "topological_order": self.topological_order,
            "ranks": self.ranks,
        }


class QueryLogicDAGBuilder:
    """
    Builder for explicit Query Logic Dependency Graph G=(V,E).

    This class does not perform retrieval.
    It only builds and preserves the graph structure.
    """

    def construct_from_dependency_texts(
        self,
        question: str,
        dependencies: List[str],
    ) -> QueryLogicDAG:
        """
        Transitional constructor.

        Current repo still outputs `dependencies`.
        We temporarily treat each dependency string as a subproblem-like node.

        Later, this should be replaced by `construct_from_subproblems()`.
        """
        nodes = self._build_nodes_from_texts(dependencies)

        raw_edges = self._infer_dependency_edges(
            question=question,
            nodes=nodes,
        )

        edges = self._validate_dependency_edges(
            nodes=nodes,
            edges=raw_edges,
        )

        dag = QueryLogicDAG(V=nodes, E=edges)
        dag.rebuild_indexes()

        return dag

    def construct_from_subproblems(
        self,
        question: str,
        subproblems: List[Dict[str, Any]],
    ) -> QueryLogicDAG:
        """
        Final-paper-aligned constructor.

        Use this when the query decomposition module outputs:
        [
            {"id": 0, "text": "..."},
            {"id": 1, "text": "..."}
        ]
        """
        nodes = self._build_nodes_from_subproblems(subproblems)

        raw_edges = self._infer_dependency_edges(
            question=question,
            nodes=nodes,
        )

        edges = self._validate_dependency_edges(
            nodes=nodes,
            edges=raw_edges,
        )

        dag = QueryLogicDAG(V=nodes, E=edges)
        dag.rebuild_indexes()

        return dag

    def _build_nodes_from_texts(
        self,
        dependency_texts: List[str],
    ) -> Dict[int, SubproblemNode]:
        nodes = {}

        for i, text in enumerate(dependency_texts):
            nodes[i] = SubproblemNode(id=i, text=text)

        return nodes

    def _build_nodes_from_subproblems(
        self,
        subproblems: List[Dict[str, Any]],
    ) -> Dict[int, SubproblemNode]:
        nodes = {}

        for i, sp in enumerate(subproblems):
            node_id = int(sp.get("id", i))
            text = sp["text"]
            nodes[node_id] = SubproblemNode(id=node_id, text=text)

        return nodes

    def _infer_dependency_edges(
        self,
        question: str,
        nodes: Dict[int, SubproblemNode],
    ) -> List[DependencyEdge]:
        """
        LLM-based DAG edge inference.

        Output direction:
            prerequisite_id -> dependent_id
        """
        nodes_payload = [
            {
                "id": node.id,
                "text": node.text,
            }
            for node in nodes.values()
        ]

        prompt = f"""
You are constructing the edge set E of a Query Logic Dependency Graph G=(V,E).

Original question:
{question}

Subproblem nodes V:
{json.dumps(nodes_payload, ensure_ascii=False, indent=2)}

Task:
Infer directed logical dependency edges among the subproblem nodes.

Definition:
An edge A -> B means:
Subproblem A must be solved before subproblem B can be solved.

Use this exact direction:
prerequisite_id -> dependent_id

Rules:
- Use only ids from the provided subproblem nodes.
- Do not create self-loops.
- Do not include independent nodes in edges.
- Prefer direct logical dependencies only.
- Do not add redundant transitive edges.
  For example, if 0 -> 1 and 1 -> 2, do not also add 0 -> 2 unless it is directly necessary.
- Each edge must include a short reason.
- If no logical dependencies exist, return an empty edge list.

Few-shot example:
Original question:
Which film was released earlier, Inception or Titanic?

Subproblem nodes V:
[
  {{"id": 0, "text": "Find the release date of Inception."}},
  {{"id": 1, "text": "Find the release date of Titanic."}},
  {{"id": 2, "text": "Compare the two release dates and determine which film was released earlier."}}
]

Output:
{{
  "edges": [
    {{
      "prerequisite_id": 0,
      "dependent_id": 2,
      "reason": "The release date of Inception is needed before comparing the two films."
    }},
    {{
      "prerequisite_id": 1,
      "dependent_id": 2,
      "reason": "The release date of Titanic is needed before comparing the two films."
    }}
  ]
}}

Return ONLY a JSON object with this schema:
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

        response = get_response_with_retry(prompt)
        response = response.strip()
        response = response.replace("```json", "").replace("```", "")

        result = fix_json_response(response)

        if result is None:
            return []

        raw_edges = result.get("edges", [])
        edges = []

        for edge in raw_edges:
            try:
                edges.append(
                    DependencyEdge(
                        prerequisite_id=int(edge["prerequisite_id"]),
                        dependent_id=int(edge["dependent_id"]),
                        reason=edge.get("reason", ""),
                    )
                )
            except Exception:
                continue

        return edges

    def _validate_dependency_edges(
        self,
        nodes: Dict[int, SubproblemNode],
        edges: List[DependencyEdge],
    ) -> List[DependencyEdge]:
        """
        Basic validation:
        - valid node ids
        - no self-loop
        - no duplicate edges

        Cycle validation should be handled by the topological-sort module.
        """
        node_ids = set(nodes.keys())
        validated = []
        seen = set()

        for edge in edges:
            pre = edge.prerequisite_id
            dep = edge.dependent_id

            if pre not in node_ids or dep not in node_ids:
                continue

            if pre == dep:
                continue

            key = (pre, dep)
            if key in seen:
                continue

            seen.add(key)
            validated.append(edge)

        return validated
