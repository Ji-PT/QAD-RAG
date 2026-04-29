"""
** non-cyclicity verification 유틸리티 **

- dependency_pairs를 검증하고, 생성된 의존성 관계가 순환하지 않는지 확인
- 3-state DFS로 cycle을 탐지
- 정규화된 graph가 DAG인 경우 topological order를 계산
- fallback에 필요한 metadata를 명시적으로 제공

- [dependent_idx, dependency_idx] : dependency_idx -> dependent_idx
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple


@dataclass
class DAGVerificationResult:
    """검증 결과를 담는 class"""
    is_dag: bool    # DAG 여부
    has_cycle: bool # cycle 존재 여부

    sorted_indices: List[int] = field(default_factory=list)
    sorted_dependencies: List[str] = field(default_factory=list)    #is_dag=True일 때의 최종 위상 정렬 결과
    
    partial_order_indices: List[int] = field(default_factory=list)
    partial_sorted_dependencies: List[str] = field(default_factory=list)    # cycle에 막히기 전까지 먼저 처리 가능한 dependency 목록
    
    blocked_indices: List[int] = field(default_factory=list)
    blocked_dependencies: List[str] = field(default_factory=list)   # cycle 또는 cycle의 영향 때문에 처리되지 못한 dependency index 목록
    
    valid_dependency_pairs: List[List[int]] = field(default_factory=list)   # 검증을 통과한 정상 dependency pair 목록
    invalid_edges: List[Dict[str, Any]] = field(default_factory=list)   # 구조적으로 잘못되어 제거된 edge 목록
    duplicate_edges: List[List[int]] = field(default_factory=list)  # 중복되어 무시된 edge 목록
    
    cycle_indices: List[int] = field(default_factory=list)
    cycle_dependencies: List[str] = field(default_factory=list) # cycle을 구성하는 경로
    input_is_valid: bool = True # 입력 형식 검증 (pair 형식, index 정수, 범위 이탈, self-loop 등)
    has_invalid_edges: bool = False # invalid edge가 하나라도 있었는지 여부
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DependencyGraphCycleError(ValueError):
    """ 그래프에 cycle이 있어 valid DAG를 만들 수 없을 때 발생하는 예외
    - 어떤 의존성들이 cycle을 만들었는지 출력
    - 어떤 edge가 structurally invalid였는지 출력
    """
    def __init__(self, dag_result: Dict[str, Any]):
        self.dag_result = dag_result
        cycle_dependencies = dag_result.get("cycle_dependencies", [])
        invalid_edges = dag_result.get("invalid_edges", [])
        message = (
            "Dependency graph is not a valid DAG. "
            f"cycle_dependencies={cycle_dependencies}, "
            f"invalid_edges={invalid_edges}"
        )

        super().__init__(message)

def _jsonable_edge(edge: Any) -> Any:
    """
    ** edge 결과를 dictionary에 넣기 좋게 변환하는 보조 함수 **
    - tuple -> list
    - list
    - other type -> 문자열
    """

    if isinstance(edge, tuple):
        return list(edge)
    if isinstance(edge, list):
        return edge
    return repr(edge)


def _normalize_dependency_pairs(
    dependency_pairs: Optional[Any],
    num_dependencies: int,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]], List[List[int]], Dict[int, List[int]]]:
    """
    index 기반 adjacency list 그래프 만들기

    Returns
    -------
    - 정상 edge -> valid_pairs, graph
    - 잘못된 edge -> invalid_edges
    - 중복 edge -> duplicate_edges

    - output = 이후 Kahn/DFS 알고리즘의 input
    """
    graph: Dict[int, List[int]] = {idx: [] for idx in range(num_dependencies)}
    valid_pairs: List[Tuple[int, int]] = []
    invalid_edges: List[Dict[str, Any]] = []
    duplicate_edges: List[List[int]] = []
    seen_graph_edges = set()

    if dependency_pairs is None:
        return valid_pairs, invalid_edges, duplicate_edges, graph

    if not isinstance(dependency_pairs, (list, tuple)):
        invalid_edges.append({
            "edge": _jsonable_edge(dependency_pairs),
            "reason": "dependency_pairs_must_be_a_list_or_tuple",
        })
        return valid_pairs, invalid_edges, duplicate_edges, graph

    for raw_edge in dependency_pairs:
        edge_for_log = _jsonable_edge(raw_edge)

        if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 2:
            invalid_edges.append({"edge": edge_for_log, "reason": "edge_must_be_pair"})
            continue

        dependent_idx, dependency_idx = raw_edge

        if (
            not isinstance(dependent_idx, int)
            or isinstance(dependent_idx, bool)
            or not isinstance(dependency_idx, int)
            or isinstance(dependency_idx, bool)
        ):
            invalid_edges.append({"edge": edge_for_log, "reason": "indices_must_be_integers"})
            continue

        if not (0 <= dependent_idx < num_dependencies) or not (0 <= dependency_idx < num_dependencies):
            invalid_edges.append({"edge": edge_for_log, "reason": "index_out_of_range"})
            continue

        if dependent_idx == dependency_idx:
            invalid_edges.append({"edge": edge_for_log, "reason": "self_loop"})
            continue

        graph_edge = (dependency_idx, dependent_idx)
        if graph_edge in seen_graph_edges:
            duplicate_edges.append([dependent_idx, dependency_idx])
            continue

        seen_graph_edges.add(graph_edge)
        valid_pairs.append((dependent_idx, dependency_idx))
        graph[dependency_idx].append(dependent_idx)

    return valid_pairs, invalid_edges, duplicate_edges, graph


def _kahn_partial_order(graph: Dict[int, List[int]], num_dependencies: int) -> Tuple[List[int], List[int]]:
    """
    cycle에 의해 진행이 막히기 전까지 처리 가능한 node 순서 계산

    1) DAG인 경우(cycle X) : processed = 전체 topological order
    2) cycle이 있는 경우 : processsed = 먼저 처리 가능한 acyclic prefix, blocked = cycle 또는 cycle의 영향 때문에 처리되지 못한 node 목록
    """
    indegree = [0] * num_dependencies
    for node in range(num_dependencies):
        for neighbor in graph[node]:
            indegree[neighbor] += 1

    queue: Deque[int] = deque(idx for idx in range(num_dependencies) if indegree[idx] == 0)
    processed: List[int] = []

    while queue:
        node = queue.popleft()
        processed.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    processed_set = set(processed)
    blocked = [idx for idx in range(num_dependencies) if idx not in processed_set]
    return processed, blocked


def _dfs_cycle_and_topological_order(
    graph: Dict[int, List[int]],
    num_dependencies: int,
) -> Tuple[bool, List[int], List[int]]:
    """
    3-state DFS로 directed cycle을 탐지하고, DFS 기반 topological order를 계산
    - 0 = UNVISITED: 아직 방문하지 않음
    - 1 = VISITING: 현재 DFS 경로 위에 있음
    - 2 = DONE: 방문 완료

    Returns
    -------
    has_cycle
    cycle_indices
    sorted_indices
    """

    UNVISITED, VISITING, DONE = 0, 1, 2
    state = [UNVISITED] * num_dependencies
    parent = [-1] * num_dependencies
    postorder: List[int] = []
    cycle_indices: List[int] = []

    def reconstruct_cycle(current: int, neighbor: int) -> List[int]:
        path = [neighbor]
        node = current
        while node != neighbor and node != -1:
            path.append(node)
            node = parent[node]
        path.append(neighbor)
        path.reverse()
        return path

    def dfs(node: int) -> bool:
        nonlocal cycle_indices
        state[node] = VISITING

        for neighbor in graph[node]:
            if state[neighbor] == UNVISITED:
                parent[neighbor] = node
                if dfs(neighbor):
                    return True
            elif state[neighbor] == VISITING:
                cycle_indices = reconstruct_cycle(node, neighbor)
                return True

        state[node] = DONE
        postorder.append(node)
        return False

    for node in range(num_dependencies):
        if state[node] == UNVISITED and dfs(node):
            return True, cycle_indices, []

    return False, [], postorder[::-1]


def verify_dag_and_topological_sort(
    dependencies: List[str],
    dependency_pairs: Optional[Iterable[Any]],
) -> Dict[str, Any]:
    """
    LogicRAG dependency의 비순환성을 검증하고 topological order를 계산
    """

    num_dependencies = len(dependencies)
    valid_pairs, invalid_edges, duplicate_edges, graph = _normalize_dependency_pairs(
        dependency_pairs,
        num_dependencies,
    )

    partial_order_indices, blocked_indices = _kahn_partial_order(graph, num_dependencies)
    has_cycle, cycle_indices, sorted_indices = _dfs_cycle_and_topological_order(
        graph,
        num_dependencies,
    )

    if has_cycle:
        result = DAGVerificationResult(
            is_dag=False,
            has_cycle=True,
            partial_order_indices=partial_order_indices,
            partial_sorted_dependencies=[dependencies[idx] for idx in partial_order_indices],
            blocked_indices=blocked_indices,
            blocked_dependencies=[dependencies[idx] for idx in blocked_indices],
            valid_dependency_pairs=[list(pair) for pair in valid_pairs],
            invalid_edges=invalid_edges,
            duplicate_edges=duplicate_edges,
            cycle_indices=cycle_indices,
            cycle_dependencies=[dependencies[idx] for idx in cycle_indices],
            input_is_valid=len(invalid_edges) == 0,
            has_invalid_edges=bool(invalid_edges),
            message="Cycle detected in dependency graph.",
        )
        return result.to_dict()

    sorted_indices = partial_order_indices
    sorted_dependencies = [dependencies[idx] for idx in sorted_indices]
    if invalid_edges:
        message = "Valid DAG after removing structurally invalid edges."
    else:
        message = "Valid DAG."

    result = DAGVerificationResult(
        is_dag=True,
        has_cycle=False,
        sorted_indices=sorted_indices,
        sorted_dependencies=sorted_dependencies,
        partial_order_indices=sorted_indices,
        partial_sorted_dependencies=sorted_dependencies,
        blocked_indices=[],
        blocked_dependencies=[],
        valid_dependency_pairs=[list(pair) for pair in valid_pairs],
        invalid_edges=invalid_edges,
        duplicate_edges=duplicate_edges,
        cycle_indices=[],
        cycle_dependencies=[],
        input_is_valid=len(invalid_edges) == 0,
        has_invalid_edges=bool(invalid_edges),
        message=message,
    )
    return result.to_dict()


def build_partial_order_fallback(
    dependencies: List[str],
    dag_result: Dict[str, Any],
) -> List[str]:
    """
    cycle이 있을 때 fallback 순서를 만드는 함수

    1. partial_order_indices에 있는 dependency를 먼저 넣는다.
    2. 아직 사용되지 않은 dependency를 원래 순서대로 뒤에 붙인다.
    """
    partial_indices = dag_result.get("partial_order_indices", []) or []
    used = set(partial_indices)
    fallback_indices = list(partial_indices) + [
        idx for idx in range(len(dependencies)) if idx not in used
    ]
    return [dependencies[idx] for idx in fallback_indices]


def topological_sort_or_raise(
    dependencies: List[str],
    dependency_pairs: Optional[Any],
) -> List[str]:
    """그래프에 cycle이 있으면 예외를 발생시키는 하위 호환 헬퍼 함수."""
    dag_result = verify_dag_and_topological_sort(dependencies, dependency_pairs)
    if not dag_result["is_dag"]:
        raise DependencyGraphCycleError(dag_result)
    return dag_result["sorted_dependencies"]