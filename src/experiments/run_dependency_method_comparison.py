"""Compare global DAG edge inference with target-wise dependency classification.

This is an analysis-only experiment. It reads fixed decomposition results,
does not call decompose_query(), and does not run retrieval/ranking/final QA.
"""

import argparse
import copy
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type


def load_query_logic_dag_builder() -> Type[Any]:
    module_path = Path(__file__).resolve().parents[1] / "models" / "query_logic_dag.py"
    spec = importlib.util.spec_from_file_location("query_logic_dag_for_analysis", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load query_logic_dag.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.QueryLogicDAGBuilder


Edge = Dict[str, Any]
EdgePair = Tuple[int, int]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def existing_subproblem_ids(subproblems: List[Dict[str, Any]]) -> Set[int]:
    ids: Set[int] = set()
    for i, subproblem in enumerate(subproblems):
        if not isinstance(subproblem, dict):
            continue
        try:
            ids.add(int(subproblem.get("id", i)))
        except Exception:
            continue
    return ids


def normalize_edges(edges: Any, valid_ids: Set[int]) -> List[Edge]:
    if not isinstance(edges, list):
        return []

    normalized: List[Edge] = []
    seen: Set[EdgePair] = set()

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        try:
            prerequisite_id = int(edge["prerequisite_id"])
            dependent_id = int(edge["dependent_id"])
        except Exception:
            continue

        if prerequisite_id not in valid_ids or dependent_id not in valid_ids:
            continue
        if prerequisite_id == dependent_id:
            continue

        pair = (prerequisite_id, dependent_id)
        if pair in seen:
            continue

        seen.add(pair)
        normalized.append(
            {
                "prerequisite_id": prerequisite_id,
                "dependent_id": dependent_id,
                "reason": str(edge.get("reason", "")).strip(),
            }
        )

    return normalized


def edge_pairs(edges: Iterable[Edge]) -> Set[EdgePair]:
    pairs: Set[EdgePair] = set()
    for edge in edges:
        pairs.add((int(edge["prerequisite_id"]), int(edge["dependent_id"])))
    return pairs


def pairs_to_edges(pairs: Iterable[EdgePair]) -> List[Edge]:
    return [
        {
            "prerequisite_id": prerequisite_id,
            "dependent_id": dependent_id,
        }
        for prerequisite_id, dependent_id in sorted(pairs)
    ]


def parse_gold_dependency_edges(gold_decomposition: Any) -> List[Edge]:
    if not isinstance(gold_decomposition, list):
        return []

    edges: List[Edge] = []
    seen: Set[EdgePair] = set()

    for dependent_idx, step in enumerate(gold_decomposition):
        if isinstance(step, dict):
            text = str(step.get("question", ""))
        else:
            text = str(step)

        for match in re.findall(r"#(\d+)", text):
            prerequisite_idx = int(match) - 1
            if prerequisite_idx < 0 or prerequisite_idx >= dependent_idx:
                continue
            pair = (prerequisite_idx, dependent_idx)
            if pair in seen:
                continue
            seen.add(pair)
            edges.append(
                {
                    "prerequisite_id": prerequisite_idx,
                    "dependent_id": dependent_idx,
                }
            )

    return edges


def compute_edge_metrics(predicted_edges: List[Edge], gold_edges: List[Edge]) -> Dict[str, Any]:
    predicted = edge_pairs(predicted_edges)
    gold = edge_pairs(gold_edges)

    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)

    precision = 1.0 if not predicted and not gold else (tp / len(predicted) if predicted else 0.0)
    recall = 1.0 if not predicted and not gold else (tp / len(gold) if gold else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "exact_match": predicted == gold,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_edge_count": len(predicted),
        "gold_edge_count": len(gold),
    }


def has_gold_path(gold_pairs: Set[EdgePair], start: int, end: int) -> bool:
    children: Dict[int, Set[int]] = defaultdict(set)
    for prerequisite_id, dependent_id in gold_pairs:
        children[prerequisite_id].add(dependent_id)

    stack = list(children.get(start, set()))
    visited: Set[int] = set()
    while stack:
        node_id = stack.pop()
        if node_id == end:
            return True
        if node_id in visited:
            continue
        visited.add(node_id)
        stack.extend(children.get(node_id, set()))
    return False


def compute_edge_error_breakdown(predicted_edges: List[Edge], gold_edges: List[Edge]) -> Dict[str, List[Edge]]:
    predicted = edge_pairs(predicted_edges)
    gold = edge_pairs(gold_edges)
    missing = gold - predicted
    extra = predicted - gold
    reversed_edges = {
        pair
        for pair in extra
        if (pair[1], pair[0]) in gold
    }
    transitive_extra = {
        pair
        for pair in extra
        if pair not in reversed_edges and has_gold_path(gold, pair[0], pair[1])
    }

    return {
        "missing_edges": pairs_to_edges(missing),
        "extra_edges": pairs_to_edges(extra),
        "reversed_edges": pairs_to_edges(reversed_edges),
        "transitive_extra_edges": pairs_to_edges(transitive_extra),
    }


def summarize_metric_rows(rows: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "exact_match_rate": 0.0,
            "micro": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
            "macro": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        }

    exact = sum(1 for row in rows if row[f"{prefix}_edge_metrics"]["exact_match"])
    tp = sum(row[f"{prefix}_edge_metrics"]["tp"] for row in rows)
    fp = sum(row[f"{prefix}_edge_metrics"]["fp"] for row in rows)
    fn = sum(row[f"{prefix}_edge_metrics"]["fn"] for row in rows)

    micro_precision = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
    micro_recall = tp / (tp + fn) if tp + fn else (1.0 if not fp else 0.0)
    micro_f1 = (
        0.0
        if micro_precision + micro_recall == 0
        else 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    )

    macro_precision = sum(row[f"{prefix}_edge_metrics"]["precision"] for row in rows) / len(rows)
    macro_recall = sum(row[f"{prefix}_edge_metrics"]["recall"] for row in rows) / len(rows)
    macro_f1 = sum(row[f"{prefix}_edge_metrics"]["f1"] for row in rows) / len(rows)

    return {
        "count": len(rows),
        "exact_match_rate": round(exact / len(rows) * 100, 2),
        "micro": {
            "precision": round(micro_precision, 6),
            "recall": round(micro_recall, 6),
            "f1": round(micro_f1, 6),
        },
        "macro": {
            "precision": round(macro_precision, 6),
            "recall": round(macro_recall, 6),
            "f1": round(macro_f1, 6),
        },
    }


def summarize_error_breakdowns(rows: List[Dict[str, Any]], prefix: str) -> Dict[str, int]:
    return {
        "missing_edges": sum(len(row[f"{prefix}_missing_edges"]) for row in rows),
        "extra_edges": sum(len(row[f"{prefix}_extra_edges"]) for row in rows),
        "reversed_edges": sum(len(row[f"{prefix}_reversed_edges"]) for row in rows),
        "transitive_extra_edges": sum(
            len(row[f"{prefix}_transitive_extra_edges"]) for row in rows
        ),
    }


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(rows),
        "global": summarize_metric_rows(rows, "global"),
        "targetwise": summarize_metric_rows(rows, "targetwise"),
    }


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    same_prediction_count = 0
    for result in results:
        if edge_pairs(result["global_dependencies"]) == edge_pairs(result["targetwise_dependencies"]):
            same_prediction_count += 1

    labels = defaultdict(int)
    for result in results:
        labels[result["comparison_label"]] += 1

    subsets = {
        "all": results,
        "step_count_match=true": [r for r in results if bool(r.get("step_count_match"))],
        "all_steps_match=true": [r for r in results if bool(r.get("all_steps_match"))],
        "holistic_match=true": [r for r in results if bool(r.get("holistic_match"))],
    }

    by_pattern: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_pattern[str(result.get("dependency_pattern", "other"))].append(result)

    return {
        "total": len(results),
        "global_parse_error_count": sum(1 for r in results if r.get("global_parse_error")),
        "global_error_count": sum(1 for r in results if r.get("global_error")),
        "targetwise_parse_error_count": sum(1 for r in results if r.get("targetwise_parse_error")),
        "targetwise_error_count": sum(1 for r in results if r.get("targetwise_error")),
        "targetwise_missing_result_count": sum(
            1 for r in results if r.get("targetwise_missing_result")
        ),
        "metric_average_note": (
            "micro aggregates tp/fp/fn across samples; macro averages sample-level "
            "precision/recall/f1. Use macro when comparing with previous target-wise-only analysis."
        ),
        "global_overall": summarize_metric_rows(results, "global"),
        "targetwise_overall": summarize_metric_rows(results, "targetwise"),
        "global_error_breakdown": summarize_error_breakdowns(results, "global"),
        "targetwise_error_breakdown": summarize_error_breakdowns(results, "targetwise"),
        "same_prediction_count": same_prediction_count,
        "different_prediction_count": len(results) - same_prediction_count,
        "both_correct_count": labels["both_correct"],
        "targetwise_only_correct_count": labels["targetwise_only_correct"],
        "global_only_correct_count": labels["global_only_correct"],
        "both_wrong_count": labels["both_wrong"],
        "subsets": {name: summarize_rows(rows) for name, rows in subsets.items()},
        "by_dependency_pattern": {
            name: summarize_rows(rows)
            for name, rows in sorted(by_pattern.items())
        },
    }


def infer_global_dependencies(
    builder: Any,
    question: str,
    subproblems: List[Dict[str, Any]],
) -> Tuple[List[Edge], Optional[str], Optional[str]]:
    try:
        dag = builder.construct_from_subproblems(
            question=question,
            subproblems=copy.deepcopy(subproblems),
        )
        global_dependencies = [
            {
                "prerequisite_id": edge.prerequisite_id,
                "dependent_id": edge.dependent_id,
                "reason": edge.reason,
            }
            for edge in dag.E
        ]
        status = dag.metadata.get("edge_inference_status")
        parse_error = status if status and status != "ok" else None
        return global_dependencies, parse_error, None
    except Exception as exc:
        return [], None, f"{exc.__class__.__name__}: {exc}"


def make_comparison_label(global_metrics: Dict[str, Any], targetwise_metrics: Dict[str, Any]) -> str:
    global_correct = bool(global_metrics["exact_match"])
    targetwise_correct = bool(targetwise_metrics["exact_match"])
    if global_correct and targetwise_correct:
        return "both_correct"
    if targetwise_correct:
        return "targetwise_only_correct"
    if global_correct:
        return "global_only_correct"
    return "both_wrong"


def compare_methods(
    rule5_data: Dict[str, Any],
    targetwise_data: Dict[str, Any],
    ids: Optional[List[str]],
    limit: Optional[int],
) -> Dict[str, Any]:
    targetwise_by_id = {
        result.get("id"): result
        for result in targetwise_data.get("results", [])
        if isinstance(result, dict) and result.get("id") is not None
    }

    rule5_results = [
        result
        for result in rule5_data.get("results", [])
        if isinstance(result, dict)
    ]

    if ids:
        allowed_ids = set(ids)
        rule5_results = [result for result in rule5_results if result.get("id") in allowed_ids]

    if limit is not None:
        rule5_results = rule5_results[:limit]

    query_logic_dag_builder = load_query_logic_dag_builder()
    builder = query_logic_dag_builder()
    comparison_results: List[Dict[str, Any]] = []

    for index, rule5_result in enumerate(rule5_results, start=1):
        sample_id = rule5_result.get("id")
        print(f"Dependency method comparison {index}/{len(rule5_results)}: {sample_id}", flush=True)

        question = str(rule5_result.get("question", ""))
        subproblems = copy.deepcopy(rule5_result.get("model_subproblems", []))
        valid_ids = existing_subproblem_ids(subproblems)
        targetwise_result = targetwise_by_id.get(sample_id, {})
        targetwise_missing_result = sample_id not in targetwise_by_id

        global_dependencies, global_parse_error, global_error = infer_global_dependencies(
            builder=builder,
            question=question,
            subproblems=subproblems,
        )
        global_dependencies = normalize_edges(global_dependencies, valid_ids)

        targetwise_dependencies = normalize_edges(
            targetwise_result.get("targetwise_dependencies", []),
            valid_ids,
        )

        gold_dependency_edges = normalize_edges(
            targetwise_result.get("gold_dependency_edges", []),
            valid_ids,
        )
        if not gold_dependency_edges:
            gold_dependency_edges = normalize_edges(
                parse_gold_dependency_edges(rule5_result.get("gold_decomposition", [])),
                valid_ids,
            )

        global_metrics = compute_edge_metrics(global_dependencies, gold_dependency_edges)
        targetwise_metrics = compute_edge_metrics(targetwise_dependencies, gold_dependency_edges)
        global_breakdown = compute_edge_error_breakdown(
            global_dependencies,
            gold_dependency_edges,
        )
        targetwise_breakdown = compute_edge_error_breakdown(
            targetwise_dependencies,
            gold_dependency_edges,
        )

        comparison_results.append(
            {
                "id": sample_id,
                "hop_type": rule5_result.get("hop_type"),
                "question": question,
                "model_subproblems": subproblems,
                "gold_decomposition": rule5_result.get("gold_decomposition", []),
                "step_count_match": rule5_result.get("step_count_match"),
                "all_steps_match": rule5_result.get("all_steps_match"),
                "holistic_match": rule5_result.get("holistic_match"),
                "global_dependencies": global_dependencies,
                "targetwise_dependencies": targetwise_dependencies,
                "targetwise_dependency_classifications": targetwise_result.get(
                    "targetwise_dependency_classifications",
                    [],
                ),
                "gold_dependency_edges": gold_dependency_edges,
                "dependency_pattern": targetwise_result.get("dependency_pattern", "other"),
                "global_edge_metrics": global_metrics,
                "targetwise_edge_metrics": targetwise_metrics,
                "global_missing_edges": global_breakdown["missing_edges"],
                "global_extra_edges": global_breakdown["extra_edges"],
                "global_reversed_edges": global_breakdown["reversed_edges"],
                "global_transitive_extra_edges": global_breakdown["transitive_extra_edges"],
                "targetwise_missing_edges": targetwise_breakdown["missing_edges"],
                "targetwise_extra_edges": targetwise_breakdown["extra_edges"],
                "targetwise_reversed_edges": targetwise_breakdown["reversed_edges"],
                "targetwise_transitive_extra_edges": targetwise_breakdown[
                    "transitive_extra_edges"
                ],
                "comparison_label": make_comparison_label(global_metrics, targetwise_metrics),
                "global_parse_error": global_parse_error,
                "global_error": global_error,
                "targetwise_missing_result": targetwise_missing_result,
                "targetwise_parse_error": targetwise_result.get("targetwise_parse_error"),
                "targetwise_error": targetwise_result.get("targetwise_error"),
            }
        )

    return {
        "summary": build_summary(comparison_results),
        "results": comparison_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare global edge inference with target-wise dependency classification.",
    )
    parser.add_argument("--rule5-input", required=True, type=Path)
    parser.add_argument("--targetwise-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists. Use --overwrite: {args.output}")

    rule5_data = load_json(args.rule5_input)
    targetwise_data = load_json(args.targetwise_input)
    output = compare_methods(
        rule5_data=rule5_data,
        targetwise_data=targetwise_data,
        ids=args.ids,
        limit=args.limit,
    )
    write_json(args.output, output)
    print(f"Wrote {len(output['results'])} result(s) to {args.output}")


if __name__ == "__main__":
    main()
