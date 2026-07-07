"""Run fixed-DAG global-vs-targetwise execution comparison.

This runner injects fixed Rule5-note subproblems and fixed edge sets from an
analysis JSON. It does not call decompose_query()'s LLM implementation and does
not rerun DAG edge inference.
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.experiments.fixed_dag_variants import LogicRAGFixedDAG
from src.models.verify_non_cyclicity import DependencyGraphCycleError
from src.utils.utils import evaluate_with_llm


DISAGREEMENT_10 = {
    "2hop__269766_43945",
    "3hop2__38326_92991_76291",
    "3hop1__498954_160713_77246",
    "4hop2__71753_729371_70784_79935",
    "4hop1__277409_49925_13759_736921",
    "4hop2__71753_648517_70784_79935",
    "4hop3__316459_41402_145282_13584",
    "4hop2__71753_73205_70784_79935",
    "4hop3__316459_41402_146281_13584",
    "2hop__684287_78303",
}

STAGE_DAG_CONSTRUCTION = "query_logic_dag_construction"
STAGE_RESOLUTION = "parent_answer_conditioned_rank_resolution_with_rolling_memory"
FIXED_EDGE_STATUS = "fixed_from_comparison_json"
VALID_EXECUTION_LABELS = {
    "both_correct",
    "both_wrong",
    "global_only_correct",
    "targetwise_only_correct",
    "indeterminate",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def edge_pair_set(edges: Any) -> Set[Tuple[int, int]]:
    pairs: Set[Tuple[int, int]] = set()
    if not isinstance(edges, list):
        return pairs
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        try:
            pairs.add((int(edge["prerequisite_id"]), int(edge["dependent_id"])))
        except Exception:
            continue
    return pairs


def subproblem_pairs(subproblems: Any) -> List[Tuple[int, str]]:
    pairs: List[Tuple[int, str]] = []
    if not isinstance(subproblems, list):
        return pairs
    for i, subproblem in enumerate(subproblems):
        if not isinstance(subproblem, dict):
            continue
        try:
            node_id = int(subproblem.get("id", i))
        except Exception:
            continue
        pairs.append((node_id, str(subproblem.get("text", "")).strip()))
    return sorted(pairs)


def dag_node_pairs(dag_dict: Any) -> List[Tuple[int, str]]:
    if not isinstance(dag_dict, dict):
        return []
    raw_nodes = dag_dict.get("nodes", {})
    if not isinstance(raw_nodes, dict):
        return []

    pairs: List[Tuple[int, str]] = []
    for raw_key, raw_node in raw_nodes.items():
        if not isinstance(raw_node, dict):
            continue
        try:
            node_id = int(raw_node.get("id", raw_key))
        except Exception:
            continue
        pairs.append((node_id, str(raw_node.get("text", "")).strip()))
    return sorted(pairs)


def get_stage(dependency_analysis: Any, stage_name: str) -> Dict[str, Any]:
    if not isinstance(dependency_analysis, list):
        return {}
    for entry in dependency_analysis:
        if isinstance(entry, dict) and entry.get("stage") == stage_name:
            return entry
    return {}


def stage5_entry(result: Dict[str, Any]) -> Dict[str, Any]:
    return get_stage(result.get("dependency_analysis", []), STAGE_RESOLUTION)


def rank_signature(result: Dict[str, Any]) -> Any:
    return stage5_entry(result).get("rank_groups", [])


def retrieval_queries(result: Dict[str, Any]) -> List[str]:
    queries: List[str] = []
    for record in result.get("retrieval_history", []) or []:
        if isinstance(record, dict) and record.get("unified_query") is not None:
            queries.append(str(record.get("unified_query")))
    return queries


def node_answer_signature(result: Dict[str, Any]) -> List[Tuple[Any, str, Any, str]]:
    signature: List[Tuple[Any, str, Any, str]] = []
    for record in result.get("retrieval_history", []) or []:
        if not isinstance(record, dict):
            continue
        rank_result = record.get("rank_result", {})
        if not isinstance(rank_result, dict):
            continue
        for answer in rank_result.get("node_answers", []) or []:
            if not isinstance(answer, dict):
                continue
            signature.append(
                (
                    answer.get("node_id"),
                    str(answer.get("answer", "")).strip(),
                    answer.get("is_answered"),
                    str(answer.get("missing_info", "") or "").strip(),
                )
            )
    return signature


def unresolved_count(result: Dict[str, Any]) -> int:
    count = 0
    for _node_id, _answer, is_answered, missing_info in node_answer_signature(result):
        if not bool(is_answered) or bool(str(missing_info).strip()):
            count += 1
    return count


def dynamic_texts(result: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    for item in stage5_entry(result).get("dynamic_adaptations", []) or []:
        if not isinstance(item, dict):
            continue
        added = item.get("added_subproblem", {})
        if not isinstance(added, dict):
            continue
        text = str(added.get("new_subproblem_text", "")).strip()
        if text:
            texts.append(text)
    return texts


def final_answer_text(result: Dict[str, Any]) -> str:
    answer = result.get("final_answer", "")
    return answer if isinstance(answer, str) else ""


def load_dataset_by_id(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    records = load_json(dataset_path)
    if not isinstance(records, list):
        raise ValueError(f"Dataset must be a list: {dataset_path}")
    by_id: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if isinstance(record, dict) and record.get("id") is not None:
            by_id[str(record["id"])] = record
    return by_id


def comparison_results_by_id(comparison_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    results = comparison_data.get("results", [])
    if not isinstance(results, list):
        raise ValueError("comparison-input must contain a list field: results")
    by_id: Dict[str, Dict[str, Any]] = {}
    for result in results:
        if isinstance(result, dict) and result.get("id") is not None:
            by_id[str(result["id"])] = result
    return by_id


def recompute_disagreement_ids(results_by_id: Dict[str, Dict[str, Any]]) -> Set[str]:
    ids: Set[str] = set()
    for sample_id, result in results_by_id.items():
        if edge_pair_set(result.get("global_dependencies", [])) != edge_pair_set(
            result.get("targetwise_dependencies", [])
        ):
            ids.add(sample_id)
    return ids


def validate_inputs(
    args: argparse.Namespace,
    comparison_data: Dict[str, Any],
    dataset_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    for path in [args.comparison_input, args.dataset, args.corpus]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
    if not args.ids:
        raise ValueError("--ids is required and must not be empty")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite: {args.output}")

    results_by_id = comparison_results_by_id(comparison_data)
    recomputed = recompute_disagreement_ids(results_by_id)
    if recomputed != DISAGREEMENT_10:
        message = (
            "Recomputed disagreement set differs from frozen DISAGREEMENT_10. "
            f"frozen={len(DISAGREEMENT_10)}, recomputed={len(recomputed)}, "
            f"frozen_only={sorted(DISAGREEMENT_10 - recomputed)}, "
            f"recomputed_only={sorted(recomputed - DISAGREEMENT_10)}"
        )
        if not args.allow_drift:
            raise ValueError(message + " Use --allow-drift to continue with recomputed set.")
        print("WARNING:", message, flush=True)

    allowed_ids = recomputed if args.allow_drift else DISAGREEMENT_10
    requested_ids = set(args.ids)
    if not requested_ids.issubset(allowed_ids):
        raise ValueError(
            "--ids must be a subset of the active disagreement set. "
            f"Invalid: {sorted(requested_ids - allowed_ids)}"
        )

    missing_in_comparison = [sample_id for sample_id in args.ids if sample_id not in results_by_id]
    if missing_in_comparison:
        raise ValueError(f"IDs missing from comparison input: {missing_in_comparison}")

    missing_dataset = [sample_id for sample_id in args.ids if sample_id not in dataset_by_id]
    if missing_dataset:
        raise ValueError(f"IDs missing from dataset: {missing_dataset}")

    empty_answers = [
        sample_id
        for sample_id in args.ids
        if not str(dataset_by_id[sample_id].get("answer", "")).strip()
    ]
    if empty_answers:
        raise ValueError(f"Dataset records with missing/empty answer: {empty_answers}")

    return results_by_id


def source_analysis(sample: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "comparison_label",
        "dependency_pattern",
        "global_dependencies",
        "targetwise_dependencies",
        "global_edge_metrics",
        "targetwise_edge_metrics",
        "global_missing_edges",
        "global_extra_edges",
        "global_reversed_edges",
        "global_transitive_extra_edges",
        "targetwise_missing_edges",
        "targetwise_extra_edges",
        "targetwise_reversed_edges",
        "targetwise_transitive_extra_edges",
    ]
    return {key: deepcopy(sample.get(key)) for key in keys}


def extract_edge_inference_status(model: LogicRAGFixedDAG) -> Optional[str]:
    dag_dict = getattr(model, "last_query_logic_dag_dict", None)
    if not isinstance(dag_dict, dict):
        return None
    metadata = dag_dict.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    return metadata.get("edge_inference_status")


def construction_edge_pairs(dependency_analysis: Any) -> Set[Tuple[int, int]]:
    stage = get_stage(dependency_analysis, STAGE_DAG_CONSTRUCTION)
    dag = stage.get("query_logic_dag", {}) if isinstance(stage, dict) else {}
    if not isinstance(dag, dict):
        return set()
    return edge_pair_set(dag.get("edges", []))


def empty_method_result(method: str) -> Dict[str, Any]:
    return {
        "edge_method": method,
        "subproblems_match_input": False,
        "fixed_edge_injection_mismatch": False,
        "edge_inference_status": None,
        "final_answer": "",
        "final_correct": None,
        "judge_status": "",
        "judge_raw_response": None,
        "answer_status": "",
        "answer_failure_type": None,
        "error": None,
        "dependency_analysis": [],
        "retrieval_history": [],
    }


def validate_injection(
    model: LogicRAGFixedDAG,
    method_result: Dict[str, Any],
    sample: Dict[str, Any],
    method: str,
) -> None:
    expected_pairs = edge_pair_set(sample.get(f"{method}_dependencies", []))
    actual_pairs = construction_edge_pairs(method_result.get("dependency_analysis", []))
    dag_dict = getattr(model, "last_query_logic_dag_dict", None)

    method_result["edge_inference_status"] = extract_edge_inference_status(model)
    method_result["subproblems_match_input"] = (
        dag_node_pairs(dag_dict) == subproblem_pairs(sample.get("model_subproblems", []))
    )
    method_result["fixed_edge_injection_mismatch"] = actual_pairs != expected_pairs

    errors: List[str] = []
    if method_result["edge_inference_status"] != FIXED_EDGE_STATUS:
        errors.append(
            f"edge_inference_status mismatch: expected={FIXED_EDGE_STATUS!r}, "
            f"actual={method_result['edge_inference_status']!r}"
        )
    if not method_result["subproblems_match_input"]:
        errors.append("subproblems do not match comparison JSON input")
    if method_result["fixed_edge_injection_mismatch"]:
        errors.append(
            "fixed edge injection mismatch: "
            f"expected={sorted(expected_pairs)}, actual={sorted(actual_pairs)}"
        )

    if errors:
        method_result["error"] = "; ".join(errors)


def run_judge(answer: str, gold_answer: str, answer_status: Any, failure_type: Any) -> Tuple[str, Optional[str], Optional[bool]]:
    if answer_status == "ok" and failure_type is None:
        return evaluate_with_llm(str(answer), str(gold_answer))
    if answer_status == "failed" and failure_type == "empty_content":
        return "not_run", None, False
    if answer_status == "failed" and failure_type == "api_error":
        return "not_run", None, None
    raise ValueError(f"Inconsistent final-answer status: {answer_status!r}, {failure_type!r}")


def run_one_method(
    sample: Dict[str, Any],
    method: str,
    corpus_path: Path,
    gold_answer: str,
    config: Dict[str, Any],
    fail_fast: bool,
) -> Dict[str, Any]:
    result = empty_method_result(method)
    try:
        model = LogicRAGFixedDAG(
            corpus_path=str(corpus_path),
            fixed_subproblems=deepcopy(sample.get("model_subproblems", [])),
            fixed_edges=deepcopy(sample.get(f"{method}_dependencies", [])),
        )
        model.set_max_rounds(config["max_rounds"])
        model.set_enable_warm_up(False)
        model.set_enable_early_stop(False)
        model.set_final_answer_policy(config["final_answer_policy"])
        model.max_dynamic_adaptations = config["max_dynamic_adaptations"]

        answer, _contexts, _rounds = model.answer_question(sample["question"])
        result["final_answer"] = answer
        result["answer_status"] = getattr(model, "last_answer_status", None)
        result["answer_failure_type"] = getattr(model, "last_answer_failure_type", None)
        result["dependency_analysis"] = getattr(model, "last_dependency_analysis", [])
        result["retrieval_history"] = getattr(model, "last_retrieval_history", [])

        validate_injection(model=model, method_result=result, sample=sample, method=method)
        if result["fixed_edge_injection_mismatch"]:
            if fail_fast:
                raise RuntimeError(result["error"] or "fixed_edge_injection_mismatch=True")
            return result

        judge_status, judge_raw_response, final_correct = run_judge(
            answer=str(answer),
            gold_answer=str(gold_answer),
            answer_status=result["answer_status"],
            failure_type=result["answer_failure_type"],
        )
        result["judge_status"] = judge_status
        result["judge_raw_response"] = judge_raw_response
        result["final_correct"] = final_correct
        return result

    except DependencyGraphCycleError as exc:
        result["error"] = f"DependencyGraphCycleError: {exc}"
        if fail_fast:
            raise
        return result
    except Exception as exc:
        result["error"] = f"{exc.__class__.__name__}: {exc}"
        if fail_fast:
            raise
        return result


def execution_comparison_label(global_result: Dict[str, Any], targetwise_result: Dict[str, Any]) -> str:
    global_correct = global_result.get("final_correct")
    targetwise_correct = targetwise_result.get("final_correct")
    if global_correct is None or targetwise_correct is None:
        return "indeterminate"
    if global_correct and targetwise_correct:
        return "both_correct"
    if global_correct and not targetwise_correct:
        return "global_only_correct"
    if targetwise_correct and not global_correct:
        return "targetwise_only_correct"
    return "both_wrong"


def build_pairwise(global_result: Dict[str, Any], targetwise_result: Dict[str, Any]) -> Dict[str, Any]:
    label = execution_comparison_label(global_result, targetwise_result)
    if label not in VALID_EXECUTION_LABELS:
        raise ValueError(f"Invalid execution_comparison_label: {label}")

    contaminated = (
        bool(global_result.get("fixed_edge_injection_mismatch"))
        or bool(targetwise_result.get("fixed_edge_injection_mismatch"))
        or global_result.get("edge_inference_status") != FIXED_EDGE_STATUS
        or targetwise_result.get("edge_inference_status") != FIXED_EDGE_STATUS
        or bool(global_result.get("error"))
        or bool(targetwise_result.get("error"))
    )

    return {
        "rank_changed": rank_signature(global_result) != rank_signature(targetwise_result),
        "retrieval_query_changed": retrieval_queries(global_result) != retrieval_queries(targetwise_result),
        "node_answer_changed": node_answer_signature(global_result) != node_answer_signature(targetwise_result),
        "unresolved_changed": unresolved_count(global_result) != unresolved_count(targetwise_result),
        "adaptation_trigger_changed": bool(dynamic_texts(global_result)) != bool(dynamic_texts(targetwise_result)),
        "added_subproblem_changed": dynamic_texts(global_result) != dynamic_texts(targetwise_result),
        "final_answer_changed": final_answer_text(global_result) != final_answer_text(targetwise_result),
        "execution_comparison_label": label,
        "global_correct": global_result.get("final_correct"),
        "targetwise_correct": targetwise_result.get("final_correct"),
        "contaminated": contaminated,
    }


def build_summary(results: List[Dict[str, Any]], ids: List[str], config: Dict[str, Any], elapsed_sec: float) -> Dict[str, Any]:
    labels = {label: 0 for label in sorted(VALID_EXECUTION_LABELS)}
    for result in results:
        label = result["pairwise"].get("execution_comparison_label", "indeterminate")
        labels[label] = labels.get(label, 0) + 1

    return {
        "sample_count": len(results),
        "ids": ids,
        "config": {
            "max_rounds": config["max_rounds"],
            "enable_warm_up": False,
            "enable_early_stop": False,
            "final_answer_policy": config["final_answer_policy"],
            "max_dynamic_adaptations": config["max_dynamic_adaptations"],
        },
        "injection_mismatch_count": sum(
            int(result["global"].get("fixed_edge_injection_mismatch", False))
            + int(result["targetwise"].get("fixed_edge_injection_mismatch", False))
            for result in results
        ),
        "method_error_count": {
            "global": sum(1 for result in results if result["global"].get("error")),
            "targetwise": sum(1 for result in results if result["targetwise"].get("error")),
        },
        "final_answer_changed_count": sum(
            1 for result in results if result["pairwise"].get("final_answer_changed")
        ),
        "execution_comparison_label_counts": labels,
        "elapsed_sec": round(elapsed_sec, 3),
    }


def run_comparison(args: argparse.Namespace) -> Dict[str, Any]:
    start_time = time.perf_counter()
    comparison_data = load_json(args.comparison_input)
    dataset_by_id = load_dataset_by_id(args.dataset)
    results_by_id = validate_inputs(args, comparison_data, dataset_by_id)
    config = {
        "max_rounds": int(args.max_rounds),
        "max_dynamic_adaptations": int(args.max_dynamic_adaptations),
        "final_answer_policy": str(args.final_answer_policy),
    }

    results: List[Dict[str, Any]] = []
    for sample_id in args.ids:
        sample = results_by_id[sample_id]
        gold_answer = str(dataset_by_id[sample_id].get("answer", "")).strip()
        print(f"Fixed-DAG edge comparison: {sample_id}", flush=True)

        global_result = run_one_method(
            sample=sample,
            method="global",
            corpus_path=args.corpus,
            gold_answer=gold_answer,
            config=config,
            fail_fast=args.fail_fast,
        )
        targetwise_result = run_one_method(
            sample=sample,
            method="targetwise",
            corpus_path=args.corpus,
            gold_answer=gold_answer,
            config=config,
            fail_fast=args.fail_fast,
        )

        results.append(
            {
                "id": sample_id,
                "question": sample.get("question", ""),
                "gold_answer": gold_answer,
                "source_analysis": source_analysis(sample),
                "global": global_result,
                "targetwise": targetwise_result,
                "pairwise": build_pairwise(global_result, targetwise_result),
            }
        )

    return {
        "summary": build_summary(results, args.ids, config, time.perf_counter() - start_time),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fixed global vs target-wise DAG edges in LogicRAG execution.",
    )
    parser.add_argument("--comparison-input", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ids", required=True, nargs="+")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-dynamic-adaptations", type=int, default=3)
    parser.add_argument("--final-answer-policy", default="structured")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-drift", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_comparison(args)
    write_json(args.output, output)
    print(f"Wrote {len(output['results'])} result(s) to {args.output}")


if __name__ == "__main__":
    main()
