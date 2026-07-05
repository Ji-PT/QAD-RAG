"""
Run target-wise dependency classification on existing decomposition results.

This script is analysis-only:
- It reads model_subproblems from an existing decomposition evaluation JSON.
- It does not call LogicRAG.decompose_query().
- It does not modify subproblem text, order, or ids.
- It stores target-wise dependency classifications for later analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.utils.utils import fix_json_response, get_response_with_retry


DEPENDENCY_TYPES = {"independent", "depends_on_one", "depends_on_multiple"}


def _clean_response(response: str) -> str:
    return response.strip().replace("```json", "").replace("```", "")


def _as_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_subproblem_payload(
    subproblems: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for fallback_pos, subproblem in enumerate(subproblems):
        if not isinstance(subproblem, dict):
            continue
        node_id = _as_int_or_none(subproblem.get("id"))
        if node_id is None:
            node_id = fallback_pos
        payload.append({
            "id": node_id,
            "text": str(subproblem.get("text", "")),
        })
    return payload


def _dependency_type_for_count(n_inputs: int) -> str:
    if n_inputs <= 0:
        return "independent"
    if n_inputs == 1:
        return "depends_on_one"
    return "depends_on_multiple"


def _normalize_classifications(
    raw_classifications: Any,
    subproblems: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    subproblem_payload = _normalize_subproblem_payload(subproblems)
    existing_ids = {item["id"] for item in subproblem_payload}
    id_to_pos = {
        item["id"]: pos
        for pos, item in enumerate(subproblem_payload)
    }

    if not isinstance(raw_classifications, list):
        raw_classifications = []

    by_target: Dict[int, Dict[str, Any]] = {}

    for raw_classification in raw_classifications:
        if not isinstance(raw_classification, dict):
            continue

        target_id = _as_int_or_none(raw_classification.get("target_id"))
        if target_id not in existing_ids:
            continue

        target_pos = id_to_pos[target_id]
        raw_input_ids = raw_classification.get("required_input_ids", [])
        if not isinstance(raw_input_ids, list):
            raw_input_ids = []

        required_input_ids: List[int] = []
        seen_inputs = set()
        for raw_input_id in raw_input_ids:
            input_id = _as_int_or_none(raw_input_id)
            if input_id is None:
                continue
            if input_id == target_id:
                continue
            if input_id not in existing_ids:
                continue
            if id_to_pos[input_id] >= target_pos:
                continue
            if input_id in seen_inputs:
                continue
            seen_inputs.add(input_id)
            required_input_ids.append(input_id)

        by_target[target_id] = {
            "target_id": target_id,
            "dependency_type": _dependency_type_for_count(len(required_input_ids)),
            "required_input_ids": required_input_ids,
            "reason": str(raw_classification.get("reason", "")).strip(),
        }

    normalized: List[Dict[str, Any]] = []
    for item in subproblem_payload:
        target_id = item["id"]
        normalized.append(by_target.get(target_id, {
            "target_id": target_id,
            "dependency_type": "independent",
            "required_input_ids": [],
            "reason": "No valid target-wise classification was returned.",
        }))

    return normalized


def _classifications_to_edges(
    classifications: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    seen_edges = set()
    for classification in classifications:
        target_id = classification["target_id"]
        reason = str(classification.get("reason", "")).strip()
        for input_id in classification.get("required_input_ids", []):
            edge_key = (input_id, target_id)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edges.append({
                "prerequisite_id": input_id,
                "dependent_id": target_id,
                "reason": reason,
            })
    return edges


def _gold_dependency_edges(
    gold_decomposition: Sequence[Dict[str, Any]],
) -> List[Dict[str, int]]:
    edges: List[Dict[str, int]] = []
    seen_edges = set()

    for dependent_idx, step in enumerate(gold_decomposition):
        if not isinstance(step, dict):
            continue
        question = str(step.get("question", ""))
        for match in re.finditer(r"#(\d+)", question):
            prerequisite_idx = int(match.group(1)) - 1
            if prerequisite_idx < 0:
                continue
            if prerequisite_idx >= dependent_idx:
                continue
            if prerequisite_idx >= len(gold_decomposition):
                continue

            edge_key = (prerequisite_idx, dependent_idx)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edges.append({
                "prerequisite_id": prerequisite_idx,
                "dependent_id": dependent_idx,
            })

    return edges


def _dependency_pattern(
    dependencies: Sequence[Dict[str, Any]],
    subproblems: Sequence[Dict[str, Any]],
) -> str:
    if not dependencies:
        return "empty"

    subproblem_payload = _normalize_subproblem_payload(subproblems)
    ordered_ids = [item["id"] for item in subproblem_payload]
    edge_set = {
        (edge.get("prerequisite_id"), edge.get("dependent_id"))
        for edge in dependencies
    }

    linear_edges = {
        (ordered_ids[i - 1], ordered_ids[i])
        for i in range(1, len(ordered_ids))
    }
    if edge_set == linear_edges:
        return "linear_chain"

    if ordered_ids:
        last_id = ordered_ids[-1]
        final_merge_edges = [
            edge
            for edge in dependencies
            if edge.get("dependent_id") == last_id
        ]
        if len(final_merge_edges) >= 2:
            return "final_merge"

    return "other"


def _build_prompt(
    question: str,
    subproblems: Sequence[Dict[str, Any]],
) -> str:
    subproblem_payload = _normalize_subproblem_payload(subproblems)
    return f"""You are given an original question and a fixed list of subproblems.

Original question:
{question}

Fixed subproblems:
{json.dumps(subproblem_payload, ensure_ascii=False, indent=2)}

Your task is NOT to rewrite, add, remove, or reorder subproblems.
Your task is only to classify the direct dependencies among the given subproblems.

For each target subproblem, decide whether answering it directly requires the concrete answer value of any earlier subproblem.

Rules:
1. A dependency exists only if the target cannot be answered without the actual answer value of an earlier subproblem.
2. Do not infer dependency from subproblem order alone.
3. Do not infer dependency from topic similarity alone.
4. Do not connect independent anchor lookups to each other.
5. If multiple independent anchors are used together in a final comparison, aggregation, location relation, or judgment, the final target may depend on multiple previous subproblems.
6. Use only direct dependencies. Do not include redundant transitive dependencies.
7. If a target can be answered directly from the original question without earlier answers, mark it as independent.
8. Do not change the subproblem text.

Return ONLY a JSON object:
{{
  "classifications": [
    {{
      "target_id": 0,
      "dependency_type": "independent",
      "required_input_ids": [],
      "reason": "..."
    }}
  ]
}}

dependency_type must be one of:
- "independent"
- "depends_on_one"
- "depends_on_multiple"
"""


def classify_targetwise_dependencies(
    question: str,
    subproblems: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], str | None]:
    prompt = _build_prompt(question=question, subproblems=subproblems)
    response = get_response_with_retry(prompt)
    parsed = fix_json_response(_clean_response(response))

    if not isinstance(parsed, dict):
        classifications = _normalize_classifications([], subproblems)
        return classifications, _classifications_to_edges(classifications), "non_dict_response"

    raw_classifications = parsed.get("classifications", [])
    if not isinstance(raw_classifications, list):
        parse_error = "classifications_not_list"
        raw_classifications = []
    else:
        parse_error = None

    classifications = _normalize_classifications(raw_classifications, subproblems)
    dependencies = _classifications_to_edges(classifications)
    return classifications, dependencies, parse_error


def _select_results(
    results: Sequence[Dict[str, Any]],
    ids: Sequence[str] | None,
    limit: int | None,
) -> List[Dict[str, Any]]:
    selected = list(results)
    if ids:
        wanted_ids = set(ids)
        selected = [result for result in selected if result.get("id") in wanted_ids]
    if limit is not None:
        selected = selected[:max(0, limit)]
    return selected


def run_analysis(
    input_path: str,
    output_path: str,
    ids: Sequence[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> None:
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Pass --overwrite to replace it."
        )

    with open(input_path, encoding="utf-8") as f:
        input_data = json.load(f)

    input_results = input_data.get("results", [])
    selected_results = _select_results(input_results, ids, limit)
    output_results: List[Dict[str, Any]] = []

    total = len(selected_results)
    for idx, result in enumerate(selected_results, start=1):
        print(f"Target-wise dependency analysis {idx}/{total}: {result.get('id')}")
        enriched = dict(result)
        subproblems = result.get("model_subproblems", [])
        gold_decomposition = result.get("gold_decomposition", [])

        try:
            classifications, dependencies, parse_error = classify_targetwise_dependencies(
                question=str(result.get("question", "")),
                subproblems=subproblems,
            )
            enriched["targetwise_dependency_classifications"] = classifications
            enriched["targetwise_dependencies"] = dependencies
            enriched["gold_dependency_edges"] = _gold_dependency_edges(gold_decomposition)
            enriched["dependency_pattern"] = _dependency_pattern(
                dependencies,
                subproblems,
            )
            if parse_error:
                enriched["targetwise_parse_error"] = parse_error
        except Exception as exc:
            classifications = _normalize_classifications([], subproblems)
            dependencies = _classifications_to_edges(classifications)
            enriched["targetwise_dependency_classifications"] = classifications
            enriched["targetwise_dependencies"] = dependencies
            enriched["gold_dependency_edges"] = _gold_dependency_edges(gold_decomposition)
            enriched["dependency_pattern"] = _dependency_pattern(
                dependencies,
                subproblems,
            )
            enriched["targetwise_error"] = str(exc)

        output_results.append(enriched)

    output_data = {
        "summary": input_data.get("summary", {}),
        "results": output_results,
    }

    if output.parent:
        os.makedirs(output.parent, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(output_results)} result(s) to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify target-wise dependencies for fixed decomposition results."
    )
    parser.add_argument("--input", required=True, help="Input decomposition evaluation JSON.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--ids", nargs="*", default=None, help="Optional sample ids to process.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of selected samples.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        input_path=args.input,
        output_path=args.output,
        ids=args.ids,
        limit=args.limit,
        overwrite=args.overwrite,
    )
