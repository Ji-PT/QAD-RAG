"""
decompose_query() 프롬프트 실험 v2 일괄 실행 스크립트.

실행:
  python -m src.experiments.run_decomp_experiments_v2
  python -m src.experiments.run_decomp_experiments_v2 --dataset dataset/musique_sample_100.json

결과 파일:
  evaluation/decomp_exp_query_type_clf_{n}_{run}.json
"""

import argparse
import json
import logging
import os

from src.evaluation.decomposition_eval import (
    DECOMP_EVAL_DATASET_PATH,
    DECOMP_EVAL_OUTPUT_DIR,
    run_decomposition_eval,
)
from src.experiments.decomp_variants_v2 import LogicRAGExpQueryTypeClassifier

EXPERIMENTS_DATASET_PATH = DECOMP_EVAL_DATASET_PATH
EXPERIMENTS_OUTPUT_DIR   = DECOMP_EVAL_OUTPUT_DIR
BASELINE_FILE            = "evaluation/decomposition_eval_results_100_1.json"

EXPERIMENTS = [
    ("query_type_clf", LogicRAGExpQueryTypeClassifier, "실험 4: Query Type Classifier"),
]


def _exp_output_path(dataset_path: str, output_dir: str, exp_name: str) -> str:
    with open(dataset_path, encoding="utf-8") as f:
        n = len(json.load(f))

    os.makedirs(output_dir, exist_ok=True)
    prefix = f"decomp_exp_{exp_name}_{n}_"
    existing = [
        fname for fname in os.listdir(output_dir)
        if fname.startswith(prefix) and fname.endswith(".json")
    ]
    next_idx = len(existing) + 1
    return os.path.join(output_dir, f"{prefix}{next_idx}.json")


def _load_summary(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["summary"]


def _print_comparison(results: list[tuple[str, str]]) -> None:
    print("\n" + "=" * 70)
    print("=== 실험 결과 비교 ===")
    print(f"{'실험':30s} {'step수':>8s} {'내용':>8s} {'holistic':>10s}")
    print("-" * 70)

    prev_files = {
        "[baseline]":           "evaluation/decomposition_eval_results_100_1.json",
        "Exp1 EntityCoT v1":    "evaluation/decomp_exp_entity_cot_100_1.json",
        "Exp2 SelfVerify v2":   "evaluation/decomp_exp_self_verify_100_2.json",
        "Exp3 HopCount":        "evaluation/decomp_exp_hop_count_100_1.json",
    }
    for label, path in prev_files.items():
        if os.path.exists(path):
            s = _load_summary(path)
            print(
                f"{label:30s} "
                f"{s['step_count_match_rate']:>7.1f}% "
                f"{s['all_steps_match_rate']:>7.1f}% "
                f"{s['holistic_match_rate']:>9.1f}%"
            )

    print("-" * 70)
    for label, path in results:
        if os.path.exists(path):
            s = _load_summary(path)
            print(
                f"{label:30s} "
                f"{s['step_count_match_rate']:>7.1f}% "
                f"{s['all_steps_match_rate']:>7.1f}% "
                f"{s['holistic_match_rate']:>9.1f}%"
            )
        else:
            print(f"{label:30s}  (결과 파일 없음)")

    print("=" * 70)


def run_all_experiments(dataset_path: str) -> None:
    completed: list[tuple[str, str]] = []

    for exp_name, ModelClass, label in EXPERIMENTS:
        print(f"\n{'─' * 60}")
        print(f"▶ {label}")
        print(f"{'─' * 60}")

        output_path = _exp_output_path(dataset_path, EXPERIMENTS_OUTPUT_DIR, exp_name)
        model = ModelClass()
        run_decomposition_eval(
            dataset_path=dataset_path,
            output_path=output_path,
            model=model,
        )
        completed.append((label, output_path))

    _print_comparison(completed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="decompose_query 프롬프트 실험 v2 일괄 실행")
    parser.add_argument("--dataset", default=EXPERIMENTS_DATASET_PATH)
    args = parser.parse_args()
    run_all_experiments(args.dataset)
