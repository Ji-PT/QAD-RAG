"""
decompose_query() 프롬프트 실험 일괄 실행 스크립트.

실험 3개를 순차 실행하고 baseline 대비 비교 요약을 출력한다.

실행:
  python -m src.experiments.query_decomposition.run_decomp_experiments
  python -m src.experiments.query_decomposition.run_decomp_experiments --dataset dataset/musique_sample_100.json

결과 파일:
  evaluation/decomp_exp_entity_cot_{n}_{run}.json
  evaluation/decomp_exp_self_verify_{n}_{run}.json
  evaluation/decomp_exp_hop_count_{n}_{run}.json

[설정]
EXPERIMENTS_DATASET_PATH : 평가할 데이터셋 경로
EXPERIMENTS_OUTPUT_DIR   : 결과 저장 폴더
BASELINE_FILE            : 비교 기준 baseline 파일 경로 (없으면 비교 생략)
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
from src.experiments.query_decomposition.decomp_variants import (
    LogicRAGExpEntityCoT,
    LogicRAGExpHopCount,
    LogicRAGExpSelfVerify,
)

# ── 설정 ──────────────────────────────────────────────────────────────────────
EXPERIMENTS_DATASET_PATH = DECOMP_EVAL_DATASET_PATH   # 기본값: decomposition_eval.py와 동일
EXPERIMENTS_OUTPUT_DIR   = DECOMP_EVAL_OUTPUT_DIR      # 기본값: evaluation/
BASELINE_FILE            = "evaluation/decomposition_eval_results_100_1.json"
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    ("entity_cot",   LogicRAGExpEntityCoT,  "실험 1: Entity-first CoT"),
    ("self_verify",  LogicRAGExpSelfVerify,  "실험 2: Self-verification loop"),
    ("hop_count",    LogicRAGExpHopCount,    "실험 3: Hop count 사전 추정"),
]


def _exp_output_path(dataset_path: str, output_dir: str, exp_name: str) -> str:
    """evaluation/decomp_exp_{name}_{샘플수}_{순번}.json 형식으로 파일명 생성."""
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
    """실험 결과를 baseline 포함 비교 테이블로 출력한다."""
    print("\n" + "=" * 70)
    print("=== 실험 결과 비교 ===")
    print(f"{'실험':30s} {'step수':>8s} {'내용':>8s} {'holistic':>10s}")
    print("-" * 70)

    if BASELINE_FILE and os.path.exists(BASELINE_FILE):
        s = _load_summary(BASELINE_FILE)
        print(
            f"{'[baseline]':30s} "
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
    parser = argparse.ArgumentParser(description="decompose_query 프롬프트 실험 일괄 실행")
    parser.add_argument("--dataset", default=EXPERIMENTS_DATASET_PATH)
    args = parser.parse_args()
    run_all_experiments(args.dataset)
