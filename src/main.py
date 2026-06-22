#!/usr/bin/env python
import argparse
import json
import logging
from typing import Dict, List

from config.config import (
    DATASET,
    DATASET_PATH,
    CORPUS_PATH,
    OUTPUT_FILE,
    LIMIT,
    TOP_K,
    EVAL_TOP_KS,
    MAX_ROUNDS,
    CHECKPOINT_INTERVAL,
    ENABLE_WARM_UP,
    ENABLE_EARLY_STOP,
    FINAL_ANSWER_POLICY,
    _RUN_TIMESTAMP,
)


logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_evaluation_data(dataset_path: str, limit: int) -> List[Dict]:
    try:
        with open(dataset_path, 'r') as f:
            eval_data = json.load(f)
        if limit and limit > 0:
            eval_data = eval_data[:limit]
        return eval_data
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return []


def _str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected one of: true, false, 1, 0, yes, no"
    )


def main():
    parser = argparse.ArgumentParser(description="Run LogicRAG evaluation.")
    parser.add_argument("--enable-warm-up", type=_str_to_bool, default=None)
    parser.add_argument("--enable-early-stop", type=_str_to_bool, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument(
        "--final-answer-policy",
        choices=["generate", "structured"],
        default=None,
    )
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    effective_enable_warm_up = (
        ENABLE_WARM_UP if args.enable_warm_up is None else args.enable_warm_up
    )
    effective_enable_early_stop = (
        ENABLE_EARLY_STOP if args.enable_early_stop is None else args.enable_early_stop
    )
    effective_max_rounds = MAX_ROUNDS if args.max_rounds is None else args.max_rounds
    effective_policy = (
        FINAL_ANSWER_POLICY if args.final_answer_policy is None else args.final_answer_policy
    )
    effective_limit = LIMIT if args.limit is None else args.limit

    eval_data = load_evaluation_data(DATASET_PATH, effective_limit)
    if not eval_data:
        logger.error("No evaluation data available. Exiting.")
        return

    if args.run_name:
        output_file = f"evaluation_results_{DATASET}_{args.run_name}_{_RUN_TIMESTAMP}.json"
    else:
        output_file = OUTPUT_FILE

    from src.evaluation.evaluation import RAGEvaluator

    evaluator = RAGEvaluator(
        model_name="logic-rag",
        corpus_path=CORPUS_PATH,
        max_rounds=effective_max_rounds,
        top_k=TOP_K,
        eval_top_ks=EVAL_TOP_KS,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        enable_warm_up=effective_enable_warm_up,
        enable_early_stop=effective_enable_early_stop,
        final_answer_policy=effective_policy,
        limit=effective_limit,
        run_name=args.run_name,
    )

    evaluator.run_single_model_evaluation(
        eval_data=eval_data,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()
