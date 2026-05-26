#!/usr/bin/env python
import json
import logging
from typing import Dict, List

from src.evaluation.evaluation import RAGEvaluator
from src.models.base_rag import BaseRAG
from src.models.logic_rag import LogicRAG
from config.config import (
    DATASET_PATH,
    CORPUS_PATH,
    OUTPUT_FILE,
    LIMIT,
    TOP_K,
    EVAL_TOP_KS,
    MAX_ROUNDS,
    CHECKPOINT_INTERVAL,
)


logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAG_MODELS = {
    "logic-rag": LogicRAG,
}


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


def main():
    eval_data = load_evaluation_data(DATASET_PATH, LIMIT)
    if not eval_data:
        logger.error("No evaluation data available. Exiting.")
        return

    evaluator = RAGEvaluator(
        model_name="logic-rag",
        corpus_path=CORPUS_PATH,
        max_rounds=MAX_ROUNDS,
        top_k=TOP_K,
        eval_top_ks=EVAL_TOP_KS,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )

    evaluator.run_single_model_evaluation(
        eval_data=eval_data,
        output_file=OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()