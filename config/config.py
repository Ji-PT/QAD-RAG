"""
Configuration file for API keys and other settings.
"""
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# SMWU FACTCHAT API Configuration
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
# Weights & Biases Configuration
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# API Rate Limiting Configuration
CALLS_PER_MINUTE = 20
PERIOD = 60

# Model Configuration
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 500

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# Cache Configuration
CACHE_DIR = "cache"
RESULT_DIR = "evaluation"

# ============================================================
# Experiment Configuration
# ============================================================

DATASET = "musique"

DATASET_PATH  = os.getenv("EVAL_DATASET_PATH", f"dataset/{DATASET}.json")
CORPUS_PATH   = f"dataset/{DATASET}_corpus.json"
_RUN_TIMESTAMP = os.getenv("RUN_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))
OUTPUT_FILE   = f"evaluation_results_{DATASET}_{_RUN_TIMESTAMP}.json"

# 평가할 질문 수 (0 = 전체)
LIMIT = 1000  # 논문: validation set 1,000개

# Retrieval 설정
TOP_K = 3          # 논문: k=3
EVAL_TOP_KS = [1, 2, 3, 5, 10, 20]  # 논문: top-k ablation

# Dynamic DAG Adaptation 최대 횟수
MAX_ROUNDS = 5     # 논문: max-rounds=5

# 체크포인트 저장 간격 (질문 수)
CHECKPOINT_INTERVAL = 5

# ── 실험 제어 플래그 (CLI: --enable-warm-up / --enable-early-stop / --final-answer-policy) ──
ENABLE_WARM_UP      = True          # warm-up gate 활성화 여부
ENABLE_EARLY_STOP   = True          # early-stop gate 활성화 여부
FINAL_ANSWER_POLICY = "structured"  # 최종 답 생성 정책: "generate" | "structured"
