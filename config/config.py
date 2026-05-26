"""
Configuration file for API keys and other settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# SMWU FACTCHAT API Configuration
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# API Rate Limiting Configuration
CALLS_PER_MINUTE = 20
PERIOD = 60
MAX_RETRIES = 3
RETRY_DELAY = 120

# Model Configuration
DEFAULT_MODEL = "gpt-5.4-mini"  # please specify your preferred LLM model
DEFAULT_MAX_TOKENS = 250

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # please specify your preferred embedding model
EMBEDDING_BATCH_SIZE = 32

# Cache Configuration
CACHE_DIR = "cache"
RESULT_DIR = "evaluation"

# ============================================================
# Experiment Configuration — 실험 세팅을 여기서 변경하세요
# ============================================================

# 데이터셋 선택: "hotpotqa" | "2wikimultihopqa" | "musique"
DATASET = "hotpotqa"

DATASET_PATH  = f"dataset/{DATASET}.json"
CORPUS_PATH   = f"dataset/{DATASET}_corpus.json"
OUTPUT_FILE   = f"evaluation_results_{DATASET}.json"

# 평가할 질문 수 (0 = 전체)
LIMIT = 5

# Retrieval 설정
TOP_K = 5
EVAL_TOP_KS = [5, 10]

# Dynamic DAG Adaptation 최대 횟수
MAX_ROUNDS = 3

# 체크포인트 저장 간격 (질문 수)
CHECKPOINT_INTERVAL = 5