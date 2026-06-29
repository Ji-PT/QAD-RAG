"""
decompose_query() 결과와 gold_decomposition을 비교하는 평가 스크립트(주석생성 w.Claude).

실행:
  python -m src.evaluation.decomposition_eval
  python -m src.evaluation.decomposition_eval --dataset dataset/musique_sample_100.json

[최종 출력 구조]
결과 JSON 파일은 두 개의 최상위 키로 구성된다.

1. "summary" — 전체 집계 지표
   {
     "total": 100,               # 평가한 전체 샘플 수
     "has_gold": 100,            # gold decomposition이 존재하는 샘플 수
                                 # (MuSiQue 일부 샘플은 gold decomp 없음)

     "step_count_match_rate": 72.0,   # step 수가 gold와 일치한 비율 (%)
                                      # 예) model=2steps, gold=2steps → 일치

     "all_steps_match_rate": 65.0,    # step 수도 같고 각 step 내용도 모두 일치한 비율 (%)
                                      # step 수가 다르면 자동으로 0으로 처리됨

     "holistic_match_rate": 78.0,     # 전체 decomposition이 같은 reasoning chain을
                                      # 커버하는지 LLM이 종합 판단한 비율 (%)
                                      # step 수가 달라도 판단 가능 — 가장 너그러운 지표

     "by_hop": {                 # hop 유형별 동일 지표 분류
       "2hop": {
         "total": 40,
         "step_count_match_rate": 85.0,
         "all_steps_match_rate": 80.0,
         "holistic_match_rate": 88.0
       },
       "3hop1": { ... },
       "4hop1": { ... }
     }
   }

2. "results" — 샘플별 상세 결과 리스트
   각 항목:
   {
     "id": "2hop__13548_13529",       # MuSiQue 데이터셋 ID
     "hop_type": "2hop",              # hop 유형
     "question": "...",               # 원본 질문
     "gold_answer": "June 1982",      # 최종 정답
     "is_simple": false,              # 모델이 single-hop으로 판단했는지 여부

     "model_subproblems": [           # decompose_query()가 실제로 출력한 subproblems
       {"id": 0, "text": "Who is..."},
       {"id": 1, "text": "When was..."}
     ],

     "gold_decomposition": [          # MuSiQue 데이터셋의 gold decomposition
       {"question": "To whom was...", "gold_answer": "Diego Maradona"},
       {"question": "When was #1...", "gold_answer": "June 1982"}
     ],

     "n_model": 2,                    # 모델이 만든 subproblem 수
     "n_gold": 2,                     # gold decomposition step 수
     "step_count_match": true,        # 두 수가 일치하는지 여부

     "step_results": [                # per-step LLM judge 결과
                                      # step 수가 같을 때만 채워짐, 다르면 빈 리스트
       {
         "step": 0,
         "model": "Who is the person...",
         "gold": "To whom was Messi's goal...",
         "match": true,               # 두 subproblem이 같은 정보를 묻는지
         "reason": "Both ask for..."  # LLM의 판단 근거
       },
       { "step": 1, ... }
     ],

     "all_steps_match": true,         # step_results가 모두 match=true인지
                                      # step 수가 다르면 항상 false

     "holistic_match": true,          # 전체 decomposition을 종합적으로 비교한 LLM 판단
     "holistic_reason": "Both decompositions identify the same intermediate entity..."
   }
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List

from tqdm import tqdm

from src.models.logic_rag import LogicRAG
from src.utils.utils import get_response_with_retry, fix_json_response

DECOMP_EVAL_DATASET_PATH = "dataset/musique_sample_100.json"  # 평가할 데이터셋 경로
DECOMP_EVAL_OUTPUT_DIR   = "evaluation"                        # 결과 저장 폴더
DECOMP_EVAL_CHECKPOINT_INTERVAL = 5                            # 체크포인트 저장 간격 (샘플 수)

logger = logging.getLogger(__name__)


def _checkpoint_path(output_path: str) -> str:
    """output 파일에 대응하는 체크포인트 파일 경로."""
    ckpt_dir = os.path.join(os.path.dirname(output_path), "checkpoints")
    basename = os.path.splitext(os.path.basename(output_path))[0]
    return os.path.join(ckpt_dir, f"{basename}.checkpoint.json")


def _save_checkpoint(results: list, output_path: str) -> None:
    ckpt_path = _checkpoint_path(output_path)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "output_path": output_path}, f, ensure_ascii=False)


def _load_checkpoint(output_path: str) -> list:
    ckpt_path = _checkpoint_path(output_path)
    if os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            data = json.load(f)
        completed = data.get("results", [])
        print(f"체크포인트 발견: {len(completed)}개 완료, 이어서 진행합니다.")
        return completed
    return []


def _clear_checkpoint(output_path: str) -> None:
    ckpt_path = _checkpoint_path(output_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


def judge_step_semantic_match(model_step: str, gold_step: str) -> Dict[str, Any]:
    """두 subproblem이 같은 정보를 묻는지 LLM으로 판단한다."""
    prompt = f"""You are evaluating whether two subproblem questions ask for the same information.

Model subproblem: "{model_step}"
Gold subproblem: "{gold_step}"

Judge whether these two questions are semantically equivalent — that is, whether answering one would also answer the other.
Minor differences in phrasing are acceptable. Focus on whether the core intent and the target entity or fact are the same.

Return ONLY a JSON object:
{{
  "match": true or false,
  "reason": "one sentence explanation"
}}"""

    try:
        response = get_response_with_retry(prompt)
        response = response.strip().replace("```json", "").replace("```", "")
        result = fix_json_response(response)
        if not isinstance(result, dict):
            return {"match": False, "reason": "parse_error"}
        return {
            "match": bool(result.get("match", False)),
            "reason": str(result.get("reason", "")),
        }
    except Exception as e:
        logger.error(f"judge_step_semantic_match error: {e}")
        return {"match": False, "reason": f"error: {e}"}


def judge_holistic_match(
    question: str,
    model_texts: List[str],
    gold_texts: List[str],
) -> Dict[str, Any]:
    """전체 decomposition이 동등한지 LLM으로 판단한다. step 수가 달라도 사용 가능."""
    prompt = f"""You are evaluating the quality of a question decomposition.

Original question: "{question}"

Model decomposition ({len(model_texts)} steps):
{json.dumps(model_texts, ensure_ascii=False, indent=2)}

Gold decomposition ({len(gold_texts)} steps):
{json.dumps(gold_texts, ensure_ascii=False, indent=2)}

Judge whether the model decomposition covers the same reasoning chain as the gold decomposition.
The model decomposition is acceptable if:
- It identifies the same key intermediate entities and facts.
- It would lead to the same final answer through retrieval.
- Minor differences in phrasing or step ordering are acceptable.

Return ONLY a JSON object:
{{
  "match": true or false,
  "reason": "one sentence explanation"
}}"""

    try:
        response = get_response_with_retry(prompt)
        response = response.strip().replace("```json", "").replace("```", "")
        result = fix_json_response(response)
        if not isinstance(result, dict):
            return {"match": False, "reason": "parse_error"}
        return {
            "match": bool(result.get("match", False)),
            "reason": str(result.get("reason", "")),
        }
    except Exception as e:
        logger.error(f"judge_holistic_match error: {e}")
        return {"match": False, "reason": f"error: {e}"}


def evaluate_decomposition(
    question: str,
    model_subproblems: List[Dict],
    gold_decomposition: List[Dict],
) -> Dict[str, Any]:
    """단일 질문의 decomposition 결과를 평가한다."""
    n_model = len(model_subproblems)
    n_gold = len(gold_decomposition)
    step_count_match = n_model == n_gold

    model_texts = [s["text"] for s in model_subproblems]
    gold_texts = [g["question"] for g in gold_decomposition]

    # per-step judge: step 수가 같을 때만 실행
    # step 수가 다르면 어떤 step끼리 비교해야 할지 알 수 없으므로 skip
    step_results = []
    if step_count_match and n_model > 0:
        for i, (mt, gt) in enumerate(zip(model_texts, gold_texts)):
            judge = judge_step_semantic_match(mt, gt)
            step_results.append({
                "step": i,
                "model": mt,
                "gold": gt,
                "match": judge["match"],
                "reason": judge["reason"],
            })

    # all_steps_match: step 수도 같고 모든 step이 의미상 일치할 때만 true
    all_steps_match = (
        step_count_match and bool(step_results) and all(r["match"] for r in step_results)
    )

    # holistic judge: step 수와 무관하게 전체 decomposition을 종합 비교
    holistic = judge_holistic_match(question, model_texts, gold_texts)

    return {
        "n_model": n_model,
        "n_gold": n_gold,
        "step_count_match": step_count_match,
        "step_results": step_results,
        "all_steps_match": all_steps_match,
        "holistic_match": holistic["match"],
        "holistic_reason": holistic["reason"],
    }


def _next_output_path(dataset_path: str, output_dir: str) -> str:
    """evaluation/{출력dir}/decomposition_eval_results_{샘플수}_{순번}.json 형식으로 파일명 자동 생성."""
    with open(dataset_path, encoding="utf-8") as f:
        n = len(json.load(f))

    os.makedirs(output_dir, exist_ok=True)
    prefix = f"decomposition_eval_results_{n}_"
    existing = [
        fname for fname in os.listdir(output_dir)
        if fname.startswith(prefix) and fname.endswith(".json")
    ]
    next_idx = len(existing) + 1
    return os.path.join(output_dir, f"{prefix}{next_idx}.json")


def run_decomposition_eval(
    dataset_path: str = DECOMP_EVAL_DATASET_PATH,
    output_path: str = None,
    model=None,
) -> None:
    if output_path is None:
        output_path = _next_output_path(dataset_path, DECOMP_EVAL_OUTPUT_DIR)

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    if model is None:
        model = LogicRAG()

    # 체크포인트에서 이어 시작
    results = _load_checkpoint(output_path)
    completed_ids = {r["id"] for r in results}
    remaining = [item for item in dataset if item["id"] not in completed_ids]

    for i, item in enumerate(tqdm(remaining, desc="Evaluating decomposition",
                                  initial=len(results), total=len(dataset))):
        question = item["question"]
        hop_type = item["id"].split("__")[0]
        gold_decomposition = [
            {"question": d["question"], "gold_answer": d["answer"]}
            for d in item.get("question_decomposition", [])
        ]

        decomp = model.decompose_query(question)
        model_subproblems = decomp.get("subproblems", [])
        is_simple = decomp.get("is_simple", False)

        if gold_decomposition:
            eval_result = evaluate_decomposition(question, model_subproblems, gold_decomposition)
        else:
            eval_result = {
                "n_model": len(model_subproblems),
                "n_gold": 0,
                "step_count_match": None,
                "step_results": [],
                "all_steps_match": None,
                "holistic_match": None,
                "holistic_reason": "no gold decomposition",
            }

        results.append({
            "id": item["id"],
            "hop_type": hop_type,
            "question": question,
            "gold_answer": item["answer"],
            "is_simple": is_simple,
            "model_subproblems": model_subproblems,
            "gold_decomposition": gold_decomposition,
            **eval_result,
        })

        # 체크포인트 저장
        if (i + 1) % DECOMP_EVAL_CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(results, output_path)

    # 집계
    has_gold = [r for r in results if r["n_gold"] > 0]
    by_hop: Dict[str, Dict[str, int]] = {}
    for r in has_gold:
        hop = r["hop_type"]
        by_hop.setdefault(hop, {"total": 0, "step_count": 0, "all_steps": 0, "holistic": 0})
        by_hop[hop]["total"] += 1
        if r["step_count_match"]:
            by_hop[hop]["step_count"] += 1
        if r["all_steps_match"]:
            by_hop[hop]["all_steps"] += 1
        if r["holistic_match"]:
            by_hop[hop]["holistic"] += 1

    def pct(n, d):
        return round(n / d * 100, 1) if d else 0.0

    n = len(has_gold)
    summary = {
        "total": len(results),
        "has_gold": n,
        "step_count_match_rate": pct(sum(1 for r in has_gold if r["step_count_match"]), n),
        "all_steps_match_rate": pct(sum(1 for r in has_gold if r["all_steps_match"]), n),
        "holistic_match_rate": pct(sum(1 for r in has_gold if r["holistic_match"]), n),
        "by_hop": {
            hop: {
                "total": v["total"],
                "step_count_match_rate": pct(v["step_count"], v["total"]),
                "all_steps_match_rate": pct(v["all_steps"], v["total"]),
                "holistic_match_rate": pct(v["holistic"], v["total"]),
            }
            for hop, v in sorted(by_hop.items())
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    _clear_checkpoint(output_path)

    print("\n=== Decomposition Evaluation Summary ===")
    print(f"총 샘플: {summary['total']} (gold 있음: {summary['has_gold']})")
    print(f"step 수 일치율   : {summary['step_count_match_rate']}%")
    print(f"step 내용 일치율 : {summary['all_steps_match_rate']}%")
    print(f"holistic 일치율  : {summary['holistic_match_rate']}%")
    print("\nhop별:")
    for hop, v in summary["by_hop"].items():
        print(f"  {hop:8s} | step수={v['step_count_match_rate']}% | 내용={v['all_steps_match_rate']}% | holistic={v['holistic_match_rate']}% (n={v['total']})")
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Evaluate decompose_query() against gold decompositions.")
    parser.add_argument("--dataset", default=DECOMP_EVAL_DATASET_PATH)
    parser.add_argument("--output", default=None, help="출력 경로 직접 지정 (기본: 자동 생성)")
    args = parser.parse_args()
    run_decomposition_eval(args.dataset, args.output)
