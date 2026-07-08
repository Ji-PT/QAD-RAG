# query_decomposition (feat/query_decomposition, exp1~5)

`decompose_query()` 프롬프트팅을 통한 decomposition 품질을 비교하는 실험 모음.
공통 평가 하네스는 `src/evaluation/decomposition_eval.py`(gold decomposition과의 구조 비교: 
step 수 / step 내용 / holistic 일치율)를 사용.

## 실험 목록

| 실험 | 클래스 (`decomp_variants*.py`) | 가설 |
|---|---|---|
| baseline | `LogicRAG.decompose_query` | 8-shot 프롬프트 하나로 모든 질문 분해 |
| exp1 Entity-first CoT | `LogicRAGExpEntityCoT` | pivot entity를 먼저 식별한 뒤 분해하면 품질이 개선되는가 |
| exp2 Self-verification | `LogicRAGExpSelfVerify` | 분해 후 각 step을 자기검증+재분할하면 개선되는가 |
| exp3 Hop count 사전 추정 | `LogicRAGExpHopCount` | hop 수를 먼저 추정한 뒤 그 수에 맞춰 분해하면 개선되는가 |
| exp4 Query type classifier | `LogicRAGExpQueryTypeClassifier` | 질문 구조(chain/branching)를 먼저 분류하고 유형별 프롬프트를 쓰면 개선되는가 |
| exp4b Short few-shot | `LogicRAGExpQueryTypeClassifierShortFewshot` | exp4의 few-shot을 유형별 핵심 예시만 남겨도 되는가 — **실패**, branching 성능 급락 |
| exp4c Mid few-shot | `LogicRAGExpQueryTypeClassifierMidFewshot` | 4b 실패를 보고 branching에 nested chain/direct attribute 예시 2개만 복원 — **최고 성능** |
| exp5 No Classifier | `LogicRAGExpBranchingOnly` | 분류 단계 없이 branching(mid) 프롬프트 하나만 모든 질문에 적용해도 되는가 (LLM 호출 2회→1회) |

## 결과 (100개 데이터셋, step수 / 내용 / holistic 일치율)

| 실험 | step수 | 내용 | holistic |
|---|---|---|---|
| baseline | 70.0% | 44.0% | 83.0% |
| exp1 EntityCoT | 70.0% | 47.0% | 86.0% |
| exp2 SelfVerify | 69.0% | 48.0% | 86.0% |
| exp3 HopCount | 69.0% | 46.0% | 83.0% |
| exp4 QueryTypeClf | 75.0% | 48.0% | 88.0% |
| exp4b Short Fewshot | 69.0% | 49.0% | 84.0% |
| **exp4c Mid Fewshot** | **76.0%** | **52.0%** | **88.0%** |
| exp5 No Classifier | 69.0% | 48.0% | 85.0% |

(exp4c가 세 지표 모두에서 가장 우수해 이후 실험(모듈2)의 baseline으로 채택)

## 실행

```bash
# exp1~3
python -m src.experiments.query_decomposition.run_decomp_experiments
python -m src.experiments.query_decomposition.run_decomp_experiments --dataset dataset/musique_sample_100.json

# exp4/4b/4c/5
python -m src.experiments.query_decomposition.run_decomp_experiments_v2
python -m src.experiments.query_decomposition.run_decomp_experiments_v2 --dataset dataset/musique_sample_100.json
```

결과 파일은 `evaluation/decomp_exp_{실험명}_{샘플수}_{순번}.json`으로 저장된다.
