# KV Cache 최적화 기술 다관점 평가

> **브랜치 안내**
> - `feat/multi-agent-orchestration` : **Agent 패턴 실습(Orchestrator-Workers) 브랜치**. 이번 실습의 작업과 PR은 이 브랜치로 머지한다.
> - `main` : 이전 **RAG 실습** 결과(순차·병렬 고정 흐름).

KV 캐시 병목을 상반된 방식으로 푸는 두 기술을 네 관점에서 평가하고, 모든 주장에 검증된 근거를 붙인 보고서를 생성하는 LangGraph Orchestrator-Workers + RAG 파이프라인.

## Quick Start

```bash
git clone git@github.com:Jieun1ee/RAG-pipeline.git && cd RAG-pipeline
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # OPENAI_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY 입력
python -m rag.indexer    # 색인 생성 (한 번만)
python app.py            # 전체 실행 → outputs/report.md
```

API 키 없이 흐름만 확인하려면 `python app.py --dry-run`. 옵션 전체는 [Usage](#usage) 참고.

## Requirements

| 항목 | 값 |
|---|---|
| Python | 3.11 |
| API 키 | `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`(추적용) |
| 임베딩 모델 | BAAI/bge-m3. 첫 실행 시 약 2GB 내려받아 로컬에서 돌린다 |
| 색인 생성 | 논문 2편 65쪽 → 청크 187개, 20초 안팎 |
| 전체 실행 | 10~15분 안팎. 오케스트레이터가 보낸 worker 수와 재계획 횟수에 따라 달라진다 |

임베딩은 로컬에서 돌아 비용이 들지 않는다. 비용이 발생하는 것은 OpenAI 호출과 Tavily 검색이다.

## Subject

데이터센터 기반 LLM Serving에서 KV 캐시 병목을 상반된 방식으로 푸는 두 기술, **MLA(SW, 데이터를 작게)** 와 **ITME(HW, 담을 공간을 넓게)** 를 기술 성숙도·시장성·이해관계자·도메인 적용성 네 관점에서 평가하는 **Orchestrator-Workers** 패턴 기반으로 설계·개발한 프로젝트. 우열을 가리지 않고, 관점에 따라 평가가 어떻게 갈리는지를 근거와 함께 제시한다.

## Overview

- **Objective** : 하나의 기술을 복수 관점에서 중립적으로 비교 평가하고, 모든 주장에 검증된 근거를 붙인 보고서를 생성한다.
- **Pattern** : Orchestrator-Workers. 평가 작업은 "기준 1개 × 기술 1개" 단위로 서로 의존하지 않아 병렬 분할에 맞다. 또 두 기술에 같은 기준을 적용하는 중립성 원칙을 계획 단계에서 코드로 강제할 수 있다. Supervisor처럼 매 스텝 LLM이 다음 에이전트를 고르는 방식보다 호출 수가 적고, 같은 계획이면 같은 작업 목록이 나와 재현성 관리가 쉽다.
- **동적 처리** : RAG 실습에서는 관점 노드 4개를 엣지로 고정해 항상 같은 순서·같은 개수로 돌았다. 이번에는 무엇을 몇 개 보낼지와 언제 멈출지를 실행 중 State를 보고 정한다.
  1. **계획** : 오케스트레이터(LLM)가 기술 조사 결과를 보고 관점별로 조사할 기준과 기준마다 출처(논문/웹)·검색 초점을 고른다. 첫 fan-out 수가 실행마다 24~56개 사이에서 달라진다.
  2. **재계획** : worker 결과를 모은 뒤 빈 곳(한쪽 근거만 나온 관점, 근거를 끝내 못 찾은 task)만 골라 추가 task를 보낸다. 빈 곳이 없으면 바로 종합으로 넘어간다.
  3. **품질 보완** : 보고서 품질 평가가 근거 부족을 확인하면 해당 관점·기술만 다시 조사하고, 표현 문제면 보고서만 다시 쓴다.
- **Tools** : LangGraph, OpenAI(Generator/Judge 분리), BAAI/bge-m3, FAISS, Tavily, PyMuPDF, LangSmith
- **Domain** : 데이터센터 기반 LLM Serving. 대규모 동시 요청과 비용에 민감한 환경이라 KV 캐시 용량과 처리량의 트레이드오프가 가장 직접적으로 드러난다.

## Selected Technologies

기술 선정은 Human 기반(2안)으로, 과제 Doc Pool에서 진영별 1건씩 직접 골랐다. 두 논문 합계 65쪽으로 200쪽 제한 안이다.

| 진영 | 기술 | 선정 이유 | 원문 |
|---|---|---|---|
| SW | MLA (Multi-head Latent Attention) | KV를 저차원 잠재 공간으로 압축해 캐시를 93.3% 줄이는 어텐션 구조 재설계. 사후 압축과 달리 구조 자체를 바꾸는 접근이고, 공개 모델과 논문으로 구현·수치가 모두 공개되어 근거 확보가 가능하다 | DeepSeek-AI(2024). *DeepSeek-V2*. arXiv, 2405.04434 |
| HW | ITME (Inference Tiered Memory Expansion) | CXL 하이브리드 메모리로 KV 캐시를 GPU 메모리 밖으로 계층 확장. 원본을 줄이지 않고 담을 공간을 넓히는 정반대 접근이며, 메모리 제조사가 제안해 HW 진영을 대표한다 | Jang, H. et al.(2026). *ITME*. arXiv, 2606.12556 |

## Evaluation Criteria

평가 기준은 임의로 만들지 않고 **공개된 표준과 프레임워크의 항목에서 가져왔다.** 기준을 직접 만들면 두 기술 중 하나에 유리한 항목이 섞여도 알아채기 어렵기 때문이다.

| 관점 | 평가 대상 | 근거가 된 프레임워크 | 결과 표현 |
|---|---|---|---|
| 기술 성숙도 | 개발 및 실제 적용 수준 | NASA TRL 정의 (단계별 HW·SW 설명) | TRL 범위 또는 판단 유보 |
| 시장성 | 산업 내 채택 및 확산 가능성 | 미국 에너지부 ARL의 채택 위험 차원 | 차원별 채택 위험 수준 |
| 이해관계자 | 도입으로 영향을 받는 주체 | Rogers 혁신 확산 이론의 혁신 속성 | 주체별 기대 효과와 도입 부담 |
| 도메인 적용성 | LLM Serving 환경에서의 적용 가능성 | ISO/IEC 25010 제품 품질 특성, MLPerf Inference 측정 지표 | 항목별 효과와 제약, 보고 수치 |

### 지표 선정 이유

**기술 성숙도 — NASA TRL.** SWOT처럼 강하다·약하다를 따지는 틀은 상대적이라 두 기술을 나란히 놓는 순간 우열로 읽힌다. TRL은 "지금 몇 단계인가"라는 절대 위치를 준다. NASA 정의가 단계마다 하드웨어 설명과 소프트웨어 설명을 따로 두고 있어, SW 기술인 MLA와 HW 기술인 ITME에 같은 척도를 적용할 수 있다는 점도 컸다. 단, KV 캐시 기술은 논문 발표와 실제 채택 사이에 시차가 있어 모든 판정에 공개 정보 기반 추정임을 명시한다.

**시장성 — 미국 에너지부 ARL.** 시장성을 시장 규모로만 보면 기술 자체의 채택 가능성이 아니라 산업 전체의 전망을 재게 된다. ARL은 채택을 막는 요인을 수요 성숙도, 가치 제안, 자원 성숙도, 운영 허가 같은 차원으로 나눠 둬서, 어디가 장벽인지를 지목할 수 있다. "위험이 높다"가 아니라 "무엇 때문에 높다"가 나온다.

**이해관계자 — Rogers 혁신 확산 이론.** 같은 기술도 주체마다 반응이 다른 이유를 설명하는 틀이다. 기대 효과는 상대적 이점(relative advantage)에, 도입 부담은 호환성(compatibility)과 복잡성(complexity)에 대응한다. 두 축을 분리했기 때문에 **효과도 크고 부담도 큰 상태를 그대로 기록**할 수 있다. 하나로 합치면 이 구간이 사라진다. 도입 주체(STK-1~3)만 두 축으로 보고, 경쟁 진영과 투자 업계 같은 관찰 주체(STK-4~5)는 축을 적용하지 않고 반응을 출처와 함께 기록한다.

**도메인 적용성 — ISO/IEC 25010과 MLPerf Inference.** 적용성 항목을 직접 나열하면 무엇이 빠졌는지 알 수 없다. 국제 표준의 품질 특성에 항목을 대응시키면 커버리지가 보장된다. 메모리 효율은 Resource Utilization, 처리량은 Capacity, 지연은 Time Behaviour, 정확도는 Functional Correctness에 대응한다. 처리량과 지연의 **측정 방식**은 MLPerf Inference를 기준으로 삼는다. 지연 제약을 지키면서 감당하는 최대 처리량을 쓰고, LLM은 TTFT와 TPOT 두 제약을 본다. 그래서 메모리 절감과 처리량은 항상 정확도 변화와 함께 기록한다.

## Features

- **관점별 평가 기준 33개를 데이터로 분리** : 기술 조사(TECH-1~5), 기술 성숙도(TRL-1~9), 시장성(MKT-1~6), 이해관계자(STK-1~5), 도메인 적용성(DOM-1~8). `data/criteria/*.yaml`에 있어 코드를 고치지 않고 기준을 바꿀 수 있다.
- **오케스트레이터 계획 규칙** : 어떤 기준을 조사할지는 LLM이 고르고, 코드는 세 가지만 강제한다. 고른 기준은 두 기술 모두에 적용(대칭), 관점마다 최소 1개, 기술 성숙도는 9단계 전체(단계를 건너뛰면 TRL 구간 계산이 틀어진다). 재계획은 첫 계획에서 고른 기준 안에서만 메운다.
- **근거 2단 검증** : 인용이 검색 결과에 실제로 있는지는 코드로 문자열 대조하고, 그 인용이 주장을 뒷받침하는지는 Judge 모델이 판정한다. 둘을 통과하지 못한 서술은 보고서에 들어가지 않는다.
- **짧은 재시도 루프** : 실패는 발생한 단계 안에서 해결한다. 상한에 도달하면 멈추지 않고 "미확인 항목"으로 기록한 뒤 다음 단계로 넘어간다.
- **worker 실패 fallback** : 예외로 끝난 task(error)는 다음 회차 전에 1회 다시 보낸다(재시도). 그래도 실패하거나 근거를 찾지 못한 task(gave_up)는 결과에서 빼고 사유를 gap으로 남긴 채 나머지로 계속 진행한다(제외 후 계속).
- **TRL 구간 규칙 계산** : 단계별 근거 수준(direct/indirect/none)은 모델이 판정하고, 구간은 코드가 규칙으로 계산한다. 직접 근거가 없으면 판단을 유보한다. 모든 TRL 서술에 공개 정보 기반 추정임을 명시한다.
- **확증편향 방지 전략** :
  - *동일 기준 적용* — 두 기술에 같은 기준을 적용하고(코드로 대칭 강제), 해당하지 않는 기준은 빼지 않고 `not_applicable`과 이유를 기록한다. 오케스트레이터가 이번 실행에서 다루지 않은 기준은 보고서 한계 절에 밝힌다.
  - *상충 의견 보존* — 같은 기준에서 반대 방향으로 나온 서술을 `dissent`로 따로 남겨 종합과 보고서가 볼 수 있게 한다.
  - *균형 검사* — 기술마다 강점과 한계가 모두 나왔는지 확인하고, 한쪽이 비면 미확인 항목으로 적는다.
  - *우열 표현 검사* — 종합과 보고서 단계에서 Judge 모델이 우열·순위·추천 표현을 찾아낸다. 점수를 매기거나 합산하지 않는다.
- **보고서 품질 평가 (Hybrid)** : 보고서를 쓴 뒤 `report_check` 노드가 형식(SUMMARY 맨 앞, REFERENCE 맨 뒤 등)을 확인하고, 네 항목을 코드 규칙과 Judge로 함께 판정한다. 규칙과 Judge 중 하나라도 미달이면 그 항목은 미달이다.
  - *Groundedness* — 본문 인용이 REFERENCE와 맞고, 검증된 Evidence에서 만든 번호인가
  - *중립성* — 추천·우열 표현이 없는가 (명백한 표현은 코드, 문맥상 우열은 Judge)
  - *편향 통제* — 관점·기술마다 강점과 한계 근거가 모두 있고, 근거가 개발사 자체 발표뿐이 아닌가
  - *관점 커버리지* — 네 관점의 절과 판정 결과가 모두 있는가

  미달이면 원인에 따라 보고서를 다시 쓰거나(rewrite), 근거가 부족한 관점·기술만 오케스트레이터로 돌려 다시 조사한다(replan). 보고서는 최대 3번, 근거 재조사는 최대 1번이다.
- **재현성** : 온도 0, 응답 파일 캐시, 설정으로 고정한 재시도·재계획 상한, SQLite 체크포인트(`--resume`). `--dry-run`으로 외부 호출 없이 전체 흐름을 확인할 수 있다.

## Tech Stack

| 구분 | 사용 |
|---|---|
| Framework | LangGraph 1.x (StateGraph, 조건부 엣지, `Send` 동적 fan-out, 서브그래프, SQLite 체크포인트) |
| LLM/Generator | `gpt-4.1-mini` — 계획·재계획, 질의 생성, 서술 생성, 수준 판정, 종합, 보고서 작성 |
| LLM/Judge | `gpt-4o-mini` — 검색 관련성, 근거 타당성, 우열 표현 검사, 보고서 품질 판정. 생성과 다른 계열로 둬 자기 글을 그대로 통과시키지 않게 했다 |
| Retrieval | FAISS(IndexFlatIP) + 자체 sparse 점수, RRF 앙상블 — Hit Rate@5 **0.967**, MRR@5 **0.892** |
| Embedding | BAAI/bge-m3 (오픈소스, MIT) |
| Web Search | Tavily |
| PDF | PyMuPDF (추출), ReportLab (보고서 PDF) |
| Schema | Pydantic v2 (구조화 출력) |
| Tracing | LangSmith (실행 트리, 결정 로그), `outputs/logs/decisions.jsonl` |

### Embedding 모델 선정

리더보드 순위는 근거로 쓰지 않았다. 범용 벤치마크의 평균이라 이 과제의 문서와 질의 조건을 반영하지 못한다. 대신 네 가지 조건을 기준으로 삼았다.

| 기준 | 이 과제에서의 조건 |
|---|---|
| 언어 | 작업 지시와 결과는 한국어, 문서는 영어 논문. 질의에 한국어가 섞여도 영어 본문을 찾아야 한다 |
| 도메인 용어 | MLA, CXL, HBM, NVMe, TTFT 같은 약어가 많다. 의미 검색만으로는 약어가 정확히 일치하는 본문을 놓친다 |
| 문서 길이 | 논문의 절 단위 청크를 잘리지 않고 한 번에 인코딩할 수 있어야 한다 |
| 실행 비용 | 오픈소스 가중치로 로컬 실행. API 비용이 없고 노트북에서 돌아가는 규모 |

설계 방향이 서로 다른 후보 세 개를 놓고 비교했다.

| 후보 | 유형 | 판단 |
|---|---|---|
| **BAAI/bge-m3** | 다국어 하이브리드 검색형. 1,024차원, 8,192토큰, 100개 이상 언어 | **선정.** 한 번의 인코딩으로 의미 벡터와 토큰 가중치를 함께 출력한다 |
| Qwen/Qwen3-Embedding-0.6B | 지시문 기반 dense 검색형. 32K 토큰 | 언어·길이·규모 조건은 충족하나 dense만 출력한다. 같은 구성을 만들려면 키워드 검색을 따로 붙여야 한다 |
| allenai/SPECTER2 | 과학 논문 특화형. 512토큰, 영어 | 제목·초록으로 유사 논문을 찾도록 학습되어 질문에 맞는 본문 단락을 찾는 용도와 맞지 않는다 |

**bge-m3를 고른 결정적 이유는 도메인 용어 조건이다.** 질의의 약어를 문서의 같은 약어와 직접 맞출 수 있고, 별도의 BM25 색인이나 한국어 형태소 분석기 없이 하이브리드 검색을 구성할 수 있다.

### RAG 파이프라인

RAG는 **기술 조사**와 **도메인 적용성**의 기본 출처다. 두 관점이 필요로 하는 근거인 동작 방식, 한계, 실험 환경과 보고 수치가 논문 원문에 있기 때문이다. 시장성·이해관계자·기술 성숙도가 다루는 채택 사례, 생태계 동향, 상용 운용 여부는 논문에 실리지 않고 시기에 따라 바뀌므로 웹 검색이 기본이다. 오케스트레이터는 기준마다 이 기본값을 바꿀 수 있다.

| 단계 | 처리 | 결과 |
|---|---|---|
| 적재 | PDF에서 본문과 표를 추출하고, 모든 쪽에 반복되는 머리글·바닥글을 걷어낸다 | 문서 |
| 분할 | 페이지 단위로 먼저 나눈 뒤 절 제목이 보이면 그 경계로 조정한다. 문단 경계에서만 자른다 | 청크 |
| 메타데이터 | 청크마다 문서 id, 쪽 번호, 절 제목, 쪽 안 순번을 붙인다 | 메타데이터가 붙은 청크 |
| 색인 | bge-m3로 dense 벡터와 sparse 가중치를 함께 만든다. dense는 정규화해 FAISS에 저장 | 인덱스 |
| 검색 | dense 상위 2k와 sparse 상위 2k를 각각 구해 RRF로 합쳐 상위 k를 돌려준다 | 검색 결과 |

**청킹 전략.** 고정 길이로 자르지 않는다. 문장 중간에서 잘리면 인용문이 두 청크에 걸쳐 원문 대조가 실패하기 때문이다. 페이지 단위에서 출발해 문단 경계에서만 자르고, 절 제목을 만나면 거기서 끊어 새 청크를 시작한다. 절 제목은 이후 청크에 계속 따라붙어 검색 결과만 보고도 문서의 어느 부분인지 알 수 있다. 상한 1,500자를 넘기 전에 문단 경계에서 끊고, 120자 미만 조각은 표·그림 캡션으로 보고 버린다. 논문 2편 65쪽이 187개 청크가 됐다.

**인용 대조용 정규화.** PDF 추출 텍스트에는 줄 끝에서 잘린 단어, 두 칸 공백, 굽은 따옴표가 섞인다. 사람 눈에 같은 문장도 글자로는 다르다. 인용문과 청크 본문 양쪽에 같은 정규화를 걸어야 멀쩡한 인용이 탈락하지 않는다.

### Retrieval 평가

청크 30개를 뽑아 각 청크로만 답할 수 있는 질문을 Generator로 만들고, 그 청크를 정답으로 하는 골든셋을 구성했다. 세 구성을 같은 질문으로 비교했다.

| 구성 | Hit Rate@5 | MRR@5 | 해석 |
|---|---|---|---|
| Dense | 0.967 | 0.872 | 30개 중 29개를 상위 5개 안에서 검색 |
| Sparse | 0.900 | 0.867 | 27개 검색. 상대적으로 누락이 많음 |
| **RRF** | **0.967** | **0.892** | 29개를 검색하면서 정답 순위도 가장 높음 |

RRF는 Dense와 같은 검색 성공률을 유지하면서 정답 청크를 더 높은 순위에 배치했다. 하이브리드 검색의 효과가 확인되어 최종 검색 방식으로 RRF를 채택했다.

## Agents

조정 계층(오케스트레이터)과 실행 계층(worker)을 나눴다. worker끼리는 서로 통신하지 않고, 결과는 State를 거쳐 collect와 오케스트레이터로만 모인다.

| 노드 | 역할 | 결과 |
|---|---|---|
| `select_tech` | 평가 대상, 평가 기준, 도메인을 상태에 올린다 | selected_tech, criteria, target_domain |
| **`orchestrator`** | 회차마다 보낼 task를 정한다. ① 기술 조사(TECH-1~5 × 2기술, 코드가 전부 보냄) ② 관점 평가 계획(LLM이 기준·출처·초점 선택) ③ 빈 곳 재계획 ④ 보고서 품질 보완 ⑤ 종료. 결정과 사유는 결정 로그로 남긴다 | plan, task_status |
| `worker` | task 하나(기준 1 × 기술 1)를 서브그래프로 처리한다. `Send`로 계획된 수만큼 동시에 뜬다 | task_results, task_status, errors |
| `collect` | 한 회차 worker 결과를 관점별로 묶어 판정·균형 검사·TRL 추정을 하고 임시 결과를 비운다 | tech_research, trl_eval, market_eval, stakeholder_eval, domain_eval |
| `synthesis` / `synthesis_check` | 관점 종합과 검사 | synthesis, synthesis_check |
| `report` / `report_check` | 보고서 생성과 품질 평가(Hybrid 4항목) | final_report, report_check |

worker가 맡는 관점과 검색 기본값은 아래와 같다. 오케스트레이터가 기준마다 출처를 바꿀 수 있다.

| 관점 | 평가 내용 | 기준 | 기본 출처 | 결과 |
|---|---|---|---|---|
| 기술 조사 | 기술별 개요, 메커니즘, 적용 범위, 한계, 보고 수치 | TECH-1~5 | RAG | PerspectiveResult |
| 기술 성숙도 | 단계별 근거 수준 판정과 TRL 구간 추정 | TRL-1~9 | Web | TRLResult |
| 시장성 | 채택 위험 수준(ARL 기반)과 근거 | MKT-1~6 | Web | PerspectiveResult |
| 이해관계자 | 주체별 기대 효과와 도입 부담 | STK-1~5 | Web | PerspectiveResult |
| 도메인 적용성 | 성능·확장성·비용·품질 영향 | DOM-1~8 | RAG | PerspectiveResult |

## State Schema

`core/state.py`. 전체 그래프가 공유하는 `MainState`와 worker 하나가 쓰는 `WorkerState` 두 층으로 나눴다.

- **제어 vs 페이로드 분리** : `MainState`를 작업 결과(관점별 결과, 종합, 보고서)와 제어 메타(`run_id`, `step_count`, `plan`, `replan_count`, `quality_replan_count`, `task_status`, `errors`)로 구분했다. 조건부 엣지의 라우터(`dispatch`, 종합·보고서 검사 뒤 분기)는 제어 메타와 검사 결과만 읽는다. 오케스트레이터는 결과를 읽고 계획을 세우되, 무엇을 보냈고 어디까지 끝났는지는 제어 메타(`plan`, `task_status`)에 남긴다.
- **관측성 위치** : 계획·재계획·종료 같은 결정과 사유는 State에 넣지 않는다. `{run_id, node, decision, reason, ts}` 형식으로 `outputs/logs/decisions.jsonl`에 남기고 LangSmith에도 보낸다.
- **지속성 비용** : 검색 원문 같은 큰 중간값은 `WorkerState` 안에서만 쓰고 버린다. 메인으로는 검증을 통과한 findings와 gaps만 올라온다. `task_results`는 collect가 관점별로 묶은 뒤 `merge_or_reset` reducer로 비워 체크포인트마다 쌓이지 않게 한다.
- **상관** : 실행마다 `run_id` 하나를 만들어 State, 로그, 결정 로그, LangSmith metadata, 체크포인트 `thread_id`에 같은 값으로 쓴다.
- **재개/복구** : SQLite 체크포인터가 단계마다 State를 저장한다. `task_status`(pending/done/gave_up/error와 보낸 횟수)와 `errors`가 있어, `--resume RUN_ID`로 이어 돌리면 끝나지 않은 task만 다시 보낸다.
- **동시 처리** : 여러 worker가 동시에 쓰는 `task_results`, `task_status`, `errors`는 task_id를 키로 합치는 reducer를 붙였다. 같은 task를 다시 돌려도 결과가 두 번 쌓이지 않는다. `step_count`는 `operator.add`로 합산한다. worker는 이 세 키만 반환한다.
- **종료 보장** : 상한을 여러 겹으로 둔다. worker 내부 재시도(검색·인용 각 2회), error task 재전송(1회), 재계획(`max_replans`), 품질 재조사(`max_quality_replans`), 종합·보고서 재작성(각 2회), 메인 노드 실행 수(`max_steps`), LangGraph `recursion_limit`.

## Architecture

### 메인 그래프

`graph.py`의 컴파일된 그래프(`graph.get_graph().draw_mermaid()`)와 같은 연결이다.

```mermaid
flowchart TD
    S([시작]) --> A(기술 선정<br/>select_tech)
    A --> O(오케스트레이터<br/>orchestrator<br/>계획·재계획·품질 보완·종료 판단)
    O -.->|"pending task를 Send × N<br/>(N은 계획에 따라 달라짐)"| W(worker<br/>기준 1 × 기술 1)
    W --> C(관점별 집계<br/>collect)
    C --> O
    O -.->|보낼 task 없음| F(평가 종합<br/>synthesis)
    F --> FC(종합 검사<br/>synthesis_check)
    FC -.->|미통과, 상한 미만| F
    FC -.->|통과 또는 상한 도달| G(평가 보고서 생성<br/>report)
    G --> GC(보고서 품질 평가<br/>report_check · Hybrid 4항목)
    GC -.->|rewrite: 표현·구성 미달| G
    GC -.->|replan: 근거 부족 관점·기술| O
    GC -.->|통과 또는 상한 도달| Z([종료])
    classDef orch fill:#FCE7F3,stroke:#BE185D
    classDef worker fill:#EEEDFE,stroke:#534AB7
    classDef check fill:#FAEEDA,stroke:#BA7517
    class O orch
    class W,C worker
    class FC,GC check
```

오케스트레이터는 회차마다 계획을 `plan`에 올리고, pending인 task만 worker로 보낸다. 같은 회차의 worker는 동시에 돌고(최대 8개), 모두 끝나면 collect가 한 번 돌아 오케스트레이터로 돌아간다. 점선은 State를 보고 정하는 조건부 분기다.

### worker 내부 서브그래프

모든 관점의 worker가 같은 구조를 쓰며, task의 출처(`paper`/`web`)에 따라 검색 도구와 인용 확인 규칙만 다르다.

```mermaid
flowchart TD
    S([시작]) --> Q(검색 질의 생성)
    Q --> R(검색<br/>RAG: 벡터 검색 / 웹: 웹 검색)
    R --> GR(검색 결과 관련성 채점<br/>Judge)
    GR -->|부족, 상한 미만| RW(질의 재작성)
    RW --> R
    GR -->|관련 있음| GEN(Finding 생성<br/>주장 + 인용)
    GEN --> CITE(인용 실재 확인<br/>코드)
    CITE -->|미통과, 상한 미만| GEN
    CITE -->|통과| JUDGE(근거 타당성 판정<br/>Judge)
    JUDGE -->|미통과, 상한 미만| GEN
    JUDGE -->|통과| OUT([결과 반환])
    GR -->|부족, 상한 도달| GAP(미확인으로 기록)
    CITE -->|미통과, 상한 도달<br/>Finding 제거| GAP
    JUDGE -->|미통과, 상한 도달<br/>Finding 제거| GAP
    GAP --> OUT
    classDef step fill:#EEEDFE,stroke:#534AB7,color:#26215C
    classDef check fill:#FAEEDA,stroke:#BA7517,color:#412402
    classDef node0 fill:#F1EFE8,stroke:#888780,color:#2C2C2A
    class Q,R,RW,GEN step
    class GR,CITE,JUDGE check
    class S,OUT,GAP node0
```

### 검증 지점

| 위치 | 코드로 검사 | Judge로 검사 | 상한 도달 시 |
|---|---|---|---|
| 검색 결과 관련성 | - | 결과가 평가 기준과 직접 관련되는가 | 미확인으로 기록하고 반환 |
| 인용 검증 | 인용이 검색 결과 안에 실제로 있는가, 필수 항목이 채워졌는가, TRL에 추정 문구가 있는가 | 인용이 주장을 뒷받침하는가 | 해당 Finding을 제거하고 미확인으로 기록 |
| 종합 검사 | 참조한 근거가 실재하는가, 항목마다 서로 다른 관점 2개 이상인가, 반대 의견이 빠지지 않았는가 | 우열을 판정하는 표현이 있는가 | 미해결 항목을 보고서 6장에 적고 진행 |
| 보고서 품질 평가 | 형식(SUMMARY 맨 앞·REFERENCE 맨 뒤, SUMMARY 분량, TRL 문구), Groundedness(인용·REFERENCE·Evidence 대응), 중립성(우열 표현 후보), 편향 통제(강점·한계 균형, 자체 발표 편중), 관점 커버리지(절·결과 존재) | 네 항목을 보고서와 근거 요약을 함께 보고 판정 | 미통과 상태를 남기고 종료 |

## Directory

```
.
├── app.py                    실행 진입점. 명령줄 인자를 읽어 그래프나 노드 하나를 돌린다
├── report_pdf.py             마크다운 보고서를 한글 지원 PDF로 변환한다
├── graph.py                  메인 그래프 조립. 노드와 조건부 엣지를 붙여 컴파일한다
├── config.yaml               모델 이름, 재시도·재계획 상한, 동시 실행 수, 기준 선택 규칙
├── .env.example              OpenAI·Tavily·LangSmith 환경 변수 예시
├── requirements.txt
│
├── core/                     계약과 로더. 모든 모듈이 공유한다
│   ├── schemas.py            노드 사이를 오가는 값의 형식
│   ├── state.py              MainState(페이로드·제어 메타, reducer)와 WorkerState
│   ├── config.py             설정 로더. 명령줄 인자가 설정값을 덮어쓴다
│   ├── llm.py                생성·판정 모델 호출 창구. 구조화 출력과 캐시
│   ├── cache.py              응답 파일 캐시
│   ├── criteria.py           평가 기준 로더
│   ├── prompts.py            프롬프트 템플릿 로더
│   └── tracing.py            LangSmith 추적, run_id metadata, 결정 로그
│
├── agents/                   노드 함수
│   ├── orchestrator.py       조정 계층. 계획·재계획·품질 보완·종료 판단, Send 분배
│   ├── subgraph.py           worker 서브그래프 (질의 → 검색 → 관련성 → 서술 → 인용 → 근거 판정)
│   ├── perspective.py        worker 실행(run_task), 관점 판정·균형 검사·TRL 구간 계산
│   ├── collect.py            회차별 worker 결과를 관점별로 집계
│   ├── synthesis.py          관점 종합
│   ├── report.py             보고서 생성, 참고문헌 조립
│   └── checks.py             인용 실재, 근거 타당성, 종합 검사, 보고서 품질 평가
│
├── rag/                      문서 검색
│   ├── loader.py             PDF에서 쪽 단위 텍스트 추출
│   ├── splitter.py           쪽을 청크로 분할
│   ├── textnorm.py           인용 대조용 정규화
│   ├── indexer.py            bge-m3 dense + sparse 색인
│   ├── retriever.py          RRF 앙상블 검색
│   ├── web_search.py         Tavily 웹 검색
│   └── eval.py               Hit Rate@K, MRR 측정
│
├── prompts/                  프롬프트 마크다운 12개 (계획·재계획, 보고서 품질 판정 포함)
│   └── perspective/          관점별 역할 지시 5개
│
├── data/
│   ├── docs/                 논문 PDF 2편
│   ├── criteria/             평가 기준 YAML 5개
│   ├── fixtures/             단독 실행용 샘플 JSON 8개
│   └── registry.yaml         원문 서지 정보
│
└── outputs/                  report.md, report.pdf, eval_set.json
                              (색인, 캐시, 로그·결정 로그, 체크포인트 DB는 git 제외)
```

코드는 `core/`, `agents/`, `rag/` 세 패키지뿐이고 `prompts/`와 `data/`에는 파이썬 파일이 없다. 평가 기준과 프롬프트를 코드 밖으로 빼 두어 내용 수정과 코드 수정을 분리했다.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # OPENAI_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY
python -m core.llm --list       # 접근 가능한 모델 확인 후 config.yaml models 채우기

python -m rag.indexer           # 색인 생성 (한 번만, bge-m3 다운로드 포함)
python app.py --dry-run         # 외부 호출 없이 그래프 전체 통과 확인
python app.py --criteria-limit 1 --retry 1   # 빠른 시험
python app.py                   # 전체 실행 → outputs/report.md, report.pdf
python app.py --no-cache        # 응답 캐시 없이 새로 실행
python app.py --resume RUN_ID   # 중단된 실행을 체크포인트에서 이어서 실행
python -m rag.eval              # 검색 평가 (Hit Rate@5, MRR)
```

LangSmith 추적은 `.env`의 `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`로 켠다. 실행 첫 줄에 `LangSmith 추적 활성화`가 나오면 켜진 것이다. 오케스트레이터의 결정은 아래처럼 확인한다.

```bash
grep -E '"decision": "(plan_eval|replan|quality_replan|finish)"' outputs/logs/decisions.jsonl
```

노드 하나만 돌려 볼 수도 있다. 앞 단계 결과는 `data/fixtures/`에서 채운다.

```bash
python app.py --only synthesis
python -m core.schemas --validate data/fixtures/
```

## Lessons Learned

- **검사 두 개가 서로를 막을 수 있다.** TRL 서술에 "공개 정보 기반 추정" 문구를 넣으라고 했더니, 필수 항목 검사는 통과하지만 근거 타당성 판정이 "인용문에 없는 전제"라며 전부 떨어뜨렸다. 전제를 `claim`에서 `conditions`로 옮겨 해결했다. 검증을 여러 겹 두면 겹끼리 충돌하는지 먼저 확인해야 한다.
- **판정 모델 선택이 실행 시간을 좌우한다.** 판정 호출이 전체의 63%인데 추론 계열 모델은 건당 9.6초, 일반 모델은 0.9초였다. 모델만 바꿔 45분이 9분이 됐다. 다만 근거 통과율이 43%에서 93%로 올라 검증 강도가 달라졌다. 속도와 엄격함은 맞바꾸는 관계다.
- **검색 소스는 기준의 성격을 따라간다.** 처음에는 성숙도를 논문에서, 도메인을 웹에서 찾게 했는데 반대였다. 처리량·지연 수치와 측정 조건은 논문에만 있고, 상용 운용·양산 같은 TRL 상위 단계는 논문에 실리지 않는다.
- **프롬프트의 사소한 군더더기가 비용이 된다.** 파일 첫 줄의 변수 안내 메모가 포맷 대상에 포함되어 검색 결과가 두 번 들어갔다. 렌더된 프롬프트가 1.8배로 부풀었고 66개 작업마다 곱해졌다.
- **인용 대조는 정규화가 전부다.** PDF의 하이픈 줄바꿈과 특수 따옴표를 맞추지 않으면 멀쩡한 인용이 탈락하고 재생성 루프가 계속 돈다.

## Contributors

판교캠퍼스 9반 4조

- **P283 김성현** : State·스키마 설계, 평가 기준 정의, 프롬프트 엔지니어링
- **P292 양지윤** : RAG 파이프라인 구현, 하이브리드 검색, 검색 성능 평가
- **P299 이지은** : 에이전트 구현, 그래프 조립, 근거 검증 로직
