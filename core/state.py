"""그래프가 단계 사이에 주고받는 상태 정의.

값은 Pydantic 객체가 아니라 model_dump()한 dict로 넣는다. 그래야 직렬화와 샘플 파일 저장이
단순해지고, 읽는 쪽에서 Model.model_validate()로 복원하면 된다.

설계 원칙
  - State에는 작업 결과(페이로드)와 흐름을 정하는 데 필요한 최소한의 값(제어 메타)만 둔다.
    라우터는 제어 메타만 읽는다.
  - 결정 로그와 트레이스 본문은 State에 넣지 않는다. 라우터와 오케스트레이터가 logger로
    {run_id, task_id, node, decision, reason, ts}를 남기고, run_id로 State와 잇는다.
  - 검색 결과처럼 덩치가 큰 중간값은 WorkerState 안에서만 쓰고 버린다. 체크포인트마다 저장되는
    MainState에는 검증을 통과한 결과만 올라온다.
  - worker 여러 개가 동시에 쓰는 키에는 reducer를 붙인다. 붙이지 않은 키에 둘 이상이 동시에 쓰면
    LangGraph가 오류를 낸다. worker가 돌려주는 키는 reducer가 있는 키로 제한한다.

LangGraph는 Annotated의 마지막 인자가 함수일 때 그것을 reducer로 쓴다. 그래서 설명 문자열을
앞에, reducer를 맨 뒤에 둔다.
"""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from core.schemas import TaskStatusName


def merge(old: dict | None, new: dict | None) -> dict:
    """키 단위로 합친다. 같은 키가 다시 오면 새 값으로 덮는다.

    task_id를 키로 쓰기 때문에, 같은 task를 다시 돌려도 결과가 두 번 쌓이지 않고 마지막 것만 남는다.
    """
    return {**(old or {}), **(new or {})}


def merge_or_reset(old: dict | None, new: dict | None) -> dict:
    """merge와 같되, None을 받으면 비운다.

    worker 결과를 관점별로 묶어 옮긴 뒤 원본을 비울 때 쓴다. 비우지 않으면 같은 서술이 두 곳에
    남아 체크포인트마다 함께 저장된다.
    """
    if new is None:
        return {}
    return merge(old, new)


class TaskStatus(TypedDict):
    """task 하나의 진행 상태. 재개할 때 무엇을 다시 보낼지, 다시 보낼 수 있는지를 여기서 본다."""

    status: TaskStatusName
    attempt: int  # 오케스트레이터가 이 task를 보낸 횟수. 다시 보내기 상한과 견준다


class MainState(TypedDict, total=False):
    """전체 그래프가 공유하는 상태. 결과 키 이름은 그 값을 쓰는 노드 이름과 같게 맞춰 두었다."""

    # ── 페이로드: 작업 결과 ─────────────────────────────────────────
    selected_tech: Annotated[list[dict], "평가 대상 기술과 선정 사유"]
    target_domain: Annotated[str, "평가 도메인 이름"]
    criteria: Annotated[dict, "관점 이름으로 묶은 평가 기준"]
    task_results: Annotated[dict[str, dict], "task_id별 worker 결과 {findings, gaps}. 관점별로 묶은 뒤 비운다", merge_or_reset]
    tech_research: Annotated[dict, "기술 조사 결과 (PerspectiveResult)"]
    trl_eval: Annotated[dict, "기술 성숙도 평가 결과 (TRLResult)"]
    market_eval: Annotated[dict, "시장성 평가 결과 (PerspectiveResult)"]
    stakeholder_eval: Annotated[dict, "이해관계자 평가 결과 (PerspectiveResult)"]
    domain_eval: Annotated[dict, "도메인 적합성 평가 결과 (PerspectiveResult)"]
    synthesis: Annotated[dict, "관점 종합 결과 (SynthesisResult)"]
    final_report: Annotated[str, "보고서 본문"]

    # ── 제어 메타: 라우팅·종료·재개에 필요한 최소치 ─────────────────
    run_id: Annotated[str, "실행 하나를 가리키는 키. 로그, LangSmith metadata, 체크포인트 thread_id에 같은 값을 쓴다"]
    step_count: Annotated[int, "메인 노드가 실행된 횟수. MAX_STEPS와 견주는 종료 가드", operator.add]
    plan: Annotated[list[dict], "오케스트레이터가 세운 현재 task 목록. worker 배분·재개에 사용하고 계획 사유는 외부 로그에 남긴다"]
    replan_count: Annotated[int, "최초 계획 이후 재계획한 횟수. 오케스트레이터가 갱신하고 재계획 상한과 견준다"]
    task_status: Annotated[dict[str, TaskStatus], "task_id별 진행 상태와 보낸 횟수", merge]
    errors: Annotated[dict[str, str], "task_id나 노드 이름별 마지막 오류", merge]
    # 검사 결과의 issues는 로그가 아니라 다시 쓸 때 프롬프트에 붙이는 입력이라 State에 둔다
    synthesis_check: Annotated[dict, "종합 결과 검사 (CheckResult)"]
    report_check: Annotated[dict, "보고서 검사 (CheckResult)"]


class WorkerState(TypedDict, total=False):
    """기준 하나와 기술 하나를 처리하는 worker 서브그래프의 상태.

    query, retrieved와 두 검사 결과는 이 안에서만 쓰는 중간값이다. 밖으로는 findings와 gaps만 나간다.
    """

    task: Annotated[dict, "처리할 일감 (TaskSpec)"]
    run_id: Annotated[str, "MainState의 run_id. worker의 로그를 같은 실행에 묶는다"]
    query: Annotated[str, "검색 질의. 줄 단위로 여러 개"]
    retrieved: Annotated[list[dict], "검색 결과 (Retrieved)"]
    retrieval_check: Annotated[dict, "검색 결과 관련성 검사 (CheckResult)"]
    findings: Annotated[list[dict], "근거를 갖춘 서술 (Finding)"]
    citation_check: Annotated[dict, "인용 실재와 근거 타당성 검사 (CheckResult)"]
    gaps: Annotated[list[str], "근거를 확보하지 못해 뺀 항목과 그 이유"]


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def load_fixture(name: str) -> Any:
    """data/fixtures/<name>.json을 읽어 파싱한 값을 돌려준다.

    아직 실행하지 않은 노드의 결과를 대신 채우는 데 쓴다. 덕분에 앞 단계가 끝나지 않아도
    뒤 단계를 따로 돌려 볼 수 있고, 외부 호출 없이 그래프 전체를 통과시킬 수 있다.
    """
    path = FIXTURES_DIR / (name if name.endswith(".json") else f"{name}.json")
    if not path.exists():
        raise FileNotFoundError(f"픽스처가 없다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
