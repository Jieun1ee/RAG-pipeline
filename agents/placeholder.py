"""worker와 collect의 임시 구현. A 담당(feature/worker)의 구현이 들어오면 graph.py의 import만 바꾸고 이 파일은 지운다.

그래프 연결과 회차 흐름을 먼저 확인하려고 둔 것이다. 모델과 검색을 부르지 않고 샘플 값만 쓴다.
계약(core/schemas.py)은 실제 구현과 똑같이 지킨다. worker는 worker_update()의 결과만 돌려주고,
collect는 task_results 전체로 관점별 결과 키를 매번 다시 만든다.
"""

from __future__ import annotations

from core.schemas import TaskSpec, worker_update
from core.state import MainState, WorkerState, load_fixture

STATE_KEY = {
    "tech": "tech_research",
    "trl": "trl_eval",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}


def worker(state: WorkerState) -> MainState:
    """샘플 결과에서 이 task의 기준과 기술에 해당하는 서술을 골라 돌려준다.

    재계획 회차는 서술 id가 첫 회차 샘플과 겹치므로 빈 결과로 둔다.
    """
    spec = TaskSpec.model_validate(state["task"])
    findings = []
    if spec.round == 0:
        findings = [
            f
            for f in load_fixture(STATE_KEY[spec.perspective]).get("findings", [])
            if f["technology"] == spec.technology and f["criterion"] == spec.criterion["id"]
        ]
    gaps = [] if findings else [f"{spec.task_id}: (임시 worker) 샘플에 해당 서술 없음"]
    return worker_update(spec, findings=findings, gaps=gaps)


def collect(state: MainState) -> MainState:
    """task_results에 나온 관점마다 결과 키를 채운다. 임시 구현이라 샘플 결과를 그대로 쓴다."""
    perspectives = {result["perspective"] for result in state.get("task_results", {}).values()}
    return {STATE_KEY[p]: load_fixture(STATE_KEY[p]) for p in perspectives}
