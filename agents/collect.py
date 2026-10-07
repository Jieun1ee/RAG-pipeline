"""현재 배치의 worker 결과를 관점별로 집계한다."""

from collections import defaultdict

from agents.perspective import STATE_KEY, aggregate, balance_gaps
from core import config
from core.state import MainState


def node(state: MainState) -> MainState:
    """기존 근거와 추가 조사 결과를 합친 뒤 임시 결과를 비운다.

    재시도 가능한 오류는 error로 유지하고, 상한에 닿은 오류는 gap으로 남긴다.
    결과 저장과 초기화는 하나의 업데이트로 반환한다.
    """
    results = state.get("task_results", {})
    if not results:
        return {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    statuses = {}
    for task_id, result in results.items():
        status = state["task_status"][task_id]
        if status["status"] == "error":
            if status["attempt"] <= config.retry_limit("dispatch"):
                continue
            result = {**result, "findings": [], "gaps": [
                f"{task_id}: 실행 재시도 상한 도달 — {state.get('errors', {}).get(task_id, '실행 실패')}"
            ]}
        grouped[result["perspective"]].append(result)
        statuses[task_id] = {
            "status": "done" if result["findings"] else "gave_up",
            "attempt": status["attempt"],
        }

    update: MainState = {"task_results": None, "task_status": statuses}
    for perspective, incoming in grouped.items():
        key = STATE_KEY[perspective]
        previous = state.get(key, {})
        # 근거가 없었던 항목도 복원해 추가 조사 후 다시 판정한다.
        groups = {
            (a["criterion"], a["technology"]): {
                "criterion_id": a["criterion"], "technology": a["technology"],
                "findings": [], "gaps": [],
            }
            for a in previous.get("assessments", [])
        }
        for finding in previous.get("findings", []):
            groups[(finding["criterion"], finding["technology"])]["findings"].append(finding)
        combined = aggregate(perspective, [*groups.values(), *incoming])
        old_balance = balance_gaps(
            previous.get("findings", []), sorted({a["technology"] for a in previous.get("assessments", [])})
        )
        combined["gaps"] = list(dict.fromkeys(
            [gap for gap in previous.get("gaps", []) if gap not in old_balance] + combined["gaps"]
        ))
        update[key] = combined
    return update
