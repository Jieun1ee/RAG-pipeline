"""LangSmith 추적을 환경 변수로 켜고 실행 정보를 붙인다.

LANGSMITH_TRACING=true일 때만 동작한다. 키가 빠졌다면 본 실행을 막지 않도록
추적만 끄고 경고를 남긴다. LangGraph와 LangChain 모델 호출은 같은 실행 아래에
자동으로 기록된다.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)
decision_log = logging.getLogger("decision")

DEFAULT_PROJECT = "rag-pipeline"
TRUE_VALUES = {"1", "true", "yes", "on"}


def _env(primary: str, legacy: str) -> str:
    """최신 LangSmith 변수와 기존 LangChain 변수 중 설정된 값을 읽는다."""
    return (os.getenv(primary) or os.getenv(legacy) or "").strip()


def enabled() -> bool:
    """현재 환경에서 LangSmith 추적이 켜져 있는지 확인한다."""
    return _env("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2").lower() in TRUE_VALUES


def configure() -> bool:
    """LangSmith 설정을 검증하고 사용할 프로젝트를 확정한다."""
    if not enabled():
        log.info("LangSmith 추적 비활성화")
        return False

    api_key = _env("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
    if not api_key:
        log.warning("LANGSMITH_TRACING=true이지만 LANGSMITH_API_KEY가 없어 추적을 끈다")
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    project = _env("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT") or DEFAULT_PROJECT
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project
    log.info("LangSmith 추적 활성화: project=%s", project)
    return True


def run_config(*, dry_run: bool, mode: str, run_id: str) -> dict[str, Any]:
    """LangGraph 최상위 실행에 표시할 이름, 태그, 메타데이터를 만든다.

    run_id를 metadata에 넣어 LangSmith에서 State, 로그, 체크포인트와 같은 값으로 찾을 수 있게 한다.
    """
    if not enabled():
        return {}
    return {
        "run_name": f"kv-cache-eval {run_id}",
        "tags": ["rag-pipeline", "dry-run" if dry_run else "live"],
        "metadata": {
            "run_id": run_id,
            "mode": mode,
            "dry_run": dry_run,
        },
    }


def log_decision(run_id: str, node: str, decision: str, reason: str, **detail: Any) -> None:
    """흐름을 정한 결정 하나를 사유와 함께 남긴다.

    결정 로그는 State에 넣지 않는다. 체크포인트마다 함께 저장되어 쌓이기만 하고, 다음 결정에 쓰이지도
    않기 때문이다. 대신 run_id를 붙여 한 줄짜리 JSON으로 남기고, 같은 run_id로 State와 LangSmith
    기록을 찾아 잇는다. detail에는 보낸 task 수처럼 결정을 되짚는 데 필요한 값만 넣는다.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "node": node,
        "decision": decision,
        "reason": reason,
        **detail,
    }
    decision_log.info(json.dumps(record, ensure_ascii=False))
