"""LangSmith 추적을 환경 변수로 켜고 실행 정보를 붙인다.

LANGSMITH_TRACING=true일 때만 동작한다. 키가 빠졌다면 본 실행을 막지 않도록
추적만 끄고 경고를 남긴다. LangGraph와 LangChain 모델 호출은 같은 실행 아래에
자동으로 기록된다.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

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


def run_config(*, dry_run: bool, mode: str) -> dict[str, Any]:
    """LangGraph 최상위 실행에 표시할 이름, 태그, 메타데이터를 만든다."""
    if not enabled():
        return {}
    return {
        "run_name": "kv-cache-eval",
        "tags": ["rag-pipeline", "dry-run" if dry_run else "live"],
        "metadata": {
            "mode": mode,
            "dry_run": dry_run,
        },
    }
