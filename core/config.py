"""config.yaml을 읽고 실행 중 설정값을 관리한다.

설정 파일은 프로세스당 한 번만 읽어 dict 하나로 공유한다. get()이 돌려주는 것은 복사본이
아니라 그 dict 자체라서, override()로 바꾼 값이 이후 모든 호출에 그대로 보인다.
노드가 전역 변수를 따로 두지 않고 설정을 항상 여기서 읽게 하려는 구조다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

_config: dict[str, Any] | None = None


def load(path: Path | None = None) -> dict[str, Any]:
    """config.yaml을 읽어 모듈 안에 담아 둔다. 두 번째 호출부터는 파일을 다시 읽지 않는다."""
    global _config
    if _config is None:
        with open(path or CONFIG_PATH, encoding="utf-8") as f:
            _config = yaml.safe_load(f) or {}
        _config.setdefault("dry_run", False)  # 파일에는 없는 값이고 명령줄로만 켠다
    return _config


def get() -> dict[str, Any]:
    """현재 설정 dict. override()로 바꾼 값이 반영된 원본 객체를 그대로 돌려준다."""
    return load()


def override(key: str, value: Any) -> None:
    """점으로 구분한 경로의 값을 바꾼다. 중간 키가 없으면 빈 dict를 만들며 내려간다.

    예: override("execution.criteria_limit", 3)
    """
    node = get()
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def apply_cli(
    *,
    dry_run: bool | None = None,
    retry: int | None = None,
    criteria_limit: int | None = None,
    no_cache: bool | None = None,
) -> None:
    """명령줄 인자를 설정에 반영한다. 넘어오지 않은 인자는 파일 값을 그대로 둔다.

    retry는 항목별로 구분하지 않고 retry 아래 모든 상한을 같은 값으로 맞춘다.
    """
    if dry_run:
        override("dry_run", True)
    if retry is not None:
        for name in list(get().get("retry", {})):
            override(f"retry.{name}", retry)
    if criteria_limit is not None:
        override("execution.criteria_limit", criteria_limit)
    if no_cache:
        override("cache.enabled", False)


def is_dry_run() -> bool:
    """외부 호출 없이 샘플 데이터로 도는 모드인지."""
    return bool(get().get("dry_run", False))


def retry_limit(name: str) -> int:
    """이름별 재시도 상한. 설정에 없는 이름은 0으로 보아 재시도하지 않는다."""
    return int(get().get("retry", {}).get(name, 0))


def resolve(path: str) -> Path:
    """설정에 적힌 상대 경로를 저장소 루트 기준 절대 경로로 바꾼다."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def reset() -> None:
    """다음 get()에서 파일을 다시 읽게 한다. 설정을 바꿔 가며 시험할 때 쓴다."""
    global _config
    _config = None
