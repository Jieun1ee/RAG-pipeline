"""모델 응답과 검색 결과를 파일로 남겨 같은 요청을 두 번 보내지 않게 한다.

키 문자열의 sha256을 파일 이름으로 삼아 값 하나를 JSON 하나에 담는다. 키에는 프롬프트 전문처럼
요청을 결정짓는 값이 전부 들어가므로, 요청이 조금이라도 달라지면 다른 파일이 된다.
캐시가 채워진 상태에서 다시 실행하면 외부 호출 없이 같은 결과가 나온다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from core import config


def _dir() -> Path:
    return config.resolve(config.get().get("cache", {}).get("dir", "outputs/cache"))


def _enabled() -> bool:
    return bool(config.get().get("cache", {}).get("enabled", True))


def key_of(key: str) -> str:
    """키 문자열을 파일 이름으로 쓸 수 있는 16진수 해시로 바꾼다."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def path_of(key: str) -> Path:
    """그 키가 저장되는 파일 경로."""
    return _dir() / f"{key_of(key)}.json"


def get(key: str) -> Any | None:
    """저장된 값. 키가 없거나 캐시를 꺼 두었으면 None."""
    if not _enabled():
        return None
    path = path_of(key)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def set(key: str, value: Any) -> None:  # noqa: A001 - 내장 set과 겹치지만 get/set 쌍을 유지한다
    """값을 JSON으로 저장한다. 캐시를 꺼 두었으면 아무것도 하지 않는다."""
    if not _enabled():
        return
    path = path_of(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
