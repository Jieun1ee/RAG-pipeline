"""평가 기준 YAML을 읽어 들인다.

data/criteria 아래 다섯 파일이 관점별 평가 기준을 담고 있다. 여기서 형식을 미리 확인해,
파일이 잘못되었을 때 한참 뒤 노드에서 엉뚱한 오류가 나는 대신 바로 원인이 드러나게 한다.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CRITERIA_DIR = Path(__file__).resolve().parents[1] / "data" / "criteria"

FILES: dict[str, str] = {
    "tech": "tech.yaml",
    "trl": "trl.yaml",
    "market": "market.yaml",
    "stakeholder": "stakeholder.yaml",
    "domain": "domain.yaml",
}


def load_one(perspective: str) -> dict:
    """관점 하나의 기준 파일을 읽는다.

    파일 안의 perspective 값이 인자와 다르거나 items 목록이 없으면 ValueError를 낸다.
    """
    path = CRITERIA_DIR / FILES[perspective]
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if data.get("perspective") != perspective:
        raise ValueError(f"{path.name}: perspective가 '{perspective}'가 아니다: {data.get('perspective')!r}")
    if not isinstance(data.get("items"), list):
        raise ValueError(f"{path.name}: items 목록이 없다")
    return data


def load_all() -> dict[str, dict]:
    """다섯 관점의 기준을 {관점 이름: 기준} 형태로 모은다."""
    return {perspective: load_one(perspective) for perspective in FILES}
