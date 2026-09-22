"""프롬프트 마크다운을 읽고 변수를 채운다.

prompts 아래 파일은 {이름} 자리에 값이 들어가는 마크다운이다. 중괄호를 글자 그대로 쓰려면 {{ }}로 적는다.
변수 이름은 프롬프트를 쓰는 쪽과 값을 채우는 코드가 맞춰야 하므로, render()는 값이 빠진 변수를
빈칸으로 넘기지 않고 예외로 알린다.

파일 첫 문단에는 그 프롬프트가 받는 변수를 적은 사람용 메모가 있다. 메모에도 {이름} 표기가 들어 있어
파일을 통째로 포맷하면 메모 자리에도 값이 한 번 더 채워진다. 그래서 모델에게 보낼 때는 body()로
메모를 떼어 낸 본문만 쓴다.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def path_of(name: str) -> Path:
    """이름을 파일 경로로 바꾼다. 확장자는 생략할 수 있고 "perspective/market"처럼 하위 폴더도 쓸 수 있다."""
    return PROMPTS_DIR / (name if name.endswith(".md") else f"{name}.md")


def load(name: str) -> str:
    """파일 내용을 그대로 읽는다. 없으면 FileNotFoundError."""
    path = path_of(name)
    if not path.exists():
        raise FileNotFoundError(f"프롬프트가 없다: {path}")
    return path.read_text(encoding="utf-8")


# 첫 문단이 "변수:" 또는 "변수 없음"으로 그 프롬프트의 입력을 적어 둔 메모다
_META = re.compile(r"\A.*?변수(?::|\s*없음).*?(?:\n\s*\n|\Z)", re.DOTALL)


def body(name: str) -> str:
    """머리글 메모를 떼어 낸 본문. 메모가 없으면 파일 그대로.

    모델에게 보내는 것은 언제나 이 본문이다. load()는 메모까지 포함한 원본을 그대로 돌려준다.
    """
    return _META.sub("", load(name), count=1).lstrip()


def variables(name: str) -> set[str]:
    """템플릿이 요구하는 변수 이름 집합. {a.b}나 {a[0]} 같은 표기는 앞부분만 센다."""
    return {
        field.split(".")[0].split("[")[0]
        for _, field, _, _ in string.Formatter().parse(body(name))
        if field
    }


def render(name: str, **vars: object) -> str:
    """변수를 채운 프롬프트 문자열. 빠진 변수가 있으면 KeyError로 이름을 알려 준다."""
    missing = sorted(variables(name) - set(vars))
    if missing:
        raise KeyError(f"프롬프트 {name}: 누락 변수 {missing}")
    return body(name).format(**vars)
