"""PDF에서 쪽 단위 텍스트를 뽑는다.

논문 PDF는 모든 쪽에 같은 머리글과 바닥글이 박혀 있다. 그대로 쪼개면 청크마다 같은 문장이 섞여
검색 결과가 그 문장 쪽으로 쏠리므로, 텍스트를 넘기기 전에 반복되는 줄을 걷어 낸다.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import TypedDict

import pymupdf

from core import config

log = logging.getLogger(__name__)


class Doc(TypedDict):
    """문서 하나에서 뽑아낸 본문과 서지 정보."""

    doc_id: str
    meta: dict  # 등록 파일에 적힌 technology, title, author_or_org, year, venue, url
    pages: list[tuple[int, str]]  # (쪽 번호, 본문). 문단은 빈 줄로 구분된다


_DIGITS = re.compile(r"\d+")
_EDGE_LINES = 3  # 머리글·바닥글 후보로 볼 쪽 위아래 줄 수
_REPEAT_RATIO = 0.4  # 이 비율 이상의 쪽에 나타나면 반복되는 줄로 본다


def _page_text(page: pymupdf.Page) -> str:
    """쪽 안의 텍스트 블록을 빈 줄로 이어 붙인다. 뒤에서 이 빈 줄을 문단 경계로 쓴다."""
    blocks = [b[4].strip() for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    return "\n\n".join(blocks)


def _signature(line: str) -> str:
    """줄을 비교하기 좋은 모양으로 바꾼다. 쪽 번호처럼 쪽마다 달라지는 숫자는 한 글자로 뭉갠다."""
    return _DIGITS.sub("#", line.strip().lower())


def strip_repeated_lines(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """여러 쪽에 되풀이되는 머리글·바닥글과 쪽 번호만 있는 줄을 지운다.

    쪽 위아래 몇 줄만 후보로 보므로 본문 중간에 우연히 같은 문장이 있어도 지워지지 않는다.
    쪽 수가 너무 적으면 반복인지 판단할 근거가 없어 손대지 않는다.
    """
    if len(pages) < 4:
        return pages
    counts: Counter[str] = Counter()
    for _, text in pages:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        edge = set(_signature(ln) for ln in lines[:_EDGE_LINES] + lines[-_EDGE_LINES:])
        counts.update(edge)
    repeated = {sig for sig, n in counts.items() if n >= max(2, len(pages) * _REPEAT_RATIO)}

    cleaned = []
    for page_no, text in pages:
        kept = []
        for ln in text.splitlines():
            stripped = ln.strip()
            if stripped and (_signature(stripped) in repeated or stripped.isdigit()):
                continue
            kept.append(ln)
        # 줄을 빼면 빈 줄이 여러 개 남아 문단 경계가 흐트러지므로 다시 하나로 줄인다
        cleaned.append((page_no, re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()))
    return cleaned


def load(registry: dict) -> list[Doc]:
    """등록 정보에 적힌 PDF를 모두 읽는다.

    파일이 없으면 경고만 남기고 넘어간다. 문서 하나가 없다고 색인 작업 전체가 멈추지 않게 하려는 것이다.
    """
    docs: list[Doc] = []
    for entry in registry.get("docs", []):
        path = config.resolve(entry["file"])
        if not path.exists():
            log.warning("PDF가 없다, 건너뜀: %s", path)
            continue
        with pymupdf.open(path) as pdf:
            pages = [(i + 1, _page_text(page)) for i, page in enumerate(pdf)]
        pages = strip_repeated_lines(pages)
        meta = {k: v for k, v in entry.items() if k != "file"}
        docs.append({"doc_id": entry["doc_id"], "meta": meta, "pages": pages})
        log.info("loaded %s: %d pages", entry["doc_id"], len(pages))
    return docs
