"""쪽 텍스트를 검색 단위로 쪼갠다.

자르는 위치는 문단 경계로 한정하고, 절 제목을 만나면 거기서 끊어 새 청크를 시작한다.
문장 중간에서 잘리면 인용문이 두 청크에 걸쳐, 나중에 인용이 원문에 있는지 대조할 때 실패한다.
"""

from __future__ import annotations

import re
from typing import TypedDict

from rag.loader import Doc


class Chunk(TypedDict):
    """검색과 인용 대조의 최소 단위. chunk_id는 문서, 쪽, 쪽 안 순번을 이어 붙인 것이다."""

    chunk_id: str
    doc_id: str
    page: int
    section: str | None
    text: str


MAX_CHARS = 1500  # 이 길이를 넘기 전에 문단 경계에서 끊는다
MIN_CHARS = 120  # 이보다 짧으면 표나 그림 캡션 조각으로 보고 버린다
_NUMBER = r"(?:\d+(?:\.\d+)*\.?|[A-Z]\.|(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.)"
_TITLE = r"[A-Z][A-Za-z][^\n]{3,80}"
_HEADING_ONE_LINE = re.compile(rf"^{_NUMBER}\s+{_TITLE}$")
_HEADING_NUMBER_ONLY = re.compile(rf"^{_NUMBER}$")
_HEADING_TITLE_ONLY = re.compile(rf"^{_TITLE}$")


def heading_of(paragraph: str) -> str | None:
    """문단이 절 제목이면 제목 문자열을, 아니면 None을 돌려준다.

    번호와 제목이 한 줄에 있는 형태와 줄이 나뉜 형태를 모두 인식한다. PDF에서 텍스트를 뽑으면
    같은 제목이 문서마다 다른 모양으로 나오기 때문이다.
    """
    if len(paragraph) >= 160:
        return None
    lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
    if not lines:
        return None
    if _HEADING_ONE_LINE.match(lines[0]):
        return lines[0]
    if len(lines) >= 2 and _HEADING_NUMBER_ONLY.match(lines[0]) and _HEADING_TITLE_ONLY.match(lines[1]):
        return f"{lines[0]} {lines[1]}"
    return None


def split(doc: Doc) -> list[Chunk]:
    """문서 하나를 청크 목록으로 바꾼다.

    절 제목은 그 뒤 청크들에 계속 붙는다. 검색 결과만 보고도 문서의 어느 부분인지 알 수 있게 하려는 것이다.
    """
    chunks: list[Chunk] = []
    section: str | None = None
    for page_no, text in doc["pages"]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buffer: list[str] = []
        page_chunks: list[Chunk] = []

        def flush() -> None:
            """모아 둔 문단을 청크 하나로 만든다."""
            if not buffer:
                return
            body = "\n\n".join(buffer)
            page_chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}:p{page_no}:{len(page_chunks)}",
                    "doc_id": doc["doc_id"],
                    "page": page_no,
                    "section": section,
                    "text": body,
                }
            )
            buffer.clear()

        for paragraph in paragraphs:
            heading = heading_of(paragraph)
            if heading is not None:
                flush()
                section = heading
                buffer.append(paragraph)
                continue
            if buffer and sum(len(p) for p in buffer) + len(paragraph) > MAX_CHARS:
                flush()
            buffer.append(paragraph)
        flush()

        # 짧은 조각은 버리되, 쪽 전체가 짧으면 그 쪽이 통째로 사라지지 않게 하나는 남긴다
        kept = [c for c in page_chunks if len(c["text"]) >= MIN_CHARS] or page_chunks[:1]
        for seq, chunk in enumerate(kept):
            chunk["chunk_id"] = f"{doc['doc_id']}:p{page_no}:{seq}"
        chunks.extend(kept)
    return chunks
