"""인용문과 원문을 견주기 전에 글자 모양을 맞춘다.

PDF에서 뽑은 텍스트에는 줄 끝에서 잘린 단어, 두 칸 공백, 곧은 따옴표 대신 들어간 굽은 따옴표가
섞여 있다. 사람 눈에는 같은 문장이어도 글자로는 다르다. 양쪽에 같은 변환을 걸어 두지 않으면
멀쩡한 인용이 원문에 없는 것으로 판정되고, 그 때문에 같은 작업을 계속 다시 시도하게 된다.
"""

from __future__ import annotations

import re
import unicodedata

_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",  # ‘ ’ ‚ ‛
    "“": '"', "”": '"', "„": '"', "‟": '"',  # “ ” „ ‟
    "′": "'", "″": '"',  # ′ ″
}
_HYPHENS = "-‐‑­"
_HYPHEN_BREAK = re.compile(rf"(\w)[{_HYPHENS}]\s*\n\s*(\w)")
_INTRAWORD_HYPHEN = re.compile(rf"(\w)[{_HYPHENS}]\s?(\w)")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """견주기 좋게 다듬은 문자열을 돌려준다.

    합자와 전각 문자를 풀고, 굽은 따옴표를 곧은 따옴표로 바꾸고, 줄 끝에서 하이픈으로 끊긴 단어를
    다시 붙인다. 단어 안의 하이픈은 아예 지운다. 끊긴 단어를 옮겨 쓸 때 하이픈을 남기는지 지우는지가
    제각각이라, 양쪽에서 똑같이 없애 버리는 편이 판정이 흔들리지 않는다.
    마지막으로 연속된 공백과 줄바꿈을 한 칸으로 모은다.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans(_QUOTES))
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _INTRAWORD_HYPHEN.sub(r"\1\2", text)
    return _WS.sub(" ", text).strip()
