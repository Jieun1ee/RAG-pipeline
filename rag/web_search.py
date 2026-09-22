"""웹 검색 결과를 논문 검색과 같은 형식으로 맞춰 돌려준다.

부르는 쪽이 논문인지 웹인지 신경 쓰지 않아도 되도록 형식을 통일한다.
검색이 실패해도 예외를 올리지 않고 빈 목록을 돌려준다. 근거를 못 찾은 것과 프로그램이 멈추는 것은
다르게 다뤄야 하고, 근거가 없는 상황은 뒤에서 재시도와 미확인 처리로 이어진다.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from core import cache, config
from core.schemas import Retrieved

log = logging.getLogger(__name__)


class WebSearch:
    """웹 검색 도구를 감싼 것. 결과 개수를 넘기지 않으면 설정값을 쓴다."""

    def __init__(self, max_results: int | None = None) -> None:
        self._max_results = max_results

    def search(self, query: str, k: int) -> list[Retrieved]:
        """질의에 대한 검색 결과. 같은 질의를 다시 부르면 저장해 둔 결과를 쓴다.

        본문이 없거나 주소가 겹치는 항목은 버린다. 같은 페이지가 여러 번 인용되면
        출처 목록이 부풀기 때문이다. 발행일이 없는 결과도 그대로 두고, 날짜를 채우는 일은
        인용을 만드는 쪽에서 처리한다.
        """
        k = int(self._max_results or k or config.get()["retrieval"]["web_max_results"])
        cache_key = f"tavily|{k}|{query}"
        cached = cache.get(cache_key)
        if cached is not None:
            return [Retrieved.model_validate(r) for r in cached]

        api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if not api_key:
            log.warning("TAVILY_API_KEY가 없어 웹 검색을 건너뛴다: %s", query[:60])
            return []
        try:
            from tavily import TavilyClient

            response = TavilyClient(api_key=api_key).search(query=query, max_results=k, search_depth="basic")
        except Exception as exc:  # noqa: BLE001 - 네트워크와 응답 오류 모두 빈 결과로 처리한다
            log.warning("Tavily 검색 실패 (%s): %s", query[:60], exc)
            return []

        results: list[Retrieved] = []
        seen: set[str] = set()
        for item in response.get("results", []):
            url = (item.get("url") or "").strip()
            text = (item.get("content") or "").strip()
            if not url or not text or url in seen:
                continue
            seen.add(url)
            results.append(
                Retrieved(
                    id=url,
                    text=text,
                    source_type="web",
                    title=item.get("title"),
                    url=url,
                    date=item.get("published_date") or None,
                    author_or_org=urlparse(url).netloc.removeprefix("www."),  # 출처를 도메인으로 적어 둔다
                )
            )
        cache.set(cache_key, [r.model_dump(mode="json") for r in results])
        return results
