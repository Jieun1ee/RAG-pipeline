"""질의를 받아 논문 청크를 찾는다.

의미 벡터 검색과 어휘 검색은 놓치는 것이 서로 다르다. 앞엣것은 표현이 다른 문장을 잘 찾지만
모델 이름이나 수치를 흘리고, 뒤엣것은 그 반대다. 두 순위를 순위 기반으로 합쳐 쓴다.
점수를 직접 더하지 않는 이유는 두 점수의 단위가 달라 그대로 섞으면 한쪽이 결과를 지배하기 때문이다.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
import yaml

from core import config
from core.schemas import Retrieved
from rag import indexer
from rag.indexer import Index

Mode = Literal["dense", "sparse", "rrf"]


@lru_cache(maxsize=1)
def registry_docs() -> dict[str, dict]:
    """문서 id로 찾아 쓰는 서지 정보.

    제목과 저자는 반드시 이 파일에서만 가져온다. 모델이 그럴듯한 서지 정보를 지어내는 것을 막으려는 것이다.
    """
    with open(config.resolve(config.get()["paths"]["registry"]), encoding="utf-8") as f:
        registry = yaml.safe_load(f)
    return {d["doc_id"]: d for d in registry.get("docs", [])}


class Retriever:
    """색인 위에서 도는 검색기. 색인을 넘기지 않으면 처음 쓸 때 파일에서 읽어 온다."""

    def __init__(self, index: Index | None = None) -> None:
        self._index = index

    @property
    def index(self) -> Index:
        if self._index is None:
            self._index = indexer.load()
        return self._index

    def search_dense(self, query_dense: np.ndarray, k: int) -> list[int]:
        """의미 벡터가 가까운 청크의 위치를 가까운 순으로."""
        scores, ids = self.index.dense.search(query_dense.reshape(1, -1), min(k, len(self.index.chunks)))
        return [int(i) for i in ids[0] if i >= 0]

    def search_sparse(self, query_sparse: dict[str, float], k: int) -> list[int]:
        """질의와 겹치는 토큰의 가중치를 곱해 더한 점수로 고른다. 겹치는 토큰이 없으면 제외한다."""
        scored = []
        for idx, weights in enumerate(self.index.sparse):
            score = sum(qw * weights[t] for t, qw in query_sparse.items() if t in weights)
            if score > 0:
                scored.append((score, idx))
        scored.sort(key=lambda s: -s[0])
        return [idx for _, idx in scored[:k]]

    def rank(self, query: str, k: int, mode: Mode = "rrf") -> list[int]:
        """질의에 대한 청크 순위. 한쪽만 쓰거나 둘을 합쳐 쓸 수 있다.

        합칠 때는 각 목록에서 몇 등이었는지만 보고 1/(상수+등수)를 더한다. 상수가 클수록 1등과
        2등의 차이가 줄어, 한 목록에서만 아주 높은 것보다 두 목록에 함께 오른 것이 위로 올라온다.
        합치기 전에는 최종 개수의 두 배씩 뽑아, 한쪽에서만 잡힌 것도 후보에 남게 한다.
        """
        dense_vec, sparse_list = indexer.encode([query], max_length=256)
        if mode == "dense":
            return self.search_dense(dense_vec[0], k)
        if mode == "sparse":
            return self.search_sparse(sparse_list[0], k)
        rrf_k = int(config.get()["retrieval"].get("rrf_k", 60))
        fused: dict[int, float] = {}
        for ranking in (self.search_dense(dense_vec[0], 2 * k), self.search_sparse(sparse_list[0], 2 * k)):
            for rank, idx in enumerate(ranking, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        return [idx for idx, _ in sorted(fused.items(), key=lambda s: -s[1])[:k]]

    def to_retrieved(self, idx: int) -> Retrieved:
        """청크에 서지 정보를 붙여 웹 검색 결과와 같은 형식으로 맞춘다."""
        chunk = self.index.chunks[idx]
        meta = registry_docs().get(chunk["doc_id"], {})
        return Retrieved(
            id=chunk["chunk_id"],
            text=chunk["text"],
            source_type="paper",
            title=meta.get("title"),
            url=meta.get("url"),
            date=str(meta["year"]) if meta.get("year") else None,
            author_or_org=meta.get("author_or_org"),
            page=chunk["page"],
            doc_id=chunk["doc_id"],
        )

    def search(self, query: str, k: int) -> list[Retrieved]:
        """질의에 가장 잘 맞는 청크 k개."""
        return [self.to_retrieved(idx) for idx in self.rank(query, k, "rrf")]
