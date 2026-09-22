"""문서 청크를 검색할 수 있는 형태로 만들어 저장한다.

한 번 인코딩하면 의미 벡터와 어휘 가중치가 함께 나온다. 의미 벡터는 표현이 달라도 뜻이 가까운
문장을 찾고, 어휘 가중치는 모델 이름이나 수치처럼 글자가 그대로 맞아야 하는 것을 찾는다.
검색할 때는 두 결과를 합쳐 쓴다.

    python -m rag.indexer     문서를 읽어 색인을 만든다. 문서가 바뀔 때만 다시 돌리면 된다
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch  # noqa: F401 - faiss보다 먼저 불러야 한다. 순서가 바뀌면 macOS에서 인코딩이 예고 없이 멈춘다

from core import config
from rag.splitter import Chunk

log = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-m3"
_model: Any = None


@dataclass
class Index:
    """색인 한 벌. 세 값이 같은 순서로 맞물려 있어 함께 저장하고 함께 읽는다."""

    chunks: list[Chunk] = field(default_factory=list)
    dense: Any = None  # 의미 벡터를 담은 faiss 인덱스
    sparse: list[dict[str, float]] = field(default_factory=list)  # 청크별 토큰 가중치


def index_dir() -> Path:
    """색인 파일을 두는 디렉토리."""
    return config.resolve(config.get()["paths"]["index_dir"])


def device() -> str:
    """쓸 수 있는 가장 빠른 장치 이름.

    장치를 지정하지 않으면 라이브러리가 별도 프로세스를 띄우다 중간에 멈추는 일이 있어 항상 명시한다.
    """
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def model() -> Any:
    """임베딩 모델. 처음 부를 때 내려받아 올려 두고 이후 재사용한다."""
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel

        log.info("loading %s on %s", MODEL_NAME, device())
        _model = BGEM3FlagModel(MODEL_NAME, use_fp16=False, devices=[device()])
    return _model


def encode(texts: list[str], batch_size: int = 8, max_length: int = 1024) -> tuple[np.ndarray, list[dict[str, float]]]:
    """문장 목록을 의미 벡터 행렬과 토큰 가중치 목록으로 바꾼다.

    벡터는 길이를 1로 맞춰 둔다. 그래야 내적만으로 코사인 유사도를 얻을 수 있다.
    """
    out = model().encode(
        texts, batch_size=batch_size, max_length=max_length, return_dense=True, return_sparse=True, return_colbert_vecs=False
    )
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)
    dense /= np.clip(np.linalg.norm(dense, axis=1, keepdims=True), 1e-12, None)
    sparse = [{str(k): float(v) for k, v in weights.items()} for weights in out["lexical_weights"]]
    return dense, sparse


def build(chunks: list[Chunk]) -> None:
    """청크를 인코딩해 색인 파일 세 개로 저장한다."""
    import faiss

    if not chunks:
        raise ValueError("청크가 없다. data/docs/에 PDF가 있는지 확인하라")
    log.info("encoding %d chunks", len(chunks))
    dense, sparse = encode([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(dense.shape[1])
    index.add(dense)

    out = index_dir()
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "dense.faiss"))
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (out / "sparse.json").write_text(json.dumps(sparse), encoding="utf-8")
    log.info("index saved to %s (dim=%d)", out, dense.shape[1])


def load() -> Index:
    """저장해 둔 색인을 읽는다. 없으면 만드는 방법을 알려 주고 멈춘다."""
    import faiss

    out = index_dir()
    if not (out / "dense.faiss").exists():
        raise FileNotFoundError(f"색인이 없다: {out}. 먼저 `python -m rag.indexer`를 실행하라")
    return Index(
        chunks=json.loads((out / "chunks.json").read_text(encoding="utf-8")),
        dense=faiss.read_index(str(out / "dense.faiss")),
        sparse=json.loads((out / "sparse.json").read_text(encoding="utf-8")),
    )


def main() -> int:
    import yaml

    from rag.loader import load as load_docs
    from rag.splitter import split

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with open(config.resolve(config.get()["paths"]["registry"]), encoding="utf-8") as f:
        registry = yaml.safe_load(f)
    chunks: list[Chunk] = []
    for doc in load_docs(registry):
        chunks.extend(split(doc))
    build(chunks)
    print(f"색인 생성 완료: 청크 {len(chunks)}개 → {index_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
