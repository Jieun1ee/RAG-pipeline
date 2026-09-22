"""검색이 얼마나 맞히는지 잰다.

정답이 붙은 질문 모음이 없으므로 청크에서 거꾸로 만든다. 청크 하나를 골라 그 안의 내용으로만
답할 수 있는 질문을 쓰게 하면, 그 청크가 곧 정답이 된다.
질문을 한 번 만들어 파일로 남겨 두어야 구성을 바꿔 가며 잰 수치를 서로 견줄 수 있다.

    python -m rag.eval

두 가지를 출력한다. 정답이 상위 K개 안에 든 질문의 비율과, 정답이 몇 등이었는지를 역수로 평균한 값이다.
앞엣것은 찾았는지를, 뒤엣것은 얼마나 위에 올렸는지를 본다.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from core import config, llm
from core.schemas import QuestionDraft
from rag.indexer import Index, load
from rag.retriever import Retriever

log = logging.getLogger(__name__)

EVAL_SET_PATH = Path("outputs/eval_set.json")
MODES = ("dense", "sparse", "rrf")

_QUESTION_PROMPT = """아래는 논문의 한 부분이다. 이 부분을 읽어야만 답할 수 있는 구체적인 질문을 영어로 하나 만든다.
질문은 이 부분에 있는 고유한 사실(수치, 구성 요소 이름, 방법)을 가리켜야 하고, 논문 전체에 대한 일반 질문이면 안 된다.

---
{text}
---"""


def build_eval_set(index: Index, n: int = 30, seed: int = 0) -> list[dict]:
    """청크를 골라 질문을 만들고 (질문, 정답 청크) 쌍을 파일로 남긴다.

    너무 짧은 청크는 제외한다. 표 조각 같은 데서는 답할 거리가 있는 질문이 나오지 않는다.
    고르는 순서를 고정해 두어 다시 돌려도 같은 청크가 뽑힌다.
    """
    candidates = [c for c in index.chunks if len(c["text"]) >= 400]
    rng = random.Random(seed)
    sample = rng.sample(candidates, min(n, len(candidates)))
    eval_set: list[dict] = []
    for chunk in sample:
        draft = llm.generate(_QUESTION_PROMPT.format(text=chunk["text"][:2500]), QuestionDraft)
        if draft is None or not draft.question.strip():
            continue
        eval_set.append({"question": draft.question.strip(), "chunk_id": chunk["chunk_id"]})
    path = config.resolve(str(EVAL_SET_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(eval_set, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("eval set saved: %d questions → %s", len(eval_set), path)
    return eval_set


def evaluate(index: Index, eval_set: list[dict], k: int = 5) -> dict[str, dict[str, float]]:
    """세 가지 검색 구성으로 같은 질문을 돌려 구성별 수치를 낸다."""
    retriever = Retriever(index)
    id_of = [c["chunk_id"] for c in index.chunks]
    results: dict[str, dict[str, float]] = {}
    for mode in MODES:
        hits = 0
        rr_sum = 0.0
        for item in eval_set:
            ranked = [id_of[i] for i in retriever.rank(item["question"], k, mode)]
            if item["chunk_id"] in ranked:
                hits += 1
                rr_sum += 1.0 / (ranked.index(item["chunk_id"]) + 1)
        n = max(1, len(eval_set))
        results[mode] = {"hit_rate": hits / n, "mrr": rr_sum / n}
    return results


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    index = load()
    path = config.resolve(str(EVAL_SET_PATH))
    if path.exists():  # 질문을 다시 만들면 이전 수치와 견줄 수 없다
        eval_set = json.loads(path.read_text(encoding="utf-8"))
        log.info("using existing eval set: %d questions", len(eval_set))
    else:
        eval_set = build_eval_set(index)
    if not eval_set:
        print("평가 집합이 비어 있다. OPENAI_API_KEY와 config models.generator를 확인하라")
        return 1
    k = 5
    results = evaluate(index, eval_set, k)
    print(f"\n검색 평가 (n={len(eval_set)}, K={k})")
    print(f"{'구성':8s} {'Hit Rate@' + str(k):>12s} {'MRR':>8s}")
    for mode in MODES:
        print(f"{mode:8s} {results[mode]['hit_rate']:>12.3f} {results[mode]['mrr']:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
