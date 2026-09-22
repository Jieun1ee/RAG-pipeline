"""명령줄 진입점.

그래프 조립은 graph.py에 있다. 여기서는 설정을 맞추고, 전체를 돌리거나 노드 하나만 돌린다.

python app.py                         전부 실행하고 보고서를 남긴다
python app.py --dry-run               모델과 검색을 부르지 않고 샘플 값으로 끝까지 통과시킨다
python app.py --only market_eval      노드 하나만 돌린다. 입력은 샘플에서 채운다
python app.py --criteria-limit 1      관점마다 기준 하나씩만 본다
python app.py --retry 0               검사에 걸려도 다시 쓰지 않는다
python app.py --no-cache              저장해 둔 응답을 쓰지 않는다
python app.py --no-pdf                PDF 변환 없이 마크다운만 저장한다
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from agents.subgraph import NODE_RUNS
from core import config, tracing
from core.state import MainState, load_fixture
from graph import NODE_INPUTS, NODES, build_graph, select_tech
from report_pdf import PDFRenderError, render_pdf

log = logging.getLogger("app")

# 샘플 파일이 있는 키. 노드 하나만 돌릴 때 이 키들은 파일에서 읽어 채운다.
FIXTURE_KEYS = {"tech_research", "trl_eval", "market_eval", "stakeholder_eval", "domain_eval", "synthesis"}
DEFAULT_CHECK = {"passed": True, "issues": [], "attempt": 0}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kv-cache-eval 실행")
    parser.add_argument("--dry-run", action="store_true", help="LLM·검색 대신 fixtures 사용")
    parser.add_argument("--only", metavar="NODE", choices=list(NODES), help="노드 하나만 실행 (입력은 fixtures로 채움)")
    parser.add_argument("--criteria-limit", type=int, metavar="N", help="관점당 앞의 N개 기준만")
    parser.add_argument("--retry", type=int, metavar="N", help="모든 재시도 상한을 N으로")
    parser.add_argument("--no-cache", action="store_true", help="캐시를 읽지도 쓰지도 않음")
    parser.add_argument("--no-pdf", action="store_true", help="보고서 PDF를 만들지 않음")
    return parser.parse_args(argv)


def setup_logging() -> None:
    """화면과 파일에 함께 로그를 남긴다. 실행이 길어 끝난 뒤 되짚어 볼 일이 많다."""
    log_dir = config.resolve("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "app.log", encoding="utf-8")],
    )


def fill_inputs(node: str) -> MainState:
    """노드 하나만 돌릴 때 그 노드가 읽는 키를 채운다.

    앞 단계 결과는 샘플 파일에서 가져오고, 대상과 기준은 실제로 읽는다. 보고서 검사처럼 앞 노드의
    출력이 필요한 경우에는 그 노드를 먼저 돌려 값을 만든다.
    """
    state: MainState = {}
    keys = NODE_INPUTS[node]
    if {"selected_tech", "criteria"} & set(keys):
        state.update(select_tech(state))
    for key in keys:
        if key in FIXTURE_KEYS:
            state[key] = load_fixture(key)
        elif key in ("synthesis_check", "report_check"):
            state[key] = dict(DEFAULT_CHECK)
        elif key == "final_report":
            state[key] = NODES["report"](state)["final_report"]
    return state


def run_only(node: str) -> MainState:
    """노드 하나만 돌리고 결과를 파일로 남긴다. 프롬프트를 고친 뒤 그 부분만 확인할 때 쓴다."""
    state = fill_inputs(node)
    log.info("--only %s: 입력 키 %s", node, sorted(state))
    result = NODES[node](state)
    out_path = config.resolve("outputs") / f"{node}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("결과 저장: %s", out_path)
    return result


def write_report(text: str, *, create_pdf: bool = True) -> tuple[Path, Path | None]:
    """보고서를 마크다운으로 쓰고, 요청한 경우 같은 이름의 PDF도 만든다."""
    path = config.resolve(config.get()["paths"]["report"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    pdf_path = render_pdf(text, path.with_suffix(".pdf")) if create_pdf else None
    return path, pdf_path


def log_node_runs() -> None:
    """어느 노드가 몇 번 돌았는지 남긴다. 재시도가 어디에 몰렸는지 여기서 드러난다."""
    main_nodes = [(n, NODE_RUNS[n]) for n in NODES if NODE_RUNS[n]]
    sub_nodes = sorted((n, c) for n, c in NODE_RUNS.items() if "/" in n)
    log.info("노드 실행 횟수 (메인): %s", ", ".join(f"{n}={c}" for n, c in main_nodes) or "-")
    log.info("노드 실행 횟수 (서브그래프): %s", ", ".join(f"{n}={c}" for n, c in sub_nodes) or "-")


def main(argv: list[str] | None = None) -> int:
    """설정을 맞추고 그래프 전체나 노드 하나를 돌린다."""
    load_dotenv(override=True)
    args = parse_args(argv)
    config.apply_cli(dry_run=args.dry_run, retry=args.retry, criteria_limit=args.criteria_limit, no_cache=args.no_cache)
    setup_logging()
    cfg = config.get()
    tracing.configure()
    log.info(
        "dry_run=%s retry=%s criteria_limit=%s cache=%s",
        cfg["dry_run"], cfg["retry"], cfg["execution"]["criteria_limit"], cfg["cache"]["enabled"],
    )

    # 모델 이름이 없으면 한참 돌다 중간에 실패한다. 시작 전에 막고 무엇을 채워야 하는지 알려 준다.
    if not cfg["dry_run"]:
        missing = [role for role in ("generator", "judge") if not (cfg["models"].get(role) or "").strip()]
        if missing:
            log.error(
                "config.yaml의 models.%s이 비어 있다. .env에 OPENAI_API_KEY를 넣고 `python -m core.llm --list`로 "
                "접근 가능한 모델을 확인해 채우거나, --dry-run으로 실행하라.",
                "/".join(missing),
            )
            return 2

    result = (
        run_only(args.only)
        if args.only
        else build_graph().invoke(
            {},
            config=tracing.run_config(dry_run=cfg["dry_run"], mode="full"),
        )
    )

    if result.get("final_report"):
        try:
            markdown_path, pdf_path = write_report(result["final_report"], create_pdf=not args.no_pdf)
        except PDFRenderError as exc:
            log.error("PDF 생성 실패: %s", exc)
            return 3
        log.info("마크다운 보고서 저장: %s", markdown_path)
        if pdf_path:
            log.info("PDF 보고서 저장: %s", pdf_path)
    log_node_runs()
    return 0


if __name__ == "__main__":
    sys.exit(main())
