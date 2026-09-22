"""명령줄 진입점.

설정을 읽어 실행 조건을 맞추고 그래프를 돌린다. 그래프 조립은 graph.py가 맡는다.

python app.py                         전부 실행하고 보고서를 남긴다
python app.py --dry-run               모델과 검색을 부르지 않고 샘플 값으로 끝까지 통과시킨다
python app.py --only market_eval      노드 하나만 돌린다. 입력은 샘플에서 채운다
python app.py --criteria-limit 1      관점마다 기준 하나씩만 본다
python app.py --retry 0               검사에 걸려도 다시 쓰지 않는다
python app.py --no-cache              저장해 둔 응답을 쓰지 않는다
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from core import config

log = logging.getLogger("app")

# 그래프에 놓일 노드 이름. 상태 키와 같은 이름을 쓴다 (core/state.py의 MainState).
NODE_NAMES = [
    "select_tech",
    "tech_research",
    "trl_eval",
    "market_eval",
    "stakeholder_eval",
    "domain_eval",
    "synthesis",
    "synthesis_check",
    "report",
    "report_check",
]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kv-cache-eval 실행")
    parser.add_argument("--dry-run", action="store_true", help="LLM·검색 대신 fixtures 사용")
    parser.add_argument("--only", metavar="NODE", choices=NODE_NAMES, help="노드 하나만 실행 (입력은 fixtures로 채움)")
    parser.add_argument("--criteria-limit", type=int, metavar="N", help="관점당 앞의 N개 기준만")
    parser.add_argument("--retry", type=int, metavar="N", help="모든 재시도 상한을 N으로")
    parser.add_argument("--no-cache", action="store_true", help="캐시를 읽지도 쓰지도 않음")
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


def write_report(text: str) -> Path:
    """보고서를 설정에 적힌 경로에 쓴다."""
    path = config.resolve(config.get()["paths"]["report"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def check_models(cfg: dict) -> bool:
    """모델 이름이 채워져 있는지 본다.

    비어 있으면 한참 돌다 중간에 실패한다. 시작 전에 막고 무엇을 채워야 하는지 알려 준다.
    """
    missing = [role for role in ("generator", "judge") if not (cfg["models"].get(role) or "").strip()]
    if not missing:
        return True
    log.error(
        "config.yaml의 models.%s이 비어 있다. .env에 OPENAI_API_KEY를 넣고 `python -m core.llm --list`로 "
        "접근 가능한 모델을 확인해 채우거나, --dry-run으로 실행하라.",
        "/".join(missing),
    )
    return False


def main(argv: list[str] | None = None) -> int:
    """설정을 맞추고 그래프를 돌린다."""
    load_dotenv(override=True)
    args = parse_args(argv)
    config.apply_cli(dry_run=args.dry_run, retry=args.retry, criteria_limit=args.criteria_limit, no_cache=args.no_cache)
    setup_logging()
    cfg = config.get()
    log.info(
        "dry_run=%s retry=%s criteria_limit=%s cache=%s",
        cfg["dry_run"], cfg["retry"], cfg["execution"]["criteria_limit"], cfg["cache"]["enabled"],
    )

    if not cfg["dry_run"] and not check_models(cfg):
        return 2

    log.error("그래프가 아직 없다. graph.py와 agents/의 노드가 들어온 뒤에 실행할 수 있다.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
