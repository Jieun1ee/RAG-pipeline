"""그래프가 단계 사이에 주고받는 상태 정의.

값은 Pydantic 객체가 아니라 model_dump()한 dict로 넣는다. 그래야 직렬화와 샘플 파일 저장이
단순해지고, 읽는 쪽에서 Model.model_validate()로 복원하면 된다.
관점 노드들이 병렬로 돌지만 서로 다른 키에만 쓰기 때문에 값을 합치는 reducer는 두지 않았다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, TypedDict


class MainState(TypedDict, total=False):
    """전체 그래프가 공유하는 상태. 키 이름을 노드 이름과 같게 맞춰 두었다."""

    selected_tech: Annotated[list[dict], "평가 대상 기술과 선정 사유"]
    criteria: Annotated[dict, "관점 이름으로 묶은 평가 기준"]
    tech_research: Annotated[dict, "기술 조사 결과 (PerspectiveResult)"]
    trl_eval: Annotated[dict, "기술 성숙도 평가 결과 (TRLResult)"]
    market_eval: Annotated[dict, "시장성 평가 결과 (PerspectiveResult)"]
    stakeholder_eval: Annotated[dict, "이해관계자 평가 결과 (PerspectiveResult)"]
    domain_eval: Annotated[dict, "도메인 적합성 평가 결과 (PerspectiveResult)"]
    synthesis: Annotated[dict, "관점 종합 결과 (SynthesisResult)"]
    synthesis_check: Annotated[dict, "종합 결과 검사 (CheckResult)"]
    final_report: Annotated[str, "보고서 본문"]
    report_check: Annotated[dict, "보고서 검사 (CheckResult)"]


class SubState(TypedDict, total=False):
    """기준 하나와 기술 하나를 처리하는 서브그래프의 상태."""

    task: Annotated[dict, "처리할 기준과 기술, 그에 딸린 설정"]
    query: Annotated[str, "검색 질의. 줄 단위로 여러 개"]
    retrieved: Annotated[list[dict], "검색 결과 (Retrieved)"]
    retrieval_check: Annotated[dict, "검색 결과 관련성 검사 (CheckResult)"]
    findings: Annotated[list[dict], "근거를 갖춘 서술 (Finding)"]
    citation_check: Annotated[dict, "인용 실재와 근거 타당성 검사 (CheckResult)"]
    gaps: Annotated[list[str], "근거를 확보하지 못해 뺀 항목과 그 이유"]


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def load_fixture(name: str) -> Any:
    """data/fixtures/<name>.json을 읽어 파싱한 값을 돌려준다.

    아직 실행하지 않은 노드의 결과를 대신 채우는 데 쓴다. 덕분에 앞 단계가 끝나지 않아도
    뒤 단계를 따로 돌려 볼 수 있고, 외부 호출 없이 그래프 전체를 통과시킬 수 있다.
    """
    path = FIXTURES_DIR / (name if name.endswith(".json") else f"{name}.json")
    if not path.exists():
        raise FileNotFoundError(f"픽스처가 없다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
