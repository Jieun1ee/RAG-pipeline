"""노드 사이를 오가는 값의 형식.

모델이 자유 문장을 돌려주면 다음 노드가 그것을 다시 해석해야 하고, 그 과정에서 근거가 사라진다.
그래서 모든 중간 결과를 여기 정의한 모델로 받는다. 상태에는 model_dump()한 dict를 넣고,
쓰는 쪽에서 Model.model_validate()로 되돌린다.

    python -m core.schemas --validate data/fixtures/     샘플 JSON이 이 형식에 맞는지 확인
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

Tech = Literal["MLA", "ITME"]
Perspective = Literal["tech", "trl", "market", "stakeholder", "domain"]


class Retrieved(BaseModel):
    """검색 도구가 실제로 돌려준 한 건.

    논문 검색과 웹 검색이 같은 형식으로 맞춰 돌려주므로 뒤쪽 코드는 둘을 구분하지 않아도 된다.
    인용이 실재하는지 확인할 때 대조 기준이 되는 것도 이 값이다.
    """

    id: str  # 논문은 청크 id, 웹은 url
    text: str
    source_type: Literal["paper", "web"]
    title: str | None = None
    url: str | None = None
    date: str | None = None  # YYYY 또는 YYYY-MM-DD
    author_or_org: str | None = None
    page: int | None = None
    doc_id: str | None = None


class Evidence(BaseModel):
    """주장 하나를 뒷받침하는 인용과 그 출처 정보."""

    ref_id: str  # 인용한 Retrieved의 id와 같아야 한다
    quote: str  # 원문 그대로. 논문이면 청크 본문 안에 있어야 한다
    source_type: Literal["paper", "web", "patent"]
    source_nature: Literal["논문", "기업 공식 자료", "산업 뉴스·리포트", "개발자 의견"]
    is_self_reported: bool  # 그 기술을 만든 쪽이 직접 발표한 자료인지
    title: str
    author_or_org: str
    venue: str
    date: str
    url: str | None = None
    locator: str | None = None  # 쪽 번호나 절 같은 위치 정보
    summary: str


class Finding(BaseModel):
    """기술 하나에 대한 서술과 그 근거.

    두 기술을 비교하는 문장은 여기에 담지 않는다. 비교는 관점별 결과를 모두 모은 뒤
    종합 단계에서만 하고, 이 단계에서는 각 기술을 따로 기술한다.
    """

    id: str  # 예: market-MKT-1-ITME-01
    technology: Tech
    criterion: str  # 평가 기준 id
    claim: str
    polarity: Literal["strength", "limitation", "neutral"]
    evidence: list[Evidence] = Field(min_length=1)
    conditions: str | None = None  # 수치를 인용했다면 그 측정 조건
    confidence: Literal["high", "medium", "low"]


class CriterionAssessment(BaseModel):
    """기준 하나와 기술 하나에 대한 판정. 서브그래프를 한 번 돌리면 하나가 나온다."""

    id: str  # 예: MKT-1-ITME
    criterion: str
    technology: Tech
    status: Literal["assessed", "insufficient_evidence", "not_applicable"]
    levels: dict[str, str] = {}  # 판정 축과 수준. 예: {"채택 위험": "중간"}
    rationale: str
    finding_ids: list[str]


class PerspectiveResult(BaseModel):
    """관점 하나가 내놓는 결과 묶음."""

    perspective: Perspective
    assessments: list[CriterionAssessment]
    findings: list[Finding]
    dissent: list[Finding] = []  # 같은 기준에서 반대 방향을 가리킨 것
    gaps: list[str] = []  # 근거를 확보하지 못해 결과에서 뺀 항목


class TRLEstimate(BaseModel):
    """기술 하나의 성숙도 구간 추정.

    단계마다 어떤 수준의 근거가 있었는지 stage_evidence에 남겨, 어디까지가 확인된 사실이고
    어디부터가 추정인지 드러나게 한다.
    """

    technology: Tech
    status: Literal["estimated", "withheld"]  # withheld는 판단을 미룬 경우
    trl_min: int | None = None
    trl_max: int | None = None
    stage_evidence: dict[int, Literal["direct", "indirect", "none"]]  # 1단계부터 9단계까지
    inference_note: str
    confidence: Literal["high", "medium", "low"]
    disclaimer: Literal["공개 정보 기반 추정"] = "공개 정보 기반 추정"


class TRLResult(PerspectiveResult):
    """성숙도 관점의 결과. 공통 항목에 기술별 구간 추정을 더한다."""

    estimates: list[TRLEstimate]


class SynthesisItem(BaseModel):
    """여러 관점의 결과를 묶은 서술 한 줄.

    finding_ids를 두 개 이상 요구해, 관점 하나만 보고 내린 결론이 종합에 섞이지 않게 한다.
    """

    statement: str
    finding_ids: list[str] = Field(min_length=2)


class SynthesisResult(BaseModel):
    """관점들이 같은 방향을 가리킨 것과 엇갈린 것."""

    agreements: list[SynthesisItem]
    conflicts: list[SynthesisItem]


class CheckResult(BaseModel):
    """검사 결과. attempt는 같은 지점을 몇 번째 시도하는지로, 재시도 상한 판단에 쓴다."""

    passed: bool
    issues: list[str] = []
    attempt: int = 0


# 아래는 모델 응답을 받기 위한 형식이다. 상태에 그대로 들어가지 않고,
# 코드가 값을 보태거나 걸러 낸 뒤 위 모델로 옮겨 담는다.
# 클래스 설명과 필드 설명은 구조화 출력을 요청할 때 모델에게 함께 전달되므로, 지시문처럼 쓴다.


class QueryPair(BaseModel):
    """같은 기준을 서로 다른 방향에서 찾기 위한 검색 질의 두 개."""

    effect: str = Field(description="효과나 채택 근거를 찾는 검색 질의")
    limitation: str = Field(description="한계나 반대 근거를 찾는 검색 질의")


class RelevanceJudgement(BaseModel):
    """검색 결과 중 어느 것이 그 기준에 쓸 만한지."""

    relevant_ids: list[str] = Field(
        default_factory=list, description="기준에 답이 될 만한 검색 결과의 id. 번호가 아니라 id를 그대로 쓴다"
    )
    reason: str = Field(description="그렇게 고른 이유. 쓸 만한 것이 없다면 무엇이 부족한지 한 문장")


class EvidenceDraft(BaseModel):
    """주장을 뒷받침하는 인용 하나."""

    ref_id: str = Field(description="인용한 검색 결과의 id")
    quote: str = Field(description="검색 결과 본문에서 글자 그대로 옮긴 문장. 바꿔 쓰거나 번역하지 않는다")
    source_nature: Literal["논문", "기업 공식 자료", "산업 뉴스·리포트", "개발자 의견"] = Field(
        description="출처의 성격"
    )
    is_self_reported: bool = Field(description="그 기술을 만든 쪽이 직접 발표한 자료이면 참")
    locator: str | None = Field(default=None, description="쪽 번호나 절처럼 인용 위치를 가리키는 표시")
    summary: str = Field(description="그 인용이 무엇을 보여 주는지 한 문장")


class FindingDraft(BaseModel):
    """기술 하나에 대한 서술과 근거."""

    claim: str = Field(description="기술 하나에 대한 서술. 다른 기술과 비교하지 않는다")
    polarity: Literal["strength", "limitation", "neutral"] = Field(description="서술의 방향")
    evidence: list[EvidenceDraft] = Field(min_length=1, description="그 서술을 뒷받침하는 인용. 하나 이상")
    conditions: str | None = Field(default=None, description="수치를 인용했다면 그 측정 조건")
    confidence: Literal["high", "medium", "low"] = Field(description="근거가 직접적일수록 높게")


class FindingList(BaseModel):
    """서술 묶음."""

    findings: list[FindingDraft] = Field(
        default_factory=list, description="근거로 뒷받침되는 서술. 근거가 없으면 빈 목록으로 둔다"
    )


class SupportJudgement(BaseModel):
    """인용이 주장을 실제로 뒷받침하는지."""

    supported: bool = Field(description="인용이 주장을 직접 뒷받침하면 참")
    reason: str = Field(description="그렇게 판단한 이유 한 문장")


class LevelEntry(BaseModel):
    """판정 축 하나와 거기에 매긴 수준."""

    axis: str = Field(description="판정 축 이름")
    level: str = Field(description="수준 정의에 있는 값 중 하나")


class LevelJudgement(BaseModel):
    """기준 하나에 대한 수준 판정."""

    status: Literal["assessed", "insufficient_evidence", "not_applicable"] = Field(description="판정 결과의 상태")
    levels: list[LevelEntry] = Field(default_factory=list, description="축마다 매긴 수준. 판정 축이 없으면 빈 목록")
    rationale: str = Field(description="판정 근거. 어느 서술을 근거로 삼았는지 id로 밝힌다")


class NeutralityJudgement(BaseModel):
    """두 기술의 우열을 가리는 표현이 있는지."""

    passed: bool = Field(description="우열을 가리는 표현이 없으면 참")
    issues: list[str] = Field(default_factory=list, description="문제가 되는 문장을 그대로 담는다")


class ReportDraft(BaseModel):
    """보고서 본문."""

    markdown: str = Field(description="마크다운 보고서 전문")


class QuestionDraft(BaseModel):
    """특정 대목을 읽어야만 답할 수 있는 질문."""

    question: str = Field(description="주어진 본문에 있는 고유한 사실을 가리키는 질문")


# 샘플 JSON 검증. 파일 이름으로 어떤 형식인지 정해 두어,
# 샘플을 만드는 쪽이 코드를 몰라도 스스로 확인할 수 있게 한다.

_SCHEMA_BY_STEM: dict[str, type[BaseModel]] = {
    "tech_research": PerspectiveResult,
    "trl_eval": TRLResult,
    "market_eval": PerspectiveResult,
    "stakeholder_eval": PerspectiveResult,
    "domain_eval": PerspectiveResult,
    "synthesis": SynthesisResult,
}
_RETRIEVED_LIST = TypeAdapter(list[Retrieved])


def schema_for(stem: str) -> type[BaseModel] | TypeAdapter | None:
    """확장자를 뗀 파일 이름에 대응하는 형식. 정해 둔 이름이 아니면 None."""
    if stem.startswith("retrieved_"):
        return _RETRIEVED_LIST
    return _SCHEMA_BY_STEM.get(stem)


def validate_file(path: Path) -> tuple[str, bool, str]:
    """파일 하나를 검증하고 (형식 이름, 통과 여부, 메시지)를 돌려준다."""
    schema = schema_for(path.stem)
    if schema is None:
        return ("-", True, "SKIP (매핑된 스키마 없음)")
    name = "list[Retrieved]" if isinstance(schema, TypeAdapter) else schema.__name__
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(schema, TypeAdapter):
            schema.validate_python(data)
        else:
            schema.model_validate(data)
    except (ValidationError, ValueError) as exc:
        first = str(exc).strip().splitlines()
        return (name, False, " | ".join(first[:3]))  # 오류 전문은 길어서 앞 세 줄만 보여 준다
    return (name, True, "OK")


def validate_dir(directory: Path) -> list[tuple[str, str, bool, str]]:
    """디렉토리 안의 JSON을 파일 이름순으로 모두 검증한다."""
    rows = []
    for path in sorted(directory.glob("*.json")):
        name, ok, msg = validate_file(path)
        rows.append((path.name, name, ok, msg))
    return rows


def _print_table(rows: list[tuple[str, str, bool, str]]) -> None:
    w_file = max([len(r[0]) for r in rows] + [4])
    w_schema = max([len(r[1]) for r in rows] + [6])
    print(f"{'file':<{w_file}}  {'schema':<{w_schema}}  result")
    print(f"{'-' * w_file}  {'-' * w_schema}  ------")
    for file, schema, ok, msg in rows:
        mark = "PASS" if ok else "FAIL"
        print(f"{file:<{w_file}}  {schema:<{w_schema}}  {mark} {msg if msg != 'OK' else ''}".rstrip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="JSON 파일이 정의된 형식에 맞는지 검사한다")
    parser.add_argument("--validate", metavar="DIR", required=True, help="검사할 디렉토리")
    args = parser.parse_args(argv)
    directory = Path(args.validate)
    if not directory.is_dir():
        print(f"디렉토리가 없다: {directory}", file=sys.stderr)
        return 2
    rows = validate_dir(directory)
    if not rows:
        print(f"JSON 파일이 없다: {directory}", file=sys.stderr)
        return 2
    _print_table(rows)
    failed = [r for r in rows if not r[2]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} 통과")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
