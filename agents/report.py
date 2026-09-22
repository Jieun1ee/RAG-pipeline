"""최종 보고서를 만든다.

목차는 템플릿으로 고정하고 각 절의 내용만 상태에서 가져온다. 참고문헌은 모델이 쓰지 않고 코드가 붙인다.
본문에 실제로 인용된 번호만 모으기 때문에, 쓰지도 않은 자료가 목록에 오르거나 번호가 어긋나는 일이 없다.
확인하지 못한 항목과 검사에서 풀리지 않은 지적도 본문에 함께 싣는다. 빠뜨린 것을 숨기지 않기 위해서다.
"""

from __future__ import annotations

import logging
import re

import yaml

from agents.synthesis import LABEL, summarize_result
from core import config, llm, prompts
from core.schemas import ReportDraft
from core.state import MainState

log = logging.getLogger(__name__)

KEY = "final_report"
RESULT_KEYS = ("tech_research", "trl_eval", "market_eval", "stakeholder_eval", "domain_eval")
PERSPECTIVE_SECTIONS = (
    ("trl_eval", "4.1 기술 성숙도 관점"),
    ("market_eval", "4.2 시장성 관점"),
    ("stakeholder_eval", "4.3 이해관계자 관점"),
    ("domain_eval", "4.4 데이터센터 적용성 관점"),
)
_CITATION = re.compile(r"\[(\d+)\]")
_REFERENCE_HEADING = re.compile(r"^#{1,6}\s*REFERENCE.*$", re.MULTILINE | re.IGNORECASE)
_ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+?)(?:v\d+)?/?$", re.IGNORECASE)


# --- 참고문헌 -------------------------------------------------------------------


def source_key(evidence: dict) -> str:
    """같은 자료를 한 항목으로 묶는 열쇠.

    논문은 쪽마다 인용 id가 다르지만 출처는 문서 하나다. 문서 단위로 묶어야 같은 논문이
    참고문헌에 여러 번 오르지 않는다. 웹은 주소가 곧 자료다.
    """
    ref_id = evidence.get("ref_id", "")
    if evidence.get("source_type") == "paper" and ":" in ref_id:
        return ref_id.split(":", 1)[0]
    return evidence.get("url") or ref_id


def build_references(state: MainState) -> tuple[dict[str, int], dict[int, dict]]:
    """모든 서술의 인용을 훑어 자료마다 번호를 매긴다.

    인용 id로 번호를 찾는 표와, 번호로 자료를 찾는 표를 함께 돌려준다. 한 자료를 여러 서술이
    인용해도 번호는 하나다.
    """
    numbers: dict[str, int] = {}
    entries: dict[int, dict] = {}
    by_source: dict[str, int] = {}
    for key in RESULT_KEYS:
        result = state.get(key, {})
        for f in result.get("findings", []) + result.get("dissent", []):
            for ev in f.get("evidence", []):
                key_of_source = source_key(ev)
                if key_of_source not in by_source:
                    by_source[key_of_source] = len(by_source) + 1
                    entries[by_source[key_of_source]] = ev
                numbers[ev["ref_id"]] = by_source[key_of_source]
    return numbers, entries


def _identifier(url: str) -> str:
    """논문 항목 끝에 붙일 식별자. arXiv 주소면 번호만 남기고, 아니면 주소를 그대로 쓴다."""
    match = _ARXIV.search(url or "")
    return match.group(1) if match else (url or "")


def format_reference(number: int, evidence: dict) -> str:
    """참고문헌 한 줄. 자료 종류마다 형식이 다르다.

    논문은 저자(연도). 제목. 발행처, 식별자.
    특허는 출원인(연월). 특허명, 번호, 주소.
    웹은 기관 또는 작성자(연월일). 제목. 사이트명, 주소.
    """
    author = evidence.get("author_or_org") or "작성자 미상"
    when = evidence.get("date") or "n.d."
    title = evidence.get("title") or "제목 미상"
    venue = evidence.get("venue") or ""
    url = evidence.get("url") or ""
    head = f"[{number}] {author}({when}). {title}"

    if evidence.get("source_type") == "paper":
        tail = ", ".join(x for x in (venue, _identifier(url)) if x)
        return f"{head}. {tail}." if tail else f"{head}."
    if evidence.get("source_type") == "patent":
        tail = ", ".join(x for x in (venue, url) if x)
        return f"{head}, {tail}" if tail else f"{head}."
    tail = ", ".join(x for x in (venue, url) if x)
    return f"{head}. {tail}" if tail else f"{head}."


def strip_reference_section(markdown: str) -> str:
    """모델이 참고문헌 절까지 써 왔으면 잘라 낸다. 그 자리는 코드가 다시 채운다."""
    match = _REFERENCE_HEADING.search(markdown)
    return markdown[: match.start()].rstrip() if match else markdown.rstrip()


def attach_references(body: str, entries: dict[int, dict]) -> str:
    """본문에 실제로 나온 번호만 모아 참고문헌 절을 붙인다."""
    cited = sorted({int(n) for n in _CITATION.findall(body) if int(n) in entries})
    lines = [format_reference(n, entries[n]) for n in cited]
    return body.rstrip() + "\n\n# REFERENCE\n\n" + ("\n".join(lines) if lines else "(본문에 인용된 출처가 없다)") + "\n"


# --- 프롬프트 입력 ---------------------------------------------------------------


def _format_findings_with_refs(result: dict, numbers: dict[str, int]) -> str:
    """서술마다 출처 번호를 붙인 목록. 모델은 여기 붙은 번호만 본문에 옮겨 쓴다."""
    lines = []
    for f in result.get("findings", []) + result.get("dissent", []):
        refs = " ".join(f"[{numbers[ev['ref_id']]}]" for ev in f.get("evidence", []) if ev["ref_id"] in numbers)
        cond = f" (조건: {f['conditions']})" if f.get("conditions") else ""
        lines.append(f"- {f['id']} [{f['technology']}, {f['polarity']}] {f['claim']}{cond} {refs}")
    return "\n".join(lines) or "(없음)"


def _format_result(key: str, result: dict, numbers: dict[str, int]) -> str:
    """관점 결과 요약과 번호가 붙은 서술 목록을 이어 붙인다."""
    return summarize_result(key, {**result, "findings": [], "dissent": []}) + "\n" + _format_findings_with_refs(result, numbers)


def template_text() -> str:
    """머리글 메모를 떼어 낸 목차 본문."""
    return prompts.body("report_template")


# --- 노드 -----------------------------------------------------------------------


def node(state: MainState) -> MainState:
    """보고서 본문을 쓰고 참고문헌을 붙여 돌려준다.

    모델이 응답을 주지 못하면 코드로 조립한 보고서로 대신한다. 마지막 단계에서 빈손으로 끝나지 않게 하려는 것이다.
    """
    if config.is_dry_run():
        return {KEY: _dry_run_report(state)}

    numbers, entries = build_references(state)
    results = "\n\n".join(_format_result(key, state.get(key, {}), numbers) for key, _ in PERSPECTIVE_SECTIONS)
    gaps = [f"({LABEL.get(key, key)}) {gap}" for key in RESULT_KEYS for gap in state.get(key, {}).get("gaps", [])]
    gaps += [f"(종합 검증 미해결) {issue}" for issue in state.get("synthesis_check", {}).get("issues", [])]
    feedback = "\n".join(state.get("report_check", {}).get("issues", [])) or "(없음)"

    prompt = prompts.render(
        "report",
        template=template_text(),
        selected_tech=yaml.safe_dump(state.get("selected_tech", []), allow_unicode=True, sort_keys=False).strip(),
        tech_research=_format_result("tech_research", state.get("tech_research", {}), numbers),
        results=results,
        synthesis=yaml.safe_dump(state.get("synthesis", {}), allow_unicode=True, sort_keys=False).strip(),
        gaps="\n".join(f"- {g}" for g in gaps) or "(없음)",
        feedback=feedback,
    )
    draft = llm.generate(prompt, ReportDraft)
    if draft is None or not draft.markdown.strip():
        log.warning("report: 구조화 출력 실패, 코드 조립 보고서로 대체")
        body = strip_reference_section(_dry_run_report(state))
    else:
        body = strip_reference_section(draft.markdown)
    return {KEY: attach_references(body, entries)}


def _dry_run_report(state: MainState) -> str:
    """모델 없이 상태의 값만으로 목차를 채운 보고서.

    모아 둔 값이 각 절에 제대로 실리는지 확인하는 용도다. 절 제목과 번호는 실제 보고서와 같고,
    서술은 다듬지 않은 채로 들어간다.
    """
    numbers, entries = build_references(state)
    techs = state.get("selected_tech", [])
    criteria = state.get("criteria", {})

    def refs(finding: dict) -> str:
        return " ".join(f"[{numbers[ev['ref_id']]}]" for ev in finding.get("evidence", []) if ev["ref_id"] in numbers)

    def claims(key: str, technology: str | None = None) -> list[str]:
        result = state.get(key, {})
        out = []
        for f in result.get("findings", []) + result.get("dissent", []):
            if technology and f.get("technology") != technology:
                continue
            out.append(f"- {f['claim']} {refs(f)}".rstrip())
        return out or ["- (검증을 통과한 근거 없음)"]

    lines = ["# SUMMARY", "", "모델 호출 없이 상태의 값만으로 채운 확인용 출력이다.", ""]

    lines += ["# 1. 분석 개요", "", "## 1.1 분석 배경 및 도메인 맥락", ""]
    lines.append(f"- 평가 도메인: {criteria.get('domain_name', '(미지정)')}")
    lines += ["", "## 1.2 분석 목적 및 범위", ""]
    lines.append(f"- 평가 대상: {', '.join(t.get('technology', '') for t in techs) or '(없음)'}")
    lines += ["", "## 1.3 평가 관점 및 기준", ""]
    for perspective, spec in criteria.items():
        if isinstance(spec, dict) and "items" in spec:
            lines.append(f"- {spec.get('name', perspective)}: 기준 {len(spec['items'])}개")

    lines += ["", "# 2. 분석 대상 선정", "", "## 2.1 선정 기준", "", "- (선정 기준 서술)", "", "## 2.2 선정 기술 및 선정 사유", ""]
    for tech in techs:
        lines.append(f"- {tech.get('technology')} ({tech.get('camp', '')}) {tech.get('name', '')}: {tech.get('reason', '')}")

    lines += ["", "# 3. 기술 개요", ""]
    for index, tech in enumerate(techs, start=1):
        lines += [f"## 3.{index} {tech.get('technology')}", ""]
        lines += claims("tech_research", tech.get("technology"))
        lines.append("")

    lines += ["# 4. 다관점 평가", ""]
    for key, section in PERSPECTIVE_SECTIONS:
        result = state.get(key, {})
        lines += [f"## {section}", ""]
        for assessment in result.get("assessments", []):
            levels = ", ".join(f"{k}: {v}" for k, v in assessment.get("levels", {}).items())
            suffix = f" ({levels})" if levels else ""
            lines.append(f"- {assessment['id']}: {assessment['status']}{suffix}. {assessment['rationale']}")
        lines += claims(key)
        if key == "trl_eval":
            for estimate in result.get("estimates", []):
                lines.append(
                    f"- {estimate['technology']} TRL {estimate.get('trl_min')}~{estimate.get('trl_max')} "
                    f"({estimate['status']}, {estimate.get('disclaimer', '공개 정보 기반 추정')})"
                )
        lines.append("")

    synthesis = state.get("synthesis", {})
    lines += ["## 4.5 다관점 평가 종합", ""]
    lines += [f"- 공통: {item['statement']} ({', '.join(item['finding_ids'])})" for item in synthesis.get("agreements", [])]
    lines += [f"- 엇갈림: {item['statement']} ({', '.join(item['finding_ids'])})" for item in synthesis.get("conflicts", [])]

    lines += ["", "# 5. 시사점", ""]
    for number, title in (
        (1, "기술적 시사점"),
        (2, "시장·산업적 시사점"),
        (3, "데이터센터 적용 시사점"),
        (4, "이해관계자별 시사점"),
        (5, "향후 관찰이 필요한 지표"),
    ):
        lines += [f"## 5.{number} {title}", "", "- (종합 결과에서 도출)", ""]

    lines += ["# 6. 분석 한계 및 근거 검증", "", "## 6.1 공개 정보 기반 분석의 한계", ""]
    gaps = [gap for key in RESULT_KEYS for gap in state.get(key, {}).get("gaps", [])]
    lines += [f"- {gap}" for gap in gaps] or ["- (미확인 항목 없음)"]

    self_reported = sum(
        1
        for key in RESULT_KEYS
        for f in state.get(key, {}).get("findings", [])
        for ev in f.get("evidence", [])
        if ev.get("is_self_reported")
    )
    lines += ["", "## 6.2 자료 및 출처의 한계", "", f"- 인용한 자료 {len(entries)}건 중 개발사 자체 발표 {self_reported}건", ""]

    lines += ["## 6.3 추정·판단의 한계", ""]
    for estimate in state.get("trl_eval", {}).get("estimates", []):
        lines.append(f"- {estimate['technology']}: {estimate.get('inference_note', '')}")
    lines += [f"- (종합 검증 미해결) {issue}" for issue in state.get("synthesis_check", {}).get("issues", [])]

    lines += ["", "## 6.4 상반된 근거 및 확증편향 검토", ""]
    dissent = [f for key in RESULT_KEYS for f in state.get(key, {}).get("dissent", [])]
    lines += [f"- {f['id']}: {f['claim']} {refs(f)}".rstrip() for f in dissent] or ["- (반대 방향 근거 없음)"]

    return attach_references("\n".join(lines), entries)
