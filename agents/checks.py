"""서술과 보고서가 규칙을 지켰는지 확인한다.

검사는 두 갈래다. 인용이 검색 결과에 실제로 있는지, 빠진 항목이 없는지는 코드로 본다. 값이 싸고
결과가 늘 같다. 인용이 주장을 뒷받침하는지, 두 기술의 우열을 가리는 표현이 있는지는 글의 뜻을
읽어야 하므로 모델에게 맡긴다.

앞쪽 함수들은 기준 하나를 처리하는 도중에 불리고, synthesis_check와 report_check는 그래프의 노드로 붙는다.
"""

from __future__ import annotations

import re

from pydantic import ValidationError

from agents.synthesis import EVAL_KEYS, all_finding_ids
from core import config, llm, prompts
from core.schemas import CheckResult, Finding, NeutralityJudgement, SupportJudgement
from core.state import MainState
from rag.textnorm import normalize

TRL_DISCLAIMER = "공개 정보 기반 추정"


def _find(ref_id: str, retrieved: list[dict]) -> dict | None:
    return next((r for r in retrieved if r.get("id") == ref_id), None)


def rag_check(evidence: dict, retrieved: list[dict]) -> CheckResult:
    """논문 인용을 확인한다. 가리키는 청크가 검색 결과에 있고 인용문이 그 본문 안에 있어야 한다."""
    ref_id = evidence.get("ref_id", "")
    quote = evidence.get("quote", "")
    issues: list[str] = []
    chunk = _find(ref_id, retrieved)
    if chunk is None:
        issues.append(f"ref_id '{ref_id}'가 검색 결과에 없다")
    elif not quote.strip():
        issues.append(f"ref_id '{ref_id}': quote가 비어 있다")
    elif normalize(quote) not in normalize(chunk.get("text", "")):
        issues.append(f"ref_id '{ref_id}': 인용문이 청크 본문에 없다: {quote[:60]!r}")
    return CheckResult(passed=not issues, issues=issues)


def web_check(evidence: dict, retrieved: list[dict]) -> CheckResult:
    """웹 인용을 확인한다. 주소가 검색 결과에 있으면 통과한다.

    웹 본문은 나중에 바뀔 수 있어 글자 대조까지는 하지 않는다.
    """
    ref_id = evidence.get("ref_id", "")
    issues: list[str] = []
    if _find(ref_id, retrieved) is None:
        issues.append(f"ref_id(url) '{ref_id}'가 검색 결과에 없다")
    return CheckResult(passed=not issues, issues=issues)


_EVIDENCE_TEXT_FIELDS = ("quote", "title", "author_or_org", "venue", "date", "summary")


def field_check(finding: dict, perspective: str) -> CheckResult:
    """서술에 빠진 항목이 없는지 본다. 성숙도 관점이면 추정이라는 표시가 문장에 있어야 한다."""
    issues: list[str] = []
    try:
        model = Finding.model_validate(finding)
    except ValidationError as exc:
        return CheckResult(passed=False, issues=[f"스키마 불일치: {e['loc']}: {e['msg']}" for e in exc.errors()])

    if not model.claim.strip():
        issues.append("claim이 비어 있다")
    for i, ev in enumerate(model.evidence):
        for name in _EVIDENCE_TEXT_FIELDS:
            if not str(getattr(ev, name, "")).strip():
                issues.append(f"evidence[{i}].{name}이 비어 있다")
    if perspective == "trl":
        text = f"{model.claim} {model.conditions or ''}"
        if TRL_DISCLAIMER not in text:
            issues.append(f"TRL 서술에 '{TRL_DISCLAIMER}' 문구가 없다")
    return CheckResult(passed=not issues, issues=issues)


def support_judge(finding: dict) -> CheckResult:
    """인용 하나하나가 주장을 뒷받침하는지 모델에게 묻는다.

    인용이 실제로 있더라도 주장과 다른 대목일 수 있어 앞의 확인과 따로 둔다.
    """
    issues: list[str] = []
    for evidence in finding.get("evidence", []):
        ref_id = evidence.get("ref_id", "?")
        prompt = prompts.render("support_judge", claim=finding.get("claim", ""), quote=evidence.get("quote", ""))
        result = llm.judge(prompt, SupportJudgement)
        if result is None:
            issues.append(f"ref_id '{ref_id}': 근거 타당성 판정 응답 실패")
        elif not result.supported:
            issues.append(f"ref_id '{ref_id}': 인용이 주장을 뒷받침하지 않음 ({result.reason})")
    return CheckResult(passed=not issues, issues=issues)


def neutrality_issues(text: str) -> list[str]:
    """두 기술의 우열을 가리는 표현을 찾는다. 없으면 빈 목록."""
    if not text.strip():
        return []
    judgement = llm.judge(prompts.render("neutrality_judge", text=text), NeutralityJudgement)
    if judgement is None or judgement.passed:
        return []
    return [f"우열 표현: {sentence}" for sentence in judgement.issues]


def synthesis_code_checks(state: MainState) -> list[str]:
    """종합 결과를 코드로 확인한다.

    가리키는 서술이 실제로 있는지, 서로 다른 관점 둘 이상이 묶였는지, 관점들이 엇갈렸는데도
    충돌 항목이 비어 있지는 않은지 본다. 마지막 것은 종합이 이견을 지워 버린 경우를 잡는다.
    """
    synthesis = state.get("synthesis", {})
    known = all_finding_ids(state, EVAL_KEYS)
    issues: list[str] = []
    for kind in ("agreements", "conflicts"):
        for number, item in enumerate(synthesis.get(kind, []), start=1):
            ids = item.get("finding_ids", [])
            unknown = [fid for fid in ids if fid not in known]
            if unknown:
                issues.append(f"{kind}[{number}]: 존재하지 않는 finding_id {unknown}")
            perspectives = {known[fid] for fid in ids if fid in known}
            if len(perspectives) < 2:
                issues.append(f"{kind}[{number}]: 서로 다른 관점의 Finding이 2개 미만이다 ({sorted(perspectives)})")
    if not synthesis.get("agreements") and not synthesis.get("conflicts"):
        issues.append("종합 항목(agreements, conflicts)이 하나도 없다")
    with_dissent = [key for key in EVAL_KEYS if state.get(key, {}).get("dissent")]
    if len(with_dissent) >= 2 and not synthesis.get("conflicts"):
        issues.append(f"관점 {with_dissent}에 반대 방향 의견(dissent)이 있는데 conflicts가 비어 있다")
    return issues


def synthesis_check(state: MainState) -> MainState:
    """종합 결과를 검사하는 노드. 걸린 것이 있으면 종합을 다시 쓰게 한다."""
    attempt = state.get("synthesis_check", {}).get("attempt", 0) + 1
    if config.is_dry_run():
        return {"synthesis_check": {"passed": True, "issues": [], "attempt": attempt}}
    issues = synthesis_code_checks(state)
    synthesis = state.get("synthesis", {})
    statements = "\n".join(item["statement"] for kind in ("agreements", "conflicts") for item in synthesis.get(kind, []))
    issues += neutrality_issues(statements)
    return {"synthesis_check": CheckResult(passed=not issues, issues=issues, attempt=attempt).model_dump()}


SUMMARY_MIN, SUMMARY_MAX = 300, 900  # 반 쪽 분량을 넘지 않게 한다
NEUTRALITY_MAX_CHARS = 12000

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_CITATION = re.compile(r"\[(\d+)\]")
_REF_ENTRY = re.compile(r"^\[(\d+)\]", re.MULTILINE)


def top_headings(text: str) -> list[str]:
    """가장 얕은 수준의 제목만 문서에 나온 순서대로.

    문서가 #를 몇 개 쓰든 같은 기준으로 절 순서를 볼 수 있다.
    """
    found = _HEADING.findall(text)
    if not found:
        return []
    depth = min(len(marks) for marks, _ in found)
    return [title for marks, title in found if len(marks) == depth]


def summary_text(text: str) -> str:
    """요약 절의 본문. 다음 제목이 나오기 전까지를 가져온다."""
    match = re.search(r"^#{1,6}\s*SUMMARY.*?$\n(.*?)(?=^#{1,6}\s)", text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def split_reference(text: str) -> tuple[str, str]:
    """본문과 출처 절을 가른다. 인용과 목록을 견주려면 둘을 나눠야 한다."""
    match = re.search(r"^#{1,6}\s*REFERENCE.*$", text, re.MULTILINE | re.IGNORECASE)
    return (text[: match.start()], text[match.end():]) if match else (text, "")


def report_code_checks(text: str) -> list[str]:
    """보고서를 코드로 확인한다.

    요약이 맨 앞이고 출처가 맨 뒤인지, 요약 길이가 적당한지, 본문 인용과 출처 목록이 서로 맞는지,
    추정이라는 표시가 있는지 본다.
    """
    issues: list[str] = []
    headings = top_headings(text)
    if not headings or not headings[0].upper().startswith("SUMMARY"):
        issues.append("SUMMARY가 맨 앞 절이 아니다")
    if not headings or not headings[-1].upper().startswith("REFERENCE"):
        issues.append("REFERENCE가 맨 뒤 절이 아니다")
    length = len(summary_text(text))
    if length < SUMMARY_MIN:
        issues.append(f"SUMMARY가 너무 짧다 ({length}자, 최소 {SUMMARY_MIN}자)")
    elif length > SUMMARY_MAX:
        issues.append(f"SUMMARY가 너무 길다 ({length}자, 최대 {SUMMARY_MAX}자)")
    body, reference = split_reference(text)
    cited = set(_CITATION.findall(body))
    listed = set(_REF_ENTRY.findall(reference))
    if not cited:
        issues.append("본문에 [n] 형식의 인용이 없다")
    if cited - listed:
        issues.append(f"본문 인용이 REFERENCE에 없다: {sorted(cited - listed, key=int)}")
    if listed - cited:
        issues.append(f"REFERENCE에 본문에서 인용하지 않은 항목이 있다: {sorted(listed - cited, key=int)}")
    if TRL_DISCLAIMER not in text:
        issues.append(f"TRL 문구 '{TRL_DISCLAIMER}'가 없다")
    return issues


def report_check(state: MainState) -> MainState:
    """보고서를 검사하는 노드. 걸린 것이 있으면 보고서를 다시 쓰게 한다."""
    attempt = state.get("report_check", {}).get("attempt", 0) + 1
    if config.is_dry_run():
        return {"report_check": {"passed": True, "issues": [], "attempt": attempt}}
    text = state.get("final_report", "")
    issues = report_code_checks(text)
    body, _ = split_reference(text)
    issues += neutrality_issues(body[:NEUTRALITY_MAX_CHARS])
    return {"report_check": CheckResult(passed=not issues, issues=issues, attempt=attempt).model_dump()}
