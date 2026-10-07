"""서술과 보고서가 규칙을 지켰는지 확인한다.

검사는 두 갈래다. 인용이 검색 결과에 실제로 있는지, 빠진 항목이 없는지는 코드로 본다. 값이 싸고
결과가 늘 같다. 인용이 주장을 뒷받침하는지, 두 기술의 우열을 가리는 표현이 있는지는 글의 뜻을
읽어야 하므로 모델에게 맡긴다.

앞쪽 함수들은 기준 하나를 처리하는 도중에 불리고, synthesis_check와 report_check는 그래프의 노드로 붙는다.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from agents.synthesis import EVAL_KEYS, all_finding_ids
from core import config, llm, prompts
from core.schemas import CheckResult, Finding, NeutralityJudgement, ReportQualityJudgement, SupportJudgement
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

QUALITY_LABEL = {
    "groundedness": "Groundedness",
    "neutrality": "중립성",
    "bias_control": "편향 통제",
    "perspective_coverage": "관점 커버리지",
}
PERSPECTIVE_RESULT = {
    "trl": "trl_eval",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}
PERSPECTIVE_SECTION = {
    "trl": "4.1",
    "market": "4.2",
    "stakeholder": "4.3",
    "domain": "4.4",
}
_PREFERENCE = re.compile(r"(?:더\s+(?:낫|우수|효율)|우월|열등|권장|선택해야|채택해야|앞선다)")

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


def _section(text: str, number: str) -> str:
    """번호로 시작하는 절의 본문. 다음 같은 수준의 번호 절이나 문서 끝까지를 가져온다."""
    match = re.search(
        rf"^#{{1,6}}\s*{re.escape(number)}(?:\s|\.).*?$\n(.*?)(?=^#{{1,6}}\s*\d+(?:\.\d+)?(?:\s|\.).*?$|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _technologies(state: MainState) -> list[str]:
    configured = list(config.get()["execution"]["technologies"])
    selected = [t.get("technology") for t in state.get("selected_tech", []) if t.get("technology")]
    return selected or configured


def _findings(state: MainState, perspective: str, technology: str) -> list[dict]:
    result = state.get(PERSPECTIVE_RESULT[perspective], {})
    return [f for f in result.get("findings", []) if f.get("technology") == technology]


def _quality_context(state: MainState) -> str:
    """Judge가 보고서의 주장을 검증된 결과와 대조할 수 있도록 필요한 값만 줄여 전달한다."""
    data = {}
    for perspective, key in PERSPECTIVE_RESULT.items():
        result = state.get(key, {})
        data[perspective] = {
            "assessments": [
                {
                    "criterion": a.get("criterion"),
                    "technology": a.get("technology"),
                    "status": a.get("status"),
                    "rationale": a.get("rationale"),
                }
                for a in result.get("assessments", [])
            ],
            "findings": [
                {
                    "id": f.get("id"),
                    "technology": f.get("technology"),
                    "criterion": f.get("criterion"),
                    "claim": f.get("claim"),
                    "polarity": f.get("polarity"),
                    "evidence": [
                        {
                            "ref_id": ev.get("ref_id"),
                            "source_nature": ev.get("source_nature"),
                            "is_self_reported": ev.get("is_self_reported"),
                            "summary": ev.get("summary"),
                        }
                        for ev in f.get("evidence", [])
                    ],
                }
                for f in result.get("findings", []) + result.get("dissent", [])
            ],
            "gaps": result.get("gaps", []),
        }
    return json.dumps(data, ensure_ascii=False, indent=2)


def report_quality_rules(state: MainState, text: str) -> tuple[dict[str, list[str]], list[dict]]:
    """네 품질 항목 중 코드로 확정할 수 있는 것과 추가 조사가 필요한 빈 곳을 찾는다."""
    issues: dict[str, list[str]] = {name: [] for name in QUALITY_LABEL}
    requests: list[dict] = []
    body, reference = split_reference(text)

    # Groundedness: 보고서 번호가 실제 State 근거로 만든 참고문헌 번호와 연결되어야 한다.
    from agents.report import build_references  # 순환 import를 피하려 런타임에만 가져온다

    _, entries = build_references(state)
    cited = set(_CITATION.findall(body))
    listed = set(_REF_ENTRY.findall(reference))
    valid = {str(number) for number in entries}
    if not cited:
        issues["groundedness"].append("본문에 검증된 출처를 가리키는 [n] 인용이 없다")
    if cited - listed:
        issues["groundedness"].append(f"본문 인용이 REFERENCE에 없다: {sorted(cited - listed, key=int)}")
    if listed - cited:
        issues["groundedness"].append(f"REFERENCE에 본문에서 사용하지 않은 항목이 있다: {sorted(listed - cited, key=int)}")
    if cited - valid:
        issues["groundedness"].append(f"검증된 Evidence에서 만들지 않은 인용 번호가 있다: {sorted(cited - valid, key=int)}")

    # 중립성: 명백한 추천·우열 표현은 코드가 먼저 잡고, 문맥상 편향은 Judge가 본다.
    preference = sorted(set(_PREFERENCE.findall(body)))
    if preference:
        issues["neutrality"].append(f"추천·우열 표현 후보가 있다: {preference}")

    max_quality = int(config.get()["orchestration"].get("max_quality_replans", 1))
    may_replan = state.get("quality_replan_count", 0) < max_quality

    # 편향 통제: State 자체가 한 방향이거나 자체 발표뿐이면 보고서만 고쳐서는 해소할 수 없다.
    for perspective in PERSPECTIVE_RESULT:
        for technology in _technologies(state):
            findings = _findings(state, perspective, technology)
            if not findings:
                continue  # 관점 결과가 통째로 없는 경우는 아래 커버리지에서 처리한다.
            polarities = {f.get("polarity") for f in findings}
            missing = [p for p in ("strength", "limitation") if p not in polarities]
            evidence = [ev for f in findings for ev in f.get("evidence", [])]
            only_self_reported = bool(evidence) and all(ev.get("is_self_reported") for ev in evidence)
            reasons = []
            if missing:
                reasons.append(f"{technology}의 {','.join(missing)} 방향 근거가 없다")
            if only_self_reported:
                reasons.append(f"{technology}의 근거가 모두 개발사 자체 발표다")
            if reasons and may_replan:
                reason = f"{perspective}: " + "; ".join(reasons)
                issues["bias_control"].append(reason)
                requests.append({
                    "criterion": "bias_control",
                    "perspective": perspective,
                    "technology": technology,
                    "reason": reason,
                    "missing": missing,
                })

    # 관점 커버리지: 절 누락은 재작성, 결과 상태까지 비어 있으면 해당 관점만 다시 조사한다.
    for perspective, key in PERSPECTIVE_RESULT.items():
        section = _section(body, PERSPECTIVE_SECTION[perspective])
        if not section:
            issues["perspective_coverage"].append(f"{PERSPECTIVE_SECTION[perspective]} {perspective} 관점 절이 없거나 비어 있다")
        result = state.get(key, {})
        has_result = bool(result.get("findings") or result.get("assessments") or result.get("gaps"))
        if not has_result and may_replan:
            for technology in _technologies(state):
                requests.append({
                    "criterion": "perspective_coverage",
                    "perspective": perspective,
                    "technology": technology,
                    "reason": f"{perspective} 관점 평가 결과와 근거 부족 기록이 모두 없다",
                    "missing": [],
                })

    unique = {
        (r["criterion"], r["perspective"], r["technology"]): r
        for r in requests
    }
    return issues, list(unique.values())


def _judge_request(state: MainState, criterion: str, verdict: dict) -> dict | None:
    """Judge가 근거 부족으로 본 항목이 실제 State에도 비어 있을 때만 재계획 요청으로 바꾼다."""
    perspective = verdict.get("perspective")
    technology = verdict.get("technology")
    if not verdict.get("missing_evidence") or perspective not in PERSPECTIVE_RESULT or technology not in _technologies(state):
        return None
    if _findings(state, perspective, technology):
        return None
    return {
        "criterion": criterion,
        "perspective": perspective,
        "technology": technology,
        "reason": verdict.get("reason", "검증된 근거가 없다"),
        "missing": [],
    }


def report_check(state: MainState) -> MainState:
    """형식과 네 품질 항목을 Hybrid로 검사하고 재작성 또는 재계획 경로를 정한다."""
    attempt = state.get("report_check", {}).get("attempt", 0) + 1
    if config.is_dry_run():
        criteria = {name: {"passed": True, "rule_passed": True, "judge_passed": True, "reason": "dry-run"} for name in QUALITY_LABEL}
        return {"report_check": {"passed": True, "issues": [], "attempt": attempt, "route": "done", "criteria": criteria}}

    text = state.get("final_report", "")
    format_issues = report_code_checks(text)
    rule_issues, requests = report_quality_rules(state, text)
    judgement = llm.judge(
        prompts.render("report_quality_judge", report=text, evidence_context=_quality_context(state)),
        ReportQualityJudgement,
    )

    criteria = {}
    issues = [f"[형식] {issue}" for issue in format_issues]
    if judgement is None:
        issues.append("[품질 Judge] 구조화 판정 응답 실패")
        for name in QUALITY_LABEL:
            criteria[name] = {
                "passed": False,
                "rule_passed": not rule_issues[name],
                "judge_passed": False,
                "reason": "품질 Judge 응답 실패",
            }
    else:
        judged = judgement.model_dump(mode="json")
        for name, label in QUALITY_LABEL.items():
            verdict = judged[name]
            rule_passed = not rule_issues[name]
            passed = rule_passed and verdict["passed"]
            criteria[name] = {
                "passed": passed,
                "rule_passed": rule_passed,
                "judge_passed": verdict["passed"],
                "reason": verdict["reason"],
            }
            issues += [f"[{label}] {issue}" for issue in rule_issues[name]]
            if not verdict["passed"]:
                issues.append(f"[{label}] {verdict['reason']}")
                request = _judge_request(state, name, verdict)
                if request:
                    requests.append(request)

    max_quality = int(config.get()["orchestration"].get("max_quality_replans", 1))
    can_replan = state.get("quality_replan_count", 0) < max_quality
    unique_requests = {
        (r["criterion"], r["perspective"], r["technology"]): r
        for r in requests
    }
    replan_requests = list(unique_requests.values()) if can_replan else []
    passed = not format_issues and all(item["passed"] for item in criteria.values())
    route = "done" if passed else "replan" if replan_requests else "rewrite"
    return {
        "report_check": {
            "passed": passed,
            "issues": list(dict.fromkeys(issues)),
            "attempt": attempt,
            "route": route,
            "criteria": criteria,
            "replan_requests": replan_requests,
        }
    }
