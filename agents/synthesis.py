"""관점별 결과를 하나로 묶는다.

여러 관점이 같은 방향을 가리킨 것과 엇갈린 것을 나누어 정리한다. 엇갈린 대목을 남기는 것이 중요하다.
한쪽으로 정리해 버리면 근거가 모자라서 그런 것인지 실제로 다른 것인지 읽는 사람이 알 수 없다.

모델이 묶어 온 것을 그대로 쓰지는 않는다. 없는 서술을 가리키거나 한 관점 안에서만 묶은 항목은 버린다.
검사에서 지적이 나왔다면 다시 쓸 때 그 내용을 프롬프트에 붙인다.
"""

from __future__ import annotations

import logging

from core import config, llm, prompts
from core.schemas import SynthesisItem, SynthesisResult
from core.state import MainState, load_fixture

log = logging.getLogger(__name__)

KEY = "synthesis"
EVAL_KEYS = ("trl_eval", "market_eval", "stakeholder_eval", "domain_eval")
LABEL = {
    "tech_research": "기술 조사",
    "trl_eval": "기술 성숙도",
    "market_eval": "시장성",
    "stakeholder_eval": "이해관계자",
    "domain_eval": "도메인 적합성",
}


def all_finding_ids(state: MainState, keys: tuple[str, ...] = ("tech_research", *EVAL_KEYS)) -> dict[str, str]:
    """서술 id로 그 서술이 나온 관점을 찾을 수 있는 표. 반대 방향 의견도 함께 넣는다."""
    ids: dict[str, str] = {}
    for key in keys:
        result = state.get(key, {})
        for f in result.get("findings", []) + result.get("dissent", []):
            ids[f["id"]] = key
    return ids


def summarize_criteria(state: MainState) -> str:
    """프롬프트에 넣을 기준 목록. 어떤 잣대로 본 결과인지 함께 보여 주려는 것이다."""
    lines = []
    for perspective, spec in state.get("criteria", {}).items():
        if not isinstance(spec, dict) or "items" not in spec:
            continue
        lines.append(f"## {spec.get('name', perspective)} ({perspective}, result_type={spec.get('result_type')})")
        lines += [f"- {item['id']} {item.get('name', '')}" for item in spec["items"]]
    return "\n".join(lines)


def summarize_result(key: str, result: dict) -> str:
    """관점 하나의 결과를 프롬프트에 넣을 크기로 줄인다.

    인용 원문은 빼고 판정과 서술만 남긴다. 종합 단계에서 필요한 것은 무엇이 나왔는지이지 원문이 아니다.
    """
    lines = [f"## {LABEL.get(key, key)} ({key})"]
    for a in result.get("assessments", []):
        levels = ", ".join(f"{k}={v}" for k, v in a.get("levels", {}).items())
        lines.append(f"- 판정 {a['id']}: {a['status']}{' (' + levels + ')' if levels else ''} — {a.get('rationale', '')[:300]}")
    for e in result.get("estimates", []):
        lines.append(f"- TRL 추정 {e['technology']}: {e['status']} {e.get('trl_min')}~{e.get('trl_max')} ({e.get('disclaimer', '')})")
    for f in result.get("findings", []):
        lines.append(f"- Finding {f['id']} [{f['technology']}, {f['polarity']}] {f['claim']}")
    if result.get("dissent"):
        lines.append("- 반대 방향 의견(dissent): " + ", ".join(f["id"] for f in result["dissent"]))
    for gap in result.get("gaps", []):
        lines.append(f"- 미확인: {gap}")
    return "\n".join(lines)


def node(state: MainState) -> MainState:
    """네 관점의 결과를 묶어 합의와 충돌로 정리한다."""
    if config.is_dry_run():
        return {KEY: load_fixture(KEY)}

    results = "\n\n".join(summarize_result(key, state.get(key, {})) for key in EVAL_KEYS)
    feedback = "\n".join(state.get("synthesis_check", {}).get("issues", [])) or "(없음)"
    prompt = prompts.render("synthesis", criteria=summarize_criteria(state), results=results, feedback=feedback)
    draft = llm.generate(prompt, SynthesisResult)

    known = all_finding_ids(state, EVAL_KEYS)

    def clean(items: list[SynthesisItem]) -> list[SynthesisItem]:
        """실제로 있는 서술만 남기고, 둘 이상이 남은 항목만 통과시킨다."""
        kept = []
        for item in items:
            ids = [fid for fid in dict.fromkeys(item.finding_ids) if fid in known]
            if len(ids) >= 2:
                kept.append(SynthesisItem(statement=item.statement, finding_ids=ids))
        return kept

    if draft is None:
        log.warning("synthesis: 구조화 출력 실패, 빈 결과")
        result = SynthesisResult(agreements=[], conflicts=[])
    else:
        result = SynthesisResult(agreements=clean(draft.agreements), conflicts=clean(draft.conflicts))
    return {KEY: result.model_dump()}
