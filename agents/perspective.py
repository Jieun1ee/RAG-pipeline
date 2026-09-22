"""관점 하나를 통째로 처리하는 노드를 만든다.

다섯 관점이 모두 같은 함수에서 나온다. 기준 목록과 검색 종류만 다르고 절차는 같다.

기준과 기술을 짝지어 할 일을 만들고, 짝마다 서브그래프를 돌려 서술을 모은다. 그다음 기준별로
수준을 판정하고, 한 기술에 강점과 한계가 모두 나왔는지 확인한다. 한쪽만 나왔다면 반대쪽을 찾지
못했다고 적어 둔다. 근거가 한쪽으로 쏠린 채 결론이 나가지 않게 하려는 것이다.

성숙도 관점만 판정 뒤에 한 단계가 더 있다. 단계별 근거에서 구간을 규칙으로 계산한다.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

import yaml

from agents import checks
from agents.subgraph import SearchFn, build_subgraph
from core import config, llm, prompts
from core.schemas import CriterionAssessment, LevelJudgement, PerspectiveResult, Retrieved, TRLEstimate, TRLResult
from core.state import MainState, load_fixture

log = logging.getLogger(__name__)

STATE_KEY: dict[str, str] = {
    "tech": "tech_research",
    "trl": "trl_eval",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}
# 답이 논문 안에 있는 관점과 논문 밖에 있는 관점을 나눈다.
# 도메인 적합성은 처리량·지연·메모리의 보고 수치와 그 측정 조건을 따지므로 논문 본문을 본다.
# 기술 성숙도는 상용 운용, 양산과 납품, 프레임워크 정식 지원처럼 논문에 실리지 않는 활동을 확인해야 해서 웹을 본다.
SOURCE: dict[str, str] = {"tech": "rag", "trl": "web", "market": "web", "stakeholder": "web", "domain": "rag"}
TRL_AXIS = "근거 수준"


def tech_summary(state: MainState, technology: str) -> str:
    """앞서 조사한 내용 중 그 기술에 대한 것만 모은다. 뒤 관점이 질의를 만들 때 배경으로 넣는다."""
    findings = state.get("tech_research", {}).get("findings", [])
    return "\n".join(f["claim"] for f in findings if f.get("technology") == technology)


def _tech_name(state: MainState, technology: str) -> str:
    """등록 정보에 적힌 정식 이름. 약칭만으로 웹을 뒤지면 엉뚱한 것이 걸리기 때문에 질의에 함께 넣는다."""
    for tech in state.get("selected_tech", []):
        if tech.get("technology") == technology:
            return tech.get("name") or technology
    return technology


def axes_for(perspective: str, spec: dict, item: dict) -> list[str]:
    """그 기준을 어떤 축으로 판정할지.

    축이 항목마다 다른 관점이 있어 항목을 먼저 보고, 없으면 관점 전체에 걸린 축을 쓴다.
    서술로만 정리하는 관점은 축이 없다.
    """
    if perspective == "stakeholder":
        return list(item.get("axes", []))
    if spec.get("level_axis"):
        return [spec["level_axis"]]
    return []


def build_tasks(perspective: str, state: MainState) -> list[dict]:
    """기준과 기술을 짝지어 할 일 목록을 만든다.

    설정에 개수 상한이 있으면 앞쪽 기준만 쓴다. 전부 돌리기 전에 흐름과 프롬프트를 확인할 때 쓴다.
    """
    cfg = config.get()
    spec = state["criteria"][perspective]
    items = list(spec["items"])
    limit = cfg["execution"].get("criteria_limit")
    if limit is not None:
        items = items[: int(limit)]
    technologies = list(cfg["execution"]["technologies"])
    source_type = "paper" if SOURCE[perspective] == "rag" else "web"
    return [
        {
            "perspective": perspective,
            "technology": technology,
            "tech_name": _tech_name(state, technology),
            "criterion": dict(item),
            "tech_summary": tech_summary(state, technology),
            "source_type": source_type,
            "levels": dict(spec.get("levels") or {}),
            "axes": axes_for(perspective, spec, item),
        }
        for item in items
        for technology in technologies
    ]


def _search_fn(perspective: str) -> SearchFn:
    """관점에 맞는 검색 함수. 외부 호출을 끈 모드에서는 샘플 결과를 돌려주는 함수를 준다."""
    source = SOURCE[perspective]
    if config.is_dry_run():
        fixture = f"retrieved_{source}"
        return lambda query, k: [Retrieved.model_validate(r) for r in load_fixture(fixture)][:k]
    if source == "rag":
        from rag.retriever import Retriever

        return Retriever().search
    from rag.web_search import WebSearch

    web = WebSearch()
    return lambda query, k: web.search(query, int(config.get()["retrieval"]["web_max_results"]))


def _subgraph(perspective: str):
    """관점에 맞는 검색과 인용 확인 방식을 끼운 서브그래프."""
    check_citation = checks.rag_check if SOURCE[perspective] == "rag" else checks.web_check
    return build_subgraph(
        search=_search_fn(perspective),
        check_citation=check_citation,
        role_prompt=f"perspective/{perspective}",
        perspective=perspective,
    )


def _format_findings(findings: list[dict]) -> str:
    """판정 프롬프트에 넣을 서술 목록. 어느 서술을 근거로 삼았는지 되짚을 수 있게 id를 붙인다."""
    lines = []
    for f in findings:
        summaries = "; ".join(e.get("summary", "") for e in f.get("evidence", []))
        lines.append(f"- {f['id']} [{f['polarity']}, confidence={f['confidence']}] {f['claim']} (근거: {summaries})")
    return "\n".join(lines) or "(없음)"


def assess(perspective: str, spec: dict, item: dict, technology: str, findings: list[dict]) -> dict:
    """기준 하나와 기술 하나에 대한 판정을 만든다.

    근거가 없으면 모델을 부르지 않고 근거 부족으로 끝낸다. 조사 관점은 판정 없이 모은 서술을 그대로 적는다.
    모델이 정의에 없는 수준을 답하면 버린다. 성숙도 관점만은 빈칸 대신 근거 없음으로 채우는데,
    뒤에서 구간을 계산할 때 단계가 비어 있으면 안 되기 때문이다.
    """
    axes = axes_for(perspective, spec, item)
    base = {"id": f"{item['id']}-{technology}", "criterion": item["id"], "technology": technology}
    finding_ids = [f["id"] for f in findings]

    if not findings:
        levels = {axis: "none" for axis in axes} if perspective == "trl" else {}
        return CriterionAssessment(
            **base, status="insufficient_evidence", levels=levels, rationale="검증을 통과한 근거를 찾지 못했다", finding_ids=[]
        ).model_dump()

    if perspective == "tech":
        rationale = " / ".join(f["claim"] for f in findings)[:800]
        return CriterionAssessment(**base, status="assessed", levels={}, rationale=rationale, finding_ids=finding_ids).model_dump()

    allowed = spec.get("levels") or {}
    prompt = prompts.render(
        "level_judge",
        technology=technology,
        criterion=yaml.safe_dump(item, allow_unicode=True, sort_keys=False).strip(),
        axes=", ".join(axes) or "(없음)",
        levels=yaml.safe_dump(allowed, allow_unicode=True, sort_keys=False).strip() if allowed else "(없음)",
        findings=_format_findings(findings),
    )
    judgement = llm.generate(prompt, LevelJudgement)
    if judgement is None:
        levels = {axis: "none" for axis in axes} if perspective == "trl" else {}
        return CriterionAssessment(
            **base, status="insufficient_evidence", levels=levels, rationale="수준 판정 응답 실패", finding_ids=finding_ids
        ).model_dump()

    levels: dict[str, str] = {}
    for entry in judgement.levels:
        if entry.axis not in axes:
            continue
        if allowed and entry.level not in allowed:
            if perspective == "trl":
                levels[entry.axis] = "none"
            continue
        levels[entry.axis] = entry.level
    if perspective == "trl":
        for axis in axes:
            levels.setdefault(axis, "none")
    return CriterionAssessment(
        **base, status=judgement.status, levels=levels, rationale=judgement.rationale, finding_ids=finding_ids
    ).model_dump()


def dissent_of(findings: list[dict]) -> list[dict]:
    """같은 기준에서 반대 방향으로 나온 서술을 따로 모은다.

    한 방향으로 정리해 버리면 엇갈린 근거가 보고서에서 사라진다. 수가 적은 쪽을 남겨,
    종합과 검사가 그 대목을 볼 수 있게 한다.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in findings:
        groups[(f["criterion"], f["technology"])].append(f)
    dissent: list[dict] = []
    for group in groups.values():
        strengths = [f for f in group if f["polarity"] == "strength"]
        limitations = [f for f in group if f["polarity"] == "limitation"]
        if strengths and limitations:
            dissent.extend(limitations if len(limitations) <= len(strengths) else strengths)
    return dissent


def balance_gaps(findings: list[dict], technologies: list[str]) -> list[str]:
    """기술마다 강점과 한계가 모두 나왔는지 본다. 한쪽이 비면 그 사실을 남긴다.

    한쪽만 찾아 놓고 결론을 내면 검색이 치우친 것인지 정말 그런 것인지 구분할 수 없다.
    """
    gaps = []
    for technology in technologies:
        polarities = {f["polarity"] for f in findings if f["technology"] == technology}
        if "strength" not in polarities:
            gaps.append(f"{technology}: 강점(strength) 방향의 검증된 근거가 없다")
        if "limitation" not in polarities:
            gaps.append(f"{technology}: 한계(limitation) 방향의 검증된 근거가 없다")
    return gaps


def trl_estimate(technology: str, assessments: list[dict]) -> dict:
    """단계별 근거를 모아 성숙도 구간을 계산한다.

    아래쪽은 1단계부터 직접 근거가 끊기지 않고 이어진 데까지로 본다. 중간이 비면 거기서 멈춘다.
    위쪽은 간접 근거라도 있는 가장 높은 단계다. 직접 근거가 하나도 없으면 구간을 내지 않고 판단을 미룬다.
    공개된 자료만 보고 매기는 값이므로 그 사실을 결과에 함께 남긴다.
    """
    stage: dict[int, str] = {}
    for a in assessments:
        if a["technology"] != technology or not a["criterion"].startswith("TRL-"):
            continue
        stage[int(a["criterion"].split("-")[1])] = a.get("levels", {}).get(TRL_AXIS, "none")
    stage_evidence = {k: stage.get(k, "none") for k in range(1, 10)}
    evaluated = sorted(stage)  # 개수를 줄여 돌리면 일부 단계는 아예 보지 않는다
    direct = [k for k, v in stage_evidence.items() if v == "direct"]
    indirect_or_better = [k for k, v in stage_evidence.items() if v in ("direct", "indirect")]

    if not direct:
        return TRLEstimate(
            technology=technology,
            status="withheld",
            stage_evidence=stage_evidence,
            inference_note=f"직접 근거가 있는 단계가 없어 판단을 유보한다. 평가한 단계: {evaluated or '없음'}",
            confidence="low",
        ).model_dump()

    trl_min = 0
    for k in range(1, 10):
        if stage_evidence[k] != "direct":
            break
        trl_min = k
    if trl_min == 0:  # 1단계가 비어 있으면 이어진 구간이 없으므로 직접 근거가 나온 첫 단계를 쓴다
        trl_min = min(direct)
    trl_max = max(max(indirect_or_better), trl_min)
    confidence = "high" if len(direct) >= 3 else "medium" if len(direct) >= 2 or len(indirect_or_better) > len(direct) else "low"
    note = (
        f"1~{trl_min} 단계는 직접 근거, {trl_max} 단계까지 간접 근거가 있다. "
        f"평가한 단계: {evaluated}. 평가하지 않은 단계는 none으로 두었다."
    )
    return TRLEstimate(
        technology=technology,
        status="estimated",
        trl_min=trl_min,
        trl_max=trl_max,
        stage_evidence=stage_evidence,
        inference_note=note,
        confidence=confidence,
    ).model_dump()


def make_node(perspective: str) -> Callable[[MainState], MainState]:
    """관점 하나를 처리하는 노드를 만든다. 노드 이름은 결과를 쓸 상태 키와 같게 맞춘다."""
    key = STATE_KEY[perspective]

    def node(state: MainState) -> MainState:
        tasks = build_tasks(perspective, state)
        subgraph = _subgraph(perspective)
        findings: list[dict] = []
        gaps: list[str] = []
        for task in tasks:
            out = subgraph.invoke({"task": task})
            findings.extend(out.get("findings", []))
            gaps.extend(out.get("gaps", []))
        log.info("%s: task %d개, findings %d개, gaps %d개", key, len(tasks), len(findings), len(gaps))

        # 외부 호출을 끈 모드에서는 여기까지 돌려 연결만 확인하고 결과는 샘플로 대신한다
        if config.is_dry_run():
            return {key: load_fixture(key)}

        spec = state["criteria"][perspective]
        technologies = list(config.get()["execution"]["technologies"])
        by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for f in findings:
            by_group[(f["criterion"], f["technology"])].append(f)
        # 근거를 못 찾은 기준도 판정 목록에 남긴다. 다루지 않은 것과 근거가 없던 것은 다르다.
        seen_items = {t["criterion"]["id"]: t["criterion"] for t in tasks}
        assessments = [
            assess(perspective, spec, item, technology, by_group.get((item_id, technology), []))
            for item_id, item in seen_items.items()
            for technology in technologies
        ]
        gaps.extend(balance_gaps(findings, technologies))

        data = {
            "perspective": perspective,
            "assessments": assessments,
            "findings": findings,
            "dissent": dissent_of(findings),
            "gaps": gaps,
        }
        if perspective == "trl":
            data["estimates"] = [trl_estimate(t, assessments) for t in technologies]
            return {key: TRLResult.model_validate(data).model_dump(mode="json")}
        return {key: PerspectiveResult.model_validate(data).model_dump(mode="json")}

    node.__name__ = key
    return node
