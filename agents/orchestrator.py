"""무엇을 누구에게 맡길지 정하는 중앙 노드.

계획은 회차 단위로 세운다. 한 회차의 task 목록을 plan에 올리고, 보낼 task만 task_status를 pending으로
적는다. dispatch는 pending인 task를 worker에게 Send로 나눠 보낸다. worker가 끝나고 collect가 결과를
묶으면 다시 여기로 돌아와 다음 회차를 정한다.

    ① 기술 조사      코드가 기술 조사 기준 × 기술을 모두 보낸다. 뒤 계획의 배경 자료라 LLM이 고르지 않는다
    ② 관점 평가 계획  LLM이 task마다 출처와 검색 초점을 정한다. 빠진 기준 × 기술은 코드가 채운다
    ③ 재계획         빈 곳(한쪽 근거만 나온 관점, 근거를 끝내 못 찾은 task)만 골라 추가 task를 보낸다
    ④ 품질 보완    보고서 검사가 추가 근거를 요청하면 해당 관점·기술만 다시 보낸다
    ⑤ 종료           더 보낼 것이 없거나 상한에 닿으면 종합으로 넘긴다

어느 회차든 실행이 예외로 끝난(error) task가 있으면, 다음 회차로 가기 전에 그것부터 다시 보낸다.
결정마다 사유를 결정 로그에 남긴다. 사유는 State에 넣지 않는다.

외부 호출을 끈 모드와 LLM 응답이 실패한 경우에는 코드가 만든 기본 계획을 쓴다. 흐름은 같다.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from agents.perspective import STATE_KEY
from core import config, llm, prompts
from core.schemas import Plan, PlannedTask, TaskSpec, task_id_of
from core.state import MainState
from core.tracing import log_decision

NODE = "orchestrator"
DONE = "done"  # dispatch가 보낼 task가 없을 때 돌려주는 경로 이름
EVAL_PERSPECTIVES = ("trl", "market", "stakeholder", "domain")
# 관점별 기본 출처. 답이 논문 안에 있는 관점은 paper, 논문 밖 활동을 봐야 하는 관점은 web이다.
# 도메인 적합성은 처리량·지연·메모리의 보고 수치와 그 측정 조건을 따지므로 논문 본문을 본다.
# 기술 성숙도는 상용 운용, 양산과 납품, 프레임워크 정식 지원처럼 논문에 실리지 않는 활동을 확인해야 해서 웹을 본다.
# 계획 단계에서 LLM이 기준별로 바꿀 수 있다.
DEFAULT_SOURCE = {"tech": "paper", "trl": "web", "market": "web", "stakeholder": "web", "domain": "paper"}
SUMMARY_MAX = 1500  # 계획 프롬프트에 넣을 기술 조사 요약의 기술별 길이 상한


# --- 계획 재료 ---------------------------------------------------------------------


def _technologies() -> list[str]:
    return list(config.get()["execution"]["technologies"])


def _items(state: MainState, perspective: str) -> list[dict]:
    """관점의 기준 목록. 설정에 개수 상한이 있으면 앞쪽만 쓴다."""
    items = list(state["criteria"][perspective]["items"])
    limit = config.get()["execution"].get("criteria_limit")
    return items[: int(limit)] if limit is not None else items


def _tech_name(state: MainState, technology: str) -> str:
    for tech in state.get("selected_tech", []):
        if tech.get("technology") == technology:
            return tech.get("name") or technology
    return technology


def _tech_summary(state: MainState, technology: str) -> str:
    findings = state.get("tech_research", {}).get("findings", [])
    return "\n".join(f["claim"] for f in findings if f.get("technology") == technology)


def _spec(state: MainState, perspective: str, item: dict, technology: str, *, source: str, focus: str | None, round_: int) -> dict:
    """계획 한 줄에 코드가 아는 값을 채워 worker가 받을 일감으로 만든다."""
    return TaskSpec(
        task_id=task_id_of(perspective, item["id"], technology, round_),
        perspective=perspective,
        technology=technology,
        tech_name=_tech_name(state, technology),
        criterion=dict(item),
        source_type=source,
        focus=focus,
        tech_summary="" if perspective == "tech" else _tech_summary(state, technology),
        levels=dict(state["criteria"][perspective].get("levels") or {}),
        round=round_,
    ).model_dump()


def _pending(specs: list[dict]) -> dict[str, dict]:
    """새 회차에 보낼 task의 상태. 처음 보내는 것이라 attempt는 1이다."""
    return {s["task_id"]: {"status": "pending", "attempt": 1} for s in specs}


# --- ① 기술 조사 -------------------------------------------------------------------


def tech_round(state: MainState) -> list[dict]:
    return [
        _spec(state, "tech", item, technology, source=DEFAULT_SOURCE["tech"], focus=None, round_=0)
        for item in _items(state, "tech")
        for technology in _technologies()
    ]


# --- ② 관점 평가 계획 ---------------------------------------------------------------


def required_pairs(state: MainState) -> dict[tuple[str, str, str], dict]:
    """계획에 반드시 들어가야 하는 (관점, 기준 id, 기술)과 그 기준 원문."""
    return {
        (perspective, item["id"], technology): item
        for perspective in EVAL_PERSPECTIVES
        for item in _items(state, perspective)
        for technology in _technologies()
    }


def default_plan(state: MainState) -> Plan:
    """모든 기준 × 기술에 기본 출처와 평가 질문을 그대로 쓴 계획. LLM을 쓰지 못할 때 대신한다."""
    return Plan(
        tasks=[
            PlannedTask(
                perspective=perspective,
                criterion_id=criterion_id,
                technology=technology,
                source_type=DEFAULT_SOURCE[perspective],
                focus=item.get("question", ""),
            )
            for (perspective, criterion_id, technology), item in required_pairs(state).items()
        ],
        rationale="기본 계획: 관점별 기본 출처와 평가 질문을 그대로 쓴다",
    )


def cover(state: MainState, plan: Plan) -> tuple[list[dict], dict[str, list[str]]]:
    """LLM 계획을 검사해 빠짐없는 일감 목록으로 만든다.

    목록에 없는 짝은 버리고, 같은 짝이 두 번 오면 앞의 것만 쓰고, 빠진 짝은 기본값으로 채운다.
    두 기술에 같은 기준을 모두 적용한다는 원칙을 LLM이 아니라 코드가 지키게 하려는 것이다.
    무엇을 고쳤는지 돌려줘 결정 로그에 남긴다.
    """
    required = required_pairs(state)
    chosen: dict[tuple[str, str, str], PlannedTask] = {}
    fixes: dict[str, list[str]] = {"unknown": [], "duplicate": [], "filled": []}
    for task in plan.tasks:
        key = (task.perspective, task.criterion_id, task.technology)
        if key not in required:
            fixes["unknown"].append("-".join(key))
        elif key in chosen:
            fixes["duplicate"].append("-".join(key))
        else:
            chosen[key] = task
    specs = []
    for key, item in required.items():
        task = chosen.get(key)
        if task is None:
            fixes["filled"].append("-".join(key))
        perspective, _, technology = key
        specs.append(
            _spec(
                state, perspective, item, technology,
                source=task.source_type if task else DEFAULT_SOURCE[perspective],
                focus=(task.focus.strip() if task else "") or item.get("question", ""),
                round_=0,
            )
        )
    return specs, fixes


def _format_technologies(state: MainState) -> str:
    techs = [t for t in state.get("selected_tech", []) if t.get("technology") in _technologies()]
    return "\n".join(f"- {t['technology']} ({t.get('camp', '')}): {t.get('name', '')}" for t in techs)


def _format_criteria(state: MainState) -> str:
    return "\n".join(
        f"- [{perspective}] {item['id']} {item.get('name', '')}: {item.get('question', '')} (기본 출처: {DEFAULT_SOURCE[perspective]})"
        for perspective in EVAL_PERSPECTIVES
        for item in _items(state, perspective)
    )


def _format_summary(state: MainState) -> str:
    blocks = [f"### {t}\n{_tech_summary(state, t)[:SUMMARY_MAX] or '(없음)'}" for t in _technologies()]
    return "\n\n".join(blocks)


def _domain(state: MainState) -> str:
    return state.get("target_domain", "")


def eval_round(state: MainState) -> tuple[list[dict], str, dict[str, list[str]]]:
    """관점 평가 계획. 일감 목록, 계획 사유, 코드가 고친 내역을 돌려준다."""
    plan = None
    if not config.is_dry_run():
        prompt = prompts.render(
            "orchestrator_plan",
            domain=_domain(state),
            technologies=_format_technologies(state),
            tech_summary=_format_summary(state),
            criteria=_format_criteria(state),
            task_count=len(required_pairs(state)),
        )
        plan = llm.generate(prompt, Plan)
    if plan is None:
        plan = default_plan(state)
    specs, fixes = cover(state, plan)
    return specs, plan.rationale, fixes


# --- ③ 재계획 ---------------------------------------------------------------------


def find_gaps(state: MainState) -> list[dict]:
    """재계획 대상이 되는 빈 곳.

    balance는 관점 하나에서 한 기술의 강점이나 한계 근거가 하나도 없는 경우다. 관점 단위로 보므로
    그 관점의 어느 기준으로 메울지는 재계획에서 정한다. gave_up은 직전 회차에서 근거를 끝내 찾지 못한
    task다. balance를 앞에 두는데, 한쪽 근거만 남으면 보고서가 한 방향으로 기울기 때문이다.
    """
    gaps: list[dict] = []
    for perspective in EVAL_PERSPECTIVES:
        findings = state.get(STATE_KEY[perspective], {}).get("findings", [])
        for technology in _technologies():
            polarities = {f["polarity"] for f in findings if f.get("technology") == technology}
            missing = [p for p in ("strength", "limitation") if p not in polarities]
            if missing:
                gaps.append({"kind": "balance", "perspective": perspective, "technology": technology, "missing": missing})
    status = state.get("task_status", {})
    for spec in state.get("plan", []):
        if spec["perspective"] in EVAL_PERSPECTIVES and status.get(spec["task_id"], {}).get("status") == "gave_up":
            gaps.append({"kind": "gave_up", "perspective": spec["perspective"], "technology": spec["technology"], "spec": spec})
    return gaps


def quality_gaps(state: MainState) -> list[dict]:
    """보고서 품질 검사에서 실제 추가 조사가 필요하다고 확정한 관점·기술."""
    gaps = []
    for request in state.get("report_check", {}).get("replan_requests", []):
        perspective = request.get("perspective")
        technology = request.get("technology")
        if perspective not in EVAL_PERSPECTIVES or technology not in _technologies():
            continue
        gaps.append({
            "kind": "quality",
            "perspective": perspective,
            "technology": technology,
            "reason": request.get("reason", "보고서 품질을 충족할 검증 근거가 부족하다"),
            "missing": list(request.get("missing") or []),
        })
    return gaps


def _previous(state: MainState, perspective: str, criterion_id: str, technology: str) -> dict | None:
    """직전 회차에서 같은 짝을 어떤 출처와 초점으로 보냈는지."""
    for spec in state.get("plan", []):
        if (spec["perspective"], spec["criterion"]["id"], spec["technology"]) == (perspective, criterion_id, technology):
            return spec
    return None


def _format_gaps(state: MainState, gaps: list[dict]) -> str:
    lines = []
    for gap in gaps:
        perspective, technology = gap["perspective"], gap["technology"]
        if gap["kind"] == "balance":
            candidates = ", ".join(item["id"] for item in _items(state, perspective))
            missing = "와 ".join({"strength": "강점", "limitation": "한계"}[m] for m in gap["missing"])
            lines.append(f"- balance | {perspective} | 기준 후보: {candidates} | {technology} | 사유: {missing} 방향의 검증된 근거가 없다")
        elif gap["kind"] == "gave_up":
            spec = gap["spec"]
            lines.append(
                f"- gave_up | {perspective} | {spec['criterion']['id']} | {technology} | "
                f"이전: {spec['source_type']}, \"{spec.get('focus') or ''}\" | 사유: 재시도 상한까지 검증된 근거를 찾지 못했다"
            )
        else:
            candidates = ", ".join(item["id"] for item in _items(state, perspective))
            lines.append(
                f"- report_quality | {perspective} | 기준 후보: {candidates} | {technology} | "
                f"사유: {gap.get('reason', '보고서 품질을 위한 근거가 부족하다')}"
            )
    return "\n".join(lines)


def fallback_followups(state: MainState, gaps: list[dict]) -> Plan:
    """LLM 없이 만든 재계획. 빈 곳의 종류에 맞춘 검색 초점으로 한 번 더 찾는다."""
    flip = {"paper": "web", "web": "paper"}
    tasks = []
    for gap in gaps:
        perspective, technology = gap["perspective"], gap["technology"]
        if gap["kind"] in ("balance", "quality"):
            item = _items(state, perspective)[0]
            previous = _previous(state, perspective, item["id"], technology)
            source = flip[previous["source_type"]] if previous else DEFAULT_SOURCE[perspective]
            if gap["kind"] == "balance":
                missing = "·".join({"strength": "효과", "limitation": "한계"}[m] for m in gap["missing"])
                focus = f"{item.get('question', '')} ({missing} 측면)"
            else:
                focus = f"{item.get('question', '')} ({gap.get('reason', '보고서 품질 보완')})"
        else:
            item = gap["spec"]["criterion"]
            source = flip[gap["spec"]["source_type"]]
            focus = item.get("question", "")
        tasks.append(PlannedTask(perspective=perspective, criterion_id=item["id"], technology=technology, source_type=source, focus=focus))
    return Plan(tasks=tasks, rationale="기본 재계획: 빈 곳마다 출처를 바꿔 한 번 더 찾는다")


def select_followups(state: MainState, plan: Plan, gaps: list[dict], round_: int) -> tuple[list[dict], list[str]]:
    """재계획을 검사해 보낼 일감만 남긴다.

    빈 곳 목록에 없는 짝, 직전과 출처·초점이 똑같은 task(캐시 때문에 같은 결과가 나온다), 같은 짝의
    중복은 버린다. 상한을 넘으면 balance와 보고서 품질을 메우는 task를 먼저 남긴다.
    """
    balance = {(g["perspective"], g["technology"]) for g in gaps if g["kind"] == "balance"}
    quality = {(g["perspective"], g["technology"]) for g in gaps if g["kind"] == "quality"}
    gave_up = {(g["perspective"], g["spec"]["criterion"]["id"], g["technology"]) for g in gaps if g["kind"] == "gave_up"}
    items = {(p, item["id"]): item for p in EVAL_PERSPECTIVES for item in _items(state, p)}
    dropped: list[str] = []
    kept: dict[str, tuple[bool, dict]] = {}
    for task in plan.tasks:
        key = (task.perspective, task.criterion_id, task.technology)
        is_balance = (task.perspective, task.technology) in balance
        is_quality = (task.perspective, task.technology) in quality
        item = items.get((task.perspective, task.criterion_id))
        if item is None or not (is_balance or is_quality or key in gave_up):
            dropped.append(f"{'-'.join(key)}: 빈 곳 목록에 없음")
            continue
        previous = _previous(state, *key)
        if previous and previous["source_type"] == task.source_type and (previous.get("focus") or "") == task.focus.strip():
            dropped.append(f"{'-'.join(key)}: 직전과 출처·초점이 같음")
            continue
        spec = _spec(state, task.perspective, item, task.technology, source=task.source_type, focus=task.focus.strip() or None, round_=round_)
        kept.setdefault(spec["task_id"], (is_balance or is_quality, spec))
    ordered = sorted(kept.values(), key=lambda pair: not pair[0])  # balance가 앞으로 온다
    limit = int(config.get()["orchestration"]["max_followup_tasks"])
    dropped += [f"{spec['task_id']}: 재계획 task 수 상한" for _, spec in ordered[limit:]]
    return [spec for _, spec in ordered[:limit]], dropped


def replan_round(state: MainState, gaps: list[dict], round_: int) -> tuple[list[dict], str, list[str]]:
    plan = None
    if not config.is_dry_run():
        prompt = prompts.render(
            "orchestrator_replan",
            domain=_domain(state),
            technologies=_format_technologies(state),
            gaps=_format_gaps(state, gaps),
            max_tasks=int(config.get()["orchestration"]["max_followup_tasks"]),
        )
        plan = llm.generate(prompt, Plan)
    if plan is None:
        plan = fallback_followups(state, gaps)
    specs, dropped = select_followups(state, plan, gaps, round_)
    return specs, plan.rationale, dropped


# --- 노드와 분배 ------------------------------------------------------------------


def errored(state: MainState) -> dict[str, int]:
    """직전 회차에서 실행이 예외로 끝났고 아직 다시 보낼 수 있는 task와 다음 시도 번호."""
    limit = config.retry_limit("dispatch")
    status = state.get("task_status", {})
    resend = {}
    for spec in state.get("plan", []):
        entry = status.get(spec["task_id"], {})
        if entry.get("status") == "error" and entry.get("attempt", 1) <= limit:
            resend[spec["task_id"]] = entry["attempt"] + 1
    return resend


def _finish(run_id: str, reason: str, **detail: Any) -> MainState:
    """더 보낼 일이 없다. plan을 비워 dispatch가 종합으로 넘기게 한다. task_results는 collect가 회차마다 비운다."""
    log_decision(run_id, NODE, "finish", reason, **detail)
    return {"plan": []}


def node(state: MainState) -> MainState:
    """다음에 보낼 회차를 정한다. 위에서부터 처음 해당하는 경우 하나만 실행한다."""
    run_id = state.get("run_id", "-")
    orchestration = config.get()["orchestration"]
    steps = state.get("step_count", 0)
    if steps >= int(orchestration["max_steps"]):
        return _finish(run_id, f"step_count {steps}이 max_steps {orchestration['max_steps']}에 닿았다. 남은 빈 곳은 미확인으로 둔다")

    resend = errored(state)
    if resend:
        log_decision(run_id, NODE, "resend", "실행이 예외로 끝난 task를 다음 회차 전에 다시 보낸다", task_ids=sorted(resend))
        return {"task_status": {task_id: {"status": "pending", "attempt": attempt} for task_id, attempt in resend.items()}}

    quality = state.get("report_check", {})
    if quality.get("route") == "in_progress":
        return _finish(run_id, "보고서 품질 보완 조사가 끝나 다시 종합한다")

    if quality.get("route") == "replan":
        quality_count = state.get("quality_replan_count", 0)
        quality_limit = int(orchestration.get("max_quality_replans", 1))
        gaps = quality_gaps(state)
        if quality_count >= quality_limit or not gaps:
            log_decision(
                run_id, NODE, "quality_replan_skip",
                "품질 재계획 상한에 닿았거나 유효한 추가 조사 요청이 없다",
                quality_replans=quality_count,
            )
            return {"plan": [], "report_check": {**quality, "route": "in_progress"}}
        round_ = state.get("replan_count", 0) + quality_count + 1
        specs, rationale, dropped = replan_round(state, gaps, round_)
        log_decision(
            run_id, NODE, "quality_replan", rationale,
            round=quality_count + 1, gaps=len(gaps), tasks=len(specs),
            task_ids=[s["task_id"] for s in specs], dropped=dropped,
        )
        return {
            "plan": specs,
            "task_status": _pending(specs),
            "quality_replan_count": quality_count + 1,
            "report_check": {**quality, "route": "in_progress"},
        }

    plan = state.get("plan", [])
    if not plan and "tech_research" not in state:
        specs = tech_round(state)
        log_decision(run_id, NODE, "plan_tech", "관점 평가 계획의 배경이 될 기술 조사를 먼저 모두 보낸다", tasks=len(specs))
        return {"plan": specs, "task_status": _pending(specs)}

    if not plan or plan[0]["perspective"] == "tech":
        specs, rationale, fixes = eval_round(state)
        log_decision(run_id, NODE, "plan_eval", rationale, tasks=len(specs), **{k: v for k, v in fixes.items() if v})
        return {"plan": specs, "task_status": _pending(specs)}

    replans = state.get("replan_count", 0)
    if replans >= int(orchestration["max_replans"]):
        return _finish(run_id, f"재계획 {replans}회로 상한 {orchestration['max_replans']}에 닿았다")
    gaps = find_gaps(state)
    if not gaps:
        return _finish(run_id, "빈 곳이 없다")
    specs, rationale, dropped = replan_round(state, gaps, replans + 1)
    if not specs:
        return _finish(run_id, f"빈 곳 {len(gaps)}개를 메울 새 task가 없다. 미확인으로 둔다", dropped=dropped)
    log_decision(
        run_id, NODE, "replan", rationale, round=replans + 1, gaps=len(gaps), tasks=len(specs),
        task_ids=[s["task_id"] for s in specs], dropped=dropped,
    )
    return {"plan": specs, "task_status": _pending(specs), "replan_count": replans + 1}


def dispatch(state: MainState) -> list[Send] | str:
    """plan에서 pending인 task를 worker에게 하나씩 보낸다. 없으면 종합으로 넘긴다.

    중단 뒤 재개할 때도 이 규칙 그대로 끝나지 않은 task만 다시 나간다.
    """
    status = state.get("task_status", {})
    run_id = state.get("run_id", "-")
    sends = [
        Send("worker", {"task": spec, "run_id": run_id})
        for spec in state.get("plan", [])
        if status.get(spec["task_id"], {}).get("status") == "pending"
    ]
    return sends or DONE
