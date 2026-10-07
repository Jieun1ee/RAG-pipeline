"""평가 그래프 조립.

    select_tech → orchestrator ─(Send × N)→ worker → collect ─┐
                      ▲                                        │
                      └────────────────────────────────────────┘
                      └─ 더 보낼 task가 없으면 → synthesis → synthesis_check → report → report_check → 끝

orchestrator가 회차마다 보낼 task를 정하고, worker는 task 하나(기준 하나 × 기술 하나)를 처리한다.
같은 회차의 worker들은 동시에 돈다. collect가 결과를 관점별로 묶으면 orchestrator로 돌아가 다음 회차를
정한다. 기술 조사, 관점 평가, 재계획 순으로 돌고, 무엇을 보낼지와 언제 멈출지는 orchestrator가 정한다.
종합과 보고서는 각각 검사를 달고 있어, 걸리면 정해진 횟수까지 다시 쓴다.

메인 노드는 실행될 때마다 step_count를 1씩 올린다. orchestrator가 이 값을 max_steps와 견줘 멈춘다.
worker는 task 수만큼 돌기 때문에 세지 않는다.

명령줄 처리와 떼어 두었다. 그래프는 도메인 로직이고 명령줄은 그것을 부르는 여러 방법 중 하나다.
모듈 수준의 graph가 컴파일된 그래프라, 시각화 도구나 다른 진입점에서 그대로 가져다 쓸 수 있다.
"""

from __future__ import annotations

from functools import wraps
from typing import Callable

import yaml
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents import checks, orchestrator, report, synthesis
from agents import placeholder  # A의 worker·collect 구현이 들어오면 교체한다
from agents.subgraph import counted
from core import config
from core.criteria import load_all
from core.state import MainState

EVAL_KEYS = ["tech_research", "trl_eval", "market_eval", "stakeholder_eval", "domain_eval"]


def select_tech(state: MainState) -> MainState:
    """평가 대상과 평가 기준을 읽어 상태에 올린다. 그래프의 첫 노드다."""
    cfg = config.get()
    with open(config.resolve(cfg["paths"]["registry"]), encoding="utf-8") as f:
        registry = yaml.safe_load(f)
    return {
        "selected_tech": [dict(t) for t in registry["selected_tech"]],
        "target_domain": registry.get("domain", ""),
        "criteria": load_all(),
    }


def stepped(fn: Callable[[MainState], MainState]) -> Callable[[MainState], MainState]:
    """메인 노드의 반환값에 step_count +1을 얹는다. step_count는 더하기 reducer라 1만 보내면 된다."""

    @wraps(fn)
    def wrapper(state: MainState) -> MainState:
        return {**fn(state), "step_count": 1}

    return wrapper


# step_count를 세는 메인 노드. worker는 Send로만 불려 여기 넣지 않는다.
NODES: dict[str, Callable[[MainState], MainState]] = {
    "select_tech": select_tech,
    "orchestrator": orchestrator.node,
    "collect": placeholder.collect,
    "synthesis": synthesis.node,
    "synthesis_check": checks.synthesis_check,
    "report": report.node,
    "report_check": checks.report_check,
}

# 각 노드가 읽는 상태 키. 노드를 하나만 돌릴 때 이 목록을 보고 입력을 채운다.
# orchestrator는 기술 조사 결과가 채워져 있으므로 관점 평가 계획부터 세운다.
NODE_INPUTS: dict[str, list[str]] = {
    "select_tech": [],
    "orchestrator": ["selected_tech", "target_domain", "criteria", "tech_research"],
    "collect": ["task_results"],
    "synthesis": ["criteria", *EVAL_KEYS, "synthesis_check"],
    "synthesis_check": ["criteria", *EVAL_KEYS, "synthesis"],
    "report": ["selected_tech", "target_domain", "criteria", *EVAL_KEYS, "synthesis", "synthesis_check", "report_check"],
    "report_check": ["selected_tech", "target_domain", *EVAL_KEYS, "synthesis", "final_report"],
}


def route_after_synthesis_check(state: MainState) -> str:
    """종합 검사 뒤의 상황 이름. 통과했거나 다시 쓸 횟수를 다 썼으면 넘어간다."""
    check = state.get("synthesis_check", {})
    if check.get("passed") or check.get("attempt", 0) > config.retry_limit("synthesis"):
        return "proceed"
    return "retry"


def route_after_report_check(state: MainState) -> str:
    """보고서 검사 뒤의 상황 이름. 통과했거나 횟수를 다 썼으면 끝낸다."""
    check = state.get("report_check", {})
    if check.get("passed") or check.get("attempt", 0) > config.retry_limit("report"):
        return "done"
    return "retry"


def build_graph(checkpointer=None) -> CompiledStateGraph:  # noqa: ANN001 - LangGraph 체크포인터라면 무엇이든
    """노드와 연결을 붙여 실행할 수 있는 그래프로 만든다.

    checkpointer를 넘기면 단계마다 상태를 저장해, 중단된 실행을 같은 thread_id로 이어서 돌릴 수 있다.
    """
    workflow = StateGraph(MainState)
    for name, fn in NODES.items():
        workflow.add_node(name, counted(name, stepped(fn)))
    workflow.add_node("worker", counted("worker", placeholder.worker))

    workflow.add_edge(START, "select_tech")
    workflow.add_edge("select_tech", "orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        orchestrator.dispatch,
        {
            "worker": "worker",                  # pending인 task를 Send로 하나씩 보낸다
            orchestrator.DONE: "synthesis",      # 더 보낼 task가 없다
        },
    )
    workflow.add_edge("worker", "collect")       # 같은 회차의 worker가 모두 끝난 뒤 한 번만 돈다
    workflow.add_edge("collect", "orchestrator")
    workflow.add_edge("synthesis", "synthesis_check")
    workflow.add_conditional_edges(
        "synthesis_check",
        route_after_synthesis_check,
        {
            "proceed": "report",    # 종합이 쓸 만하니 보고서로 넘어간다
            "retry": "synthesis",   # 지적을 붙여 종합을 다시 쓴다
        },
    )
    workflow.add_edge("report", "report_check")
    workflow.add_conditional_edges(
        "report_check",
        route_after_report_check,
        {
            "done": END,
            "retry": "report",      # 지적을 붙여 보고서를 다시 쓴다
        },
    )
    return workflow.compile(name="kv-cache-eval", checkpointer=checkpointer)


# 시각화 도구와 다른 진입점이 가져다 쓰는 이름. 조립만 하므로 불러들이는 비용이 거의 없다.
graph = build_graph()
