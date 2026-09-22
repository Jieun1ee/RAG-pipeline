"""평가 그래프 조립.

    select_tech → tech_research → [trl_eval, market_eval, stakeholder_eval, domain_eval] → synthesis
    → synthesis_check → report → report_check → 끝

기술 조사를 먼저 하는 것은 뒤 네 관점이 그 결과를 배경으로 질의를 만들기 때문이다. 네 관점은 서로를
기다릴 이유가 없어 동시에 돈다. 종합과 보고서는 각각 검사를 달고 있어, 걸리면 정해진 횟수까지 다시 쓴다.

명령줄 처리와 떼어 두었다. 그래프는 도메인 로직이고 명령줄은 그것을 부르는 여러 방법 중 하나다.
모듈 수준의 graph가 컴파일된 그래프라, 시각화 도구나 다른 진입점에서 그대로 가져다 쓸 수 있다.
"""

from __future__ import annotations

from typing import Callable

import yaml
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents import checks, perspective, report, synthesis
from agents.subgraph import counted
from core import config
from core.criteria import load_all
from core.state import MainState

PERSPECTIVE_NODES = ["trl_eval", "market_eval", "stakeholder_eval", "domain_eval"]


def select_tech(state: MainState) -> MainState:
    """평가 대상과 평가 기준을 읽어 상태에 올린다. 그래프의 첫 노드다."""
    cfg = config.get()
    with open(config.resolve(cfg["paths"]["registry"]), encoding="utf-8") as f:
        registry = yaml.safe_load(f)
    criteria = load_all()
    criteria["domain_name"] = registry.get("domain", "")
    return {"selected_tech": [dict(t) for t in registry["selected_tech"]], "criteria": criteria}


NODES: dict[str, Callable[[MainState], MainState]] = {
    "select_tech": select_tech,
    "tech_research": perspective.make_node("tech"),
    "trl_eval": perspective.make_node("trl"),
    "market_eval": perspective.make_node("market"),
    "stakeholder_eval": perspective.make_node("stakeholder"),
    "domain_eval": perspective.make_node("domain"),
    "synthesis": synthesis.node,
    "synthesis_check": checks.synthesis_check,
    "report": report.node,
    "report_check": checks.report_check,
}

# 각 노드가 읽는 상태 키. 노드를 하나만 돌릴 때 이 목록을 보고 입력을 채운다.
_EVALS = ["tech_research", *PERSPECTIVE_NODES]
NODE_INPUTS: dict[str, list[str]] = {
    "select_tech": [],
    "tech_research": ["selected_tech", "criteria"],
    "trl_eval": ["selected_tech", "criteria", "tech_research"],
    "market_eval": ["selected_tech", "criteria", "tech_research"],
    "stakeholder_eval": ["selected_tech", "criteria", "tech_research"],
    "domain_eval": ["selected_tech", "criteria", "tech_research"],
    "synthesis": ["criteria", *_EVALS, "synthesis_check"],
    "synthesis_check": ["criteria", *_EVALS, "synthesis"],
    "report": ["selected_tech", "criteria", *_EVALS, "synthesis", "synthesis_check", "report_check"],
    "report_check": ["selected_tech", *_EVALS, "synthesis", "final_report"],
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


def build_graph() -> CompiledStateGraph:
    """노드와 연결을 붙여 실행할 수 있는 그래프로 만든다.

    네 관점은 동시에 돌지만 서로 다른 상태 키에만 쓰기 때문에 결과가 덮이지 않는다.
    """
    workflow = StateGraph(MainState)
    for name, fn in NODES.items():
        workflow.add_node(name, counted(name, fn))

    workflow.add_edge(START, "select_tech")
    workflow.add_edge("select_tech", "tech_research")
    for name in PERSPECTIVE_NODES:
        workflow.add_edge("tech_research", name)
    workflow.add_edge(PERSPECTIVE_NODES, "synthesis")  # 네 관점이 모두 끝난 뒤 한 번만 돈다
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
    return workflow.compile(name="kv-cache-eval")


# 시각화 도구와 다른 진입점이 가져다 쓰는 이름. 조립만 하므로 불러들이는 비용이 거의 없다.
graph = build_graph()
