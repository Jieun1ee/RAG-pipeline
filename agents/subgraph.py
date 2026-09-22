"""기준 하나를 근거까지 확인해 처리하는 작은 그래프.

관점은 다섯이지만 절차는 같다. 질의를 만들고, 검색하고, 쓸 만한 결과인지 보고, 서술을 쓰고,
인용이 실제로 있는지 확인하고, 그 인용이 주장을 뒷받침하는지 판정한다. 달라지는 것은 어떤 검색을
쓰는지, 인용을 어떻게 확인하는지, 어떤 역할 지시를 앞에 붙이는지 셋뿐이라 그 셋만 인자로 받는다.

    query_gen → search → relevance_judge → finding_gen → citation_check → support_judge → 끝

검사에 걸리면 앞 단계로 돌아간다. 검색 결과가 쓸 만하지 않으면 질의를 고쳐 다시 찾고, 인용이나
근거가 어긋나면 서술을 다시 쓴다. 정해 둔 횟수를 넘기면 되돌아가는 대신 그 항목을 버리고 이유를
남긴다. 근거가 약한 서술을 남기지 않으면서 실행이 끝나게 하려는 것이다.

외부 호출을 끈 모드에서는 모델을 쓰는 노드가 샘플 값을 돌려준다. 연결과 분기만 따로 확인할 수 있다.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from functools import wraps
from typing import Callable
from urllib.parse import urlparse

import yaml
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents import checks
from core import config, llm, prompts
from core.schemas import CheckResult, FindingDraft, FindingList, QueryPair, RelevanceJudgement, Retrieved
from core.state import SubState, load_fixture

log = logging.getLogger(__name__)

SearchFn = Callable[[str, int], list[Retrieved]]
CitationCheckFn = Callable[[dict, list[dict]], CheckResult]

# 노드별 실행 횟수. 실행이 끝난 뒤 어디서 재시도가 몰렸는지 보려고 모아 둔다.
NODE_RUNS: Counter[str] = Counter()

FIXTURE_BY_PERSPECTIVE = {
    "tech": "tech_research",
    "trl": "trl_eval",
    "market": "market_eval",
    "stakeholder": "stakeholder_eval",
    "domain": "domain_eval",
}
MAX_TEXT = 1800  # 프롬프트에 넣을 검색 결과 본문 길이 상한
DIRECTION = "효과·채택 / 한계·반대 근거 (두 방향 모두)"
_DATE_HEAD = re.compile(r"\d{4}(?:-\d{2}){0,2}")


def publication_date(raw: str | None) -> str:
    """검색 결과가 알려 준 발행일. 모르면 n.d.로 둔다.

    자료를 찾은 날짜를 발행일 자리에 넣으면 그 자료가 언제 나온 것인지 잘못 전달된다.
    검색 도구가 날짜를 주지 않으면 없는 대로 둔다. 참고문헌에는 n.d.로 적힌다.
    시각까지 붙어 오는 경우가 있어 앞의 날짜 부분만 남긴다.
    """
    text = (raw or "").strip()
    if not text:
        return "n.d."
    match = _DATE_HEAD.match(text)
    return match.group(0) if match else text


def counted(name: str, fn: Callable[..., dict]) -> Callable[..., dict]:
    """노드가 몇 번 불렸는지 세는 덮개.

    재시도 때문에 같은 노드가 여러 번 돈다. 실행이 끝난 뒤 흐름을 되짚으려면 횟수가 필요하다.
    """

    @wraps(fn)
    def wrapper(state):  # noqa: ANN001 - 그래프가 상태 하나만 넘긴다
        NODE_RUNS[name] += 1
        return fn(state)

    return wrapper


def format_retrieved(retrieved: list[dict]) -> str:
    """검색 결과를 프롬프트에 넣을 형태로 만든다.

    항목마다 태그로 경계를 두른다. 본문에 줄바꿈이나 목록이 들어가도 어디서 끊기는지 헷갈리지 않는다.
    id를 함께 실어, 모델이 인용할 때 그 값을 그대로 옮겨 쓰게 한다. 본문은 상한까지만 넣는다.
    """
    if not retrieved:
        return "(검색 결과 없음)"
    blocks = []
    for r in retrieved:
        meta = "".join(
            f"<{tag}>{value}</{tag}>"
            for tag, value in (
                ("title", r.get("title")),
                ("author", r.get("author_or_org")),
                ("date", r.get("date")),
                ("page", r.get("page")),
            )
            if value
        )
        blocks.append(
            f"<document><id>{r['id']}</id>{meta}"
            f"<content>{r.get('text', '')[:MAX_TEXT]}</content></document>"
        )
    return "\n".join(blocks)


def _query_prompt(role_prompt: str, task: dict) -> str:
    """관점 역할 지시와 질의 작성 지시를 이어 붙인 프롬프트."""
    criterion = task["criterion"]
    body = prompts.render(
        "query_gen",
        technology=task["technology"],
        tech_name=task.get("tech_name") or task["technology"],
        criterion_id=criterion["id"],
        criterion_name=criterion.get("name", ""),
        criterion_question=criterion.get("question", ""),
        direction=DIRECTION,
        tech_summary=task.get("tech_summary") or "(없음)",
        source_type=task.get("source_type", "paper"),
    )
    return prompts.body(role_prompt) + "\n\n" + body


def _queries_to_str(pair: QueryPair | None, task: dict) -> str:
    """질의 두 개를 줄 단위로 이어 붙인다. 모델이 답을 주지 못하면 기술과 기준 이름으로 대신한다."""
    if pair is None:
        return f"{task.get('tech_name') or task['technology']} {task['criterion'].get('name', '')}".strip()
    return "\n".join(q.strip() for q in (pair.effect, pair.limitation) if q and q.strip())


def _query_gen(role_prompt: str, perspective: str) -> Callable[[SubState], SubState]:
    """기준을 검색 질의로 바꾸는 노드를 만든다."""

    def query_gen(state: SubState) -> SubState:
        task = state["task"]
        if config.is_dry_run():
            return {"query": f"[dry-run] {task['technology']} {task['criterion']['id']}"}
        pair = llm.generate(_query_prompt(role_prompt, task), QueryPair)
        return {"query": _queries_to_str(pair, task)}

    return query_gen


def _search(search: SearchFn) -> Callable[[SubState], SubState]:
    """넘겨받은 검색 함수로 질의를 실행하는 노드를 만든다."""

    def search_node(state: SubState) -> SubState:
        # 질의가 여러 줄이면 각각 검색한 뒤 id로 합친다. 두 질의에 같은 문서가 걸려도 한 번만 남는다.
        k = int(config.get()["retrieval"]["top_k"])
        merged: dict[str, dict] = {}
        for query in [q for q in state["query"].splitlines() if q.strip()]:
            for r in search(query, k):
                item = r.model_dump(mode="json") if isinstance(r, Retrieved) else dict(r)
                merged.setdefault(item["id"], item)
        return {"retrieved": list(merged.values())[: 2 * k]}

    return search_node


def _relevance_judge(state: SubState) -> SubState:
    """검색 결과에서 그 기준에 답이 될 만한 것만 남긴다.

    남는 것이 없으면 실패로 표시한다. 관련 없는 결과를 그대로 넘기면 모델이 억지로 근거를 만들어 낸다.
    """
    attempt = state.get("retrieval_check", {}).get("attempt", 0) + 1
    if config.is_dry_run():
        return {"retrieval_check": CheckResult(passed=True, issues=[], attempt=attempt).model_dump()}
    retrieved = state.get("retrieved", [])
    if not retrieved:
        return {"retrieval_check": CheckResult(passed=False, issues=["검색 결과가 없다"], attempt=attempt).model_dump()}
    criterion = state["task"]["criterion"]
    prompt = prompts.render(
        "relevance_judge",
        criterion_name=criterion.get("name", ""),
        criterion_question=criterion.get("question", ""),
        retrieved=format_retrieved(retrieved),
    )
    judgement = llm.judge(prompt, RelevanceJudgement)
    relevant_ids = set(judgement.relevant_ids) if judgement else set()
    relevant = [r for r in retrieved if r["id"] in relevant_ids]
    if relevant:
        return {"retrieval_check": CheckResult(passed=True, issues=[], attempt=attempt).model_dump(), "retrieved": relevant}
    reason = judgement.reason if judgement else "관련성 판정 응답 실패"
    return {"retrieval_check": CheckResult(passed=False, issues=[reason], attempt=attempt).model_dump()}


def _query_rewrite(role_prompt: str) -> Callable[[SubState], SubState]:
    """직전 질의가 왜 빗나갔는지를 붙여 질의를 다시 만드는 노드."""

    def query_rewrite(state: SubState) -> SubState:
        if config.is_dry_run():
            return {"query": f"{state['query']} (rewrite)"}
        task = state["task"]
        issues = "; ".join(state.get("retrieval_check", {}).get("issues", []))
        prompt = (
            _query_prompt(role_prompt, task)
            + f"\n\n이전 질의:\n{state['query']}\n\n이전 질의로 관련 결과를 찾지 못했다. 이유: {issues}\n"
            "다른 표현, 동의어, 더 구체적이거나 더 넓은 키워드로 질의를 다시 만든다."
        )
        pair = llm.generate(prompt, QueryPair)
        return {"query": _queries_to_str(pair, task)}

    return query_rewrite


def materialize(draft: FindingDraft, seq: int, task: dict, retrieved: list[dict]) -> dict:
    """모델이 쓴 초안에 코드가 아는 값을 채워 완성된 서술로 만든다.

    제목, 저자, 발행처, 발행일은 검색 결과와 등록 정보에서 가져온다. 모델에게 맡기면 그럴듯하게 지어낼 수
    있는 값이라서다. 발행일을 알 수 없으면 비워 두지 않고 n.d.로 적어, 참고문헌에서 날짜가 없다는 사실이
    드러나게 한다.
    """
    from rag.retriever import registry_docs

    by_id = {r["id"]: r for r in retrieved}
    evidence = []
    for ev in draft.evidence:
        r = by_id.get(ev.ref_id, {})
        source_type = r.get("source_type") or task.get("source_type") or "web"
        if source_type == "paper":
            venue = registry_docs().get(r.get("doc_id", ""), {}).get("venue", "")
        else:
            venue = urlparse(r.get("url") or ev.ref_id).netloc.removeprefix("www.")
        locator = ev.locator
        if source_type == "paper" and not locator and r.get("page"):
            locator = f"p.{r['page']}"
        evidence.append(
            {
                "ref_id": ev.ref_id,
                "quote": ev.quote,
                "source_type": source_type,
                "source_nature": ev.source_nature,
                "is_self_reported": ev.is_self_reported,
                "title": r.get("title") or "",
                "author_or_org": r.get("author_or_org") or "",
                "venue": venue or "",
                "date": publication_date(r.get("date")),
                "url": r.get("url"),
                "locator": locator,
                "summary": ev.summary,
            }
        )
    return {
        "id": f"{task['perspective']}-{task['criterion']['id']}-{task['technology']}-{seq:02d}",
        "technology": task["technology"],
        "criterion": task["criterion"]["id"],
        "claim": draft.claim,
        "polarity": draft.polarity,
        "evidence": evidence,
        "conditions": draft.conditions,
        "confidence": draft.confidence,
    }


def _finding_gen(role_prompt: str, perspective: str) -> Callable[[SubState], SubState]:
    """검색 결과를 근거로 서술을 쓰는 노드를 만든다."""

    def finding_gen(state: SubState) -> SubState:
        task = state["task"]
        if config.is_dry_run():
            data = load_fixture(FIXTURE_BY_PERSPECTIVE[perspective])
            findings = [
                f
                for f in data.get("findings", [])
                if f.get("technology") == task["technology"] and f.get("criterion") == task["criterion"]["id"]
            ]
            return {"findings": findings}
        retrieved = state.get("retrieved", [])
        # 직전 검사에서 걸린 이유를 그대로 붙인다. 같은 잘못을 반복하지 않게 하려는 것이다.
        feedback = "\n".join(state.get("citation_check", {}).get("issues", [])) or "(없음)"
        levels = task.get("levels") or {}
        body = prompts.render(
            "finding_gen",
            technology=task["technology"],
            criterion=yaml.safe_dump(task["criterion"], allow_unicode=True, sort_keys=False).strip(),
            levels=yaml.safe_dump(levels, allow_unicode=True, sort_keys=False).strip() if levels else "(없음)",
            retrieved=format_retrieved(retrieved),
            feedback=feedback,
        )
        drafts = llm.generate(prompts.body(role_prompt) + "\n\n" + body, FindingList)
        if drafts is None:
            return {"findings": []}
        # 기준 하나에서 너무 많이 받으면 뒤따르는 검사 비용만 늘어난다
        return {"findings": [materialize(d, i, task, retrieved) for i, d in enumerate(drafts.findings[:3], start=1)]}

    return finding_gen


def _citation_check(check_citation: CitationCheckFn, perspective: str) -> Callable[[SubState], SubState]:
    """서술의 필수 항목과 인용 실재를 확인하는 노드를 만든다.

    모델을 부르지 않는 검사라 값이 싸고 결과가 늘 같다. 그래서 비용이 드는 판정보다 먼저 둔다.
    """

    def citation_check(state: SubState) -> SubState:
        attempt = state.get("citation_check", {}).get("attempt", 0) + 1
        retrieved = state.get("retrieved", [])
        findings = state.get("findings", [])
        issues: list[str] = []
        # 서술이 하나도 없으면 검사할 것이 없어 그냥 통과해 버린다. 그러면 다시 써 보지도,
        # 미확인으로 남기지도 않고 끝난다. 실패로 돌려 재시도 규칙에 태운다.
        # 샘플로 도는 모드에서는 대부분의 기준이 비어 있는 것이 정상이라 이 검사를 건너뛴다.
        if not findings and not config.is_dry_run():
            issues.append("검색 결과는 관련 있다고 보았으나 서술이 하나도 만들어지지 않았다")
        for finding in findings:
            fid = finding.get("id", "?")
            issues += [f"{fid}: {msg}" for msg in checks.field_check(finding, perspective).issues]
            for evidence in finding.get("evidence", []):
                issues += [f"{fid}: {msg}" for msg in check_citation(evidence, retrieved).issues]
        return {"citation_check": CheckResult(passed=not issues, issues=issues, attempt=attempt).model_dump()}

    return citation_check


def _support_judge(state: SubState) -> SubState:
    """인용이 주장을 실제로 뒷받침하는지 판정한다.

    결과를 앞 검사와 같은 자리에 쓴다. 둘 다 서술을 다시 쓰게 만드는 검사라 시도 횟수를 함께 세야
    되돌아가는 횟수가 두 배로 늘지 않는다.
    """
    previous = state.get("citation_check", {})
    attempt = previous.get("attempt", 1)
    if config.is_dry_run():
        return {"citation_check": CheckResult(passed=True, issues=[], attempt=attempt).model_dump()}
    issues: list[str] = []
    for finding in state.get("findings", []):
        issues += [f"{finding.get('id', '?')}: {msg}" for msg in checks.support_judge(finding).issues]
    return {"citation_check": CheckResult(passed=not issues, issues=issues, attempt=attempt).model_dump()}


def _record_gaps(state: SubState) -> SubState:
    """되돌릴 수 있는 횟수를 다 쓴 항목을 버리고 그 이유를 남긴다.

    검색을 끝내 못 했는지 서술이 검사를 통과하지 못했는지를 구분해 적는다. 보고서에서 무엇을 확인하지
    못했는지 밝히는 근거가 된다.
    """
    task = state["task"]
    label = f"{task['perspective']} {task['criterion']['id']} {task['technology']}"
    if not state.get("retrieval_check", {}).get("passed", False):
        reason = f"{label}: 관련 검색 결과를 찾지 못함 (재시도 상한 도달)"
    else:
        issues = "; ".join(state.get("citation_check", {}).get("issues", []))
        reason = f"{label}: Finding 검증 실패로 제거 (재시도 상한 도달): {issues}"
    return {"findings": [], "gaps": list(state.get("gaps", [])) + [reason]}


def route_after_relevance(state: SubState) -> str:
    """검색 결과 판정 뒤의 상황 이름. 어느 노드로 갈지는 그래프를 엮을 때 정한다."""
    check = state.get("retrieval_check", {})
    if check.get("passed"):
        return "relevant"
    if check.get("attempt", 0) <= config.retry_limit("retrieval"):
        return "retry"
    return "give up"


def route_after_finding_check(state: SubState) -> str:
    """서술 검사 뒤의 상황 이름.

    인용 확인과 근거 판정이 같은 규칙을 쓰고 시도 횟수도 함께 세므로 한 함수로 둔다.
    통과했을 때 갈 곳만 두 자리에서 다르게 잇는다.
    """
    check = state.get("citation_check", {})
    if check.get("passed"):
        return "verified"
    if check.get("attempt", 0) <= config.retry_limit("citation"):
        return "retry"
    return "give up"


def build_subgraph(search: SearchFn, check_citation: CitationCheckFn, role_prompt: str, perspective: str) -> CompiledStateGraph:
    """관점 하나가 쓸 서브그래프를 만들어 컴파일한다.

    search: 질의와 개수를 받아 검색 결과를 돌려주는 함수. 논문 검색이나 웹 검색을 넣는다.
    check_citation: 인용 하나가 검색 결과에 실재하는지 확인하는 함수. 검색 종류에 맞는 것을 넣는다.
    role_prompt: 질의와 서술 지시 앞에 붙일 역할 지시 파일 이름.
    perspective: 기록과 샘플 조회에 쓰는 관점 이름.
    """
    nodes: dict[str, Callable[[SubState], SubState]] = {
        "query_gen": _query_gen(role_prompt, perspective),
        "search": _search(search),
        "relevance_judge": _relevance_judge,
        "query_rewrite": _query_rewrite(role_prompt),
        "finding_gen": _finding_gen(role_prompt, perspective),
        "citation_check": _citation_check(check_citation, perspective),
        "support_judge": _support_judge,
        "record_gaps": _record_gaps,
    }
    workflow = StateGraph(SubState)
    for name, fn in nodes.items():
        workflow.add_node(name, counted(f"{perspective}/{name}", fn))

    workflow.add_edge(START, "query_gen")
    workflow.add_edge("query_gen", "search")
    workflow.add_edge("search", "relevance_judge")
    workflow.add_conditional_edges(
        "relevance_judge",
        route_after_relevance,
        {
            "relevant": "finding_gen",   # 쓸 만한 결과를 찾았으니 서술로 넘어간다
            "retry": "query_rewrite",    # 질의를 고쳐 다시 검색한다
            "give up": "record_gaps",    # 더 시도하지 않고 미확인으로 남긴다
        },
    )
    workflow.add_edge("query_rewrite", "search")
    workflow.add_edge("finding_gen", "citation_check")
    workflow.add_conditional_edges(
        "citation_check",
        route_after_finding_check,
        {
            "verified": "support_judge",  # 인용이 실재하니 근거 판정으로 넘어간다
            "retry": "finding_gen",       # 서술을 다시 쓴다
            "give up": "record_gaps",
        },
    )
    workflow.add_conditional_edges(
        "support_judge",
        route_after_finding_check,
        {
            "verified": END,              # 근거까지 확인됐으면 이 기준은 끝난다
            "retry": "finding_gen",
            "give up": "record_gaps",
        },
    )
    workflow.add_edge("record_gaps", END)
    return workflow.compile(name=f"subgraph:{perspective}")
