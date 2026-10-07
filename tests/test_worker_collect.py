"""외부 호출 없이 worker와 collect의 연결·재집계를 검증한다."""

import unittest
from copy import deepcopy
from unittest.mock import patch

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents import collect, perspective, subgraph
from core import config
from core.criteria import load_all
from core.state import MainState, merge, merge_or_reset


class WorkerCollectTest(unittest.TestCase):
    def setUp(self):
        config.apply_cli(dry_run=True, criteria_limit=1, retry=0)
        config.override("retry.dispatch", 1)
        self.state = {"criteria": load_all(), "selected_tech": []}

    def tearDown(self):
        config.reset()

    def test_single_workers_and_collect(self):
        for view, key in perspective.STATE_KEY.items():
            with self.subTest(view=view):
                task = perspective.build_tasks(view, self.state)[0]
                update = perspective.run_task(task)
                self.assertNotIn("errors", update)
                original = deepcopy(update)
                result = collect.node(update)
                self.assertTrue(result[key]["findings"])
                self.assertEqual(result["task_status"][task["task_id"]]["status"], "done")
                self.assertEqual(merge_or_reset(update["task_results"], result["task_results"]), {})
                self.assertEqual(update, original)

    def test_parallel_fanout_and_recollection(self):
        tasks = perspective.build_tasks("tech", self.state)
        workflow = StateGraph(MainState)
        workflow.add_node("worker", lambda state: perspective.run_task(state["task"]))
        workflow.add_node("collect", collect.node)
        workflow.add_conditional_edges(START, lambda state: [Send("worker", {"task": t}) for t in tasks])
        workflow.add_edge("worker", "collect")
        workflow.add_edge("collect", END)
        cfg = config.get()
        state = workflow.compile().invoke({}, config={
            "max_concurrency": cfg["execution"]["max_concurrency"],
            "recursion_limit": cfg["orchestration"]["recursion_limit"],
        })
        self.assertEqual(len(state["task_status"]), len(tasks))
        self.assertEqual(state["task_results"], {})
        before = state["tech_research"]["findings"]
        task = {**tasks[0], "task_id": tasks[0]["task_id"] + "-r1", "round": 1}
        update = perspective.run_task(task)
        state.update(update)
        collected = collect.node(state)
        after = collected["tech_research"]["findings"]
        self.assertGreater(len(after), len(before))
        self.assertTrue({f["id"] for f in before} <= {f["id"] for f in after})
        state.update(collected)
        state.update(update)
        again = collect.node(state)
        self.assertEqual(again["tech_research"]["findings"], after)

    def test_empty_and_failed_tasks(self):
        task = perspective.build_tasks("market", self.state)[0]
        with patch.object(perspective, "_subgraph") as build:
            build.return_value.invoke.return_value = {"findings": [], "gaps": ["근거 부족"]}
            state = perspective.run_task(task)
            result = collect.node(state)
            self.assertEqual(result["task_status"][task["task_id"]]["status"], "gave_up")
            self.assertIn("근거 부족", result["market_eval"]["gaps"])
            build.return_value.invoke.side_effect = RuntimeError("검색 실패")
            with patch.object(perspective.log, "exception"):
                state = perspective.run_task(task)
            result = collect.node(state)
            self.assertNotIn("market_eval", result)
            statuses = merge(state["task_status"], result["task_status"])
            self.assertEqual(statuses[task["task_id"]]["status"], "error")
            self.assertIn(task["task_id"], state["errors"])
            state["task_status"][task["task_id"]]["attempt"] = 2
            exhausted = collect.node(state)
            self.assertEqual(exhausted["task_status"][task["task_id"]]["status"], "gave_up")
            self.assertIn("검색 실패", " ".join(exhausted["market_eval"]["gaps"]))
        self.assertEqual(collect.node({}), {})

    def test_focus_and_checkpoint(self):
        task = perspective.build_tasks("market", self.state)[0]
        task["focus"] = "독립적인 상용 채택 사례"
        self.assertIn(task["focus"], subgraph._query_prompt("perspective/market", task))
        self.assertIn(task["focus"], subgraph._queries_to_str(None, task))
        with patch.object(subgraph.llm, "generate", return_value=None) as generate:
            config.override("dry_run", False)
            subgraph._query_rewrite("perspective/market")({"task": task, "query": "previous"})
            self.assertIn(task["focus"], generate.call_args.args[0])
        config.override("dry_run", True)
        self.assertIs(perspective._subgraph("market", "web").checkpointer, False)

    def test_planned_source_overrides_default(self):
        task = perspective.build_tasks("market", self.state)[0]
        task["source_type"] = "paper"
        with patch.object(perspective, "_search_fn", return_value=lambda query, k: [] ) as search, \
                patch.object(perspective, "build_subgraph") as build:
            build.return_value.invoke.return_value = {"findings": [], "gaps": []}
            perspective.run_task(task)
            search.assert_called_once_with("paper")
            self.assertIs(build.call_args.kwargs["check_citation"], perspective.checks.rag_check)


if __name__ == "__main__":
    unittest.main()
