"""
Evaluation harness.  Run from the project folder:

    python evals.py --reset-db                  # reseed Postgres, then run every suite
    python evals.py --suite routing guardrail   # run some suites
    python evals.py --suite rag

Suites (cases live in evals/golden.jsonl):
  guardrail  deterministic  - does the input filter block/allow the right messages?
  routing    deterministic  - does the supervisor pick the right intent?
  e2e        deterministic  - full graph run; asserts intent, tool calls, HITL pause, FORBIDDEN tool calls, reply text
  rag        LLM-as-judge   - correctness vs reference, faithfulness to retrieved docs, and abstention on unanswerable questions

Exits non-zero if any metric is below its threshold, so it can gate CI.
"""
import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

from langgraph.types import Command
from pydantic import BaseModel

from gateway import gateway
from rag import rag_answer
from support_graph import build_graph, run_guard, run_route

HERE = Path(__file__).parent
THRESHOLDS = {
    "guardrail_acc": 0.90, "routing_acc": 0.90, "e2e_pass": 0.90,
    "rag_correctness": 3.5, "rag_faithfulness": 0.90, "rag_abstain_acc": 0.90,
}


class Judge(BaseModel):
    correctness: int      # 1-5 vs the reference answer
    faithful: bool        # every claim supported by the retrieved context
    reason: str


JUDGE_PROMPT = (
    "You are grading a customer-support answer.\n\nQuestion: {q}\nReference answer: {ref}\n"
    "Retrieved context:\n{ctx}\n\nCandidate answer: {ans}\n\n"
    "correctness: 1-5, how well the candidate matches the reference (5 = fully correct, 1 = wrong/missing).\n"
    "faithful: true only if every claim in the candidate is supported by the retrieved context.\n"
    "reason: one sentence."
)


def load_cases():
    cases = {}
    for line in (HERE / "evals" / "golden.jsonl").read_text().splitlines():
        if line.strip():
            c = json.loads(line)
            cases.setdefault(c["type"], []).append(c)
    return cases


async def gather(fn, cases, limit=4):
    sem = asyncio.Semaphore(limit)

    async def one(c):
        async with sem:
            try:
                return await fn(c)
            except Exception as e:  # noqa: BLE001 - an erroring case is a failing case
                return e
    return await asyncio.gather(*(one(c) for c in cases))


# ---------- suites ----------
async def suite_guardrail(cases):
    res = await gather(lambda c: run_guard(c["input"]), cases)
    ok = [(not isinstance(r, Exception)) and r.safe == c["expect_safe"] for r, c in zip(res, cases)]
    return {"guardrail_acc": sum(ok) / len(ok)}, [c["input"] for c, k in zip(cases, ok) if not k]


async def suite_routing(cases):
    res = await gather(lambda c: run_route(c["input"]), cases)
    ok = [(not isinstance(r, Exception)) and r.intent == c["expect_intent"] for r, c in zip(res, cases)]
    return {"routing_acc": sum(ok) / len(ok)}, [c["input"] for c, k in zip(cases, ok) if not k]


def _tools(state):
    return [t.split(":", 1)[1] for t in state.get("trace", []) if t.startswith("tool:")]


async def run_e2e_case(graph, c) -> list[str]:
    """Returns a list of failed check names (empty = pass)."""
    cfg = {"configurable": {"thread_id": f"{c['customer_id']}:eval-{uuid.uuid4().hex[:8]}"}}
    out = await graph.ainvoke({"messages": [("user", c["input"])], "customer_id": c["customer_id"]}, cfg)
    interrupted = "__interrupt__" in out
    tools_before = _tools((await graph.aget_state(cfg)).values)

    if interrupted and "resume" in c:
        out = await graph.ainvoke(Command(resume=c["resume"]), cfg)
    state = (await graph.aget_state(cfg)).values
    tools, reply = _tools(state), (out["messages"][-1].content if out.get("messages") else "")

    failed = []
    if "expect_intent" in c and state.get("intent") != c["expect_intent"]:
        failed.append(f"intent={state.get('intent')}")
    if c.get("expect_interrupt", False) != interrupted:
        failed.append(f"interrupt={interrupted}")
    if not all(t in tools for t in c.get("expect_tools", [])):
        failed.append(f"missing_tools:{tools}")
    if any(t in tools for t in c.get("forbid_tools", [])):
        failed.append(f"FORBIDDEN_tool_called:{tools}")
    if any(t in tools_before for t in c.get("forbid_before_resume", [])):
        failed.append(f"FORBIDDEN_before_approval:{tools_before}")
    if "reply_contains" in c and not any(k.lower() in reply.lower() for k in c["reply_contains"]):
        failed.append(f"reply_missing_any_of:{c['reply_contains']}")
    return failed


async def suite_e2e(cases):
    graph = build_graph()
    results, failures = [], []
    for c in cases:  # sequential: cases share one database
        try:
            failed = await run_e2e_case(graph, c)
        except Exception as e:  # noqa: BLE001
            failed = [f"exception:{type(e).__name__}:{e}"]
        results.append(not failed)
        if failed:
            failures.append(f"{c['input']} -> {failed}")
    return {"e2e_pass": sum(results) / len(results)}, failures


async def suite_rag(cases):
    async def one(c):
        r = await rag_answer(c["question"])
        if not c.get("answerable", True):
            return {"abstain_ok": not r["confident"]}
        j = await gateway.ainvoke("judge", JUDGE_PROMPT.format(
            q=c["question"], ref=c["reference"], ctx="\n".join(r["docs"]) or "(none)", ans=r["answer"]),
            schema=Judge, tag="judge")
        return {"score": j.correctness, "faithful": j.faithful, "why": j.reason, "q": c["question"]}

    res = await gather(one, cases, limit=2)
    qa = [r for r in res if isinstance(r, dict) and "score" in r]
    ab = [r for r, c in zip(res, cases) if not c.get("answerable", True)]
    metrics, failures = {}, []
    if qa:
        metrics["rag_correctness"] = sum(r["score"] for r in qa) / len(qa)
        metrics["rag_faithfulness"] = sum(r["faithful"] for r in qa) / len(qa)
        failures += [f"{r['q']} -> score={r['score']} faithful={r['faithful']}: {r['why']}"
                     for r in qa if r["score"] < 4 or not r["faithful"]]
    if ab:
        metrics["rag_abstain_acc"] = sum(isinstance(r, dict) and r["abstain_ok"] for r in ab) / len(ab)
    failures += [f"error: {r}" for r in res if isinstance(r, Exception)]
    return metrics, failures


SUITES = {"guardrail": suite_guardrail, "routing": suite_routing, "e2e": suite_e2e, "rag": suite_rag}


def reset_db():
    import psycopg
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
        conn.execute((HERE / "schema.sql").read_text())
    print("database reset and reseeded")


async def main(args):
    if args.reset_db:
        reset_db()
    cases = load_cases()
    all_metrics, all_failures = {}, {}
    for name in args.suite:
        metrics, failures = await SUITES[name](cases.get(name, []))
        all_metrics.update(metrics)
        all_failures[name] = failures

    print("\n=== EVAL RESULTS ===")
    bad = False
    for k, v in all_metrics.items():
        ok = v >= THRESHOLDS[k]
        bad |= not ok
        print(f"{'PASS' if ok else 'FAIL'}  {k:18s} {v:6.2f}  (threshold {THRESHOLDS[k]})")
    for name, fl in all_failures.items():
        for f in fl:
            print(f"  [{name}] {f}")
    rep = gateway.report()
    print(f"\nLLM cost this run: ${rep['total_cost_usd']}  (set MODEL_PRICES to enable cost tracking)")
    (HERE / "evals" / "report.json").write_text(json.dumps(
        {"metrics": all_metrics, "failures": all_failures, "gateway": rep}, indent=2))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+", choices=list(SUITES), default=list(SUITES))
    ap.add_argument("--reset-db", action="store_true")
    asyncio.run(main(ap.parse_args()))
