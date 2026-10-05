# Customer Support AI — LangGraph + MCP + Gateway + Self-correcting RAG + Evals

```
USER -> API (auth: customer_id) -> guardrail -> supervisor
          order agent ──► MCP tools ─┐
          refund agent ─► MCP tools ─┼─► Postgres (only the MCP server touches it)
          knowledge ───► self-correcting RAG ─► Chroma
          human ───────► ticket
                    └──────────► decision (deterministic rules)
                                   ├─ normal ─► respond / execute_refund
                                   ├─ risky ──► HITL (interrupt) ─► human approves/rejects
                                   └─ unsure ─► handoff ticket
every LLM call ─► gateway (tiers, fallback, retries, breaker, cache, budget, metrics)
every change ───► evals (guardrail, routing, e2e, RAG) with CI thresholds
```

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env            # add keys, DATABASE_URL
psql "$DATABASE_URL" -f schema.sql
python rag.py ingest            # build the vector index (needs OPENAI_API_KEY for embeddings)
python support_graph.py         # demo run (refund on a $450 order -> HITL -> approved)
uvicorn app:app --reload        # API
python evals.py --reset-db      # run all evals
```

## What each new piece does
| File | Role |
|---|---|
| `mcp_server.py` | Postgres tools: `get_order`, `update_address`, `check_refund_eligibility`, `create_refund`, `create_ticket`. Enforces ownership, eligibility, idempotency server-side. |
| `mcp_client.py` | Calls tools; `customer_id` is injected by code from auth, never by the LLM. |
| `gateway.py` | Tiers `fast/smart/judge`, provider fallback, retries + backoff, circuit breaker, timeout, cache for classifier calls, budget cap, card redaction, tokens/cost/latency in `/metrics`. |
| `rag.py` | retrieve → grade docs → rewrite query & retry → generate → check groundedness/answerability → regenerate or abstain. Abstain triggers a human handoff. |
| `evals.py` + `evals/golden.jsonl` | Guardrail & routing accuracy, end-to-end assertions (incl. *forbidden* tool calls and "no refund before approval"), RAG correctness/faithfulness (LLM judge) and abstention. Exit code 1 if below thresholds. |

## Notes / limits
- Set `MODEL_PRICES` to get real cost numbers; otherwise cost shows 0.
- `MemorySaver` and the in-memory `PENDING` dict are for dev. Use `PostgresSaver` and a `pending_reviews` table in production.
- Auth headers in `app.py` are placeholders for real JWT auth.
- Grow `golden.jsonl` from real transcripts; add each production failure as a new case.
- The code was syntax-checked but not run end-to-end (no network/DB/API keys in my sandbox), so expect to fix small integration issues on first run (library versions, model names).
- Optional next steps: LangSmith/Langfuse tracing, a LiteLLM proxy behind `gateway.py`, rate limiting per customer, a human-agent review UI.
