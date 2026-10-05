"""
FastAPI layer.  Run:  python app.py      (then open http://localhost:8000)

Auth here is a stand-in: X-Customer-Id for customers, X-Agent-Token for human agents.
Replace with real JWT/session auth -- customer_id MUST come from the verified identity.
"""
from dotenv import load_dotenv

load_dotenv()  # first: gateway.py / support_graph.py read env vars when they are imported

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel

from gateway import gateway
from support_graph import build_graph

app = FastAPI(title="Customer Support AI")
graph = build_graph()
PENDING: dict[str, dict] = {}   # thread_id -> interrupt payload (use a DB table in production)


class ChatIn(BaseModel):
    message: str
    thread_id: str | None = None


class ApproveIn(BaseModel):
    thread_id: str
    approved: bool
    feedback: str | None = None


def _agent_only(token: str | None):
    if not token or token != os.getenv("AGENT_TOKEN", "change-me"):
        raise HTTPException(403, "agent token required")


def _result(out: dict, thread_id: str) -> dict:
    if "__interrupt__" in out:
        PENDING[thread_id] = out["__interrupt__"][0].value
        return {"thread_id": thread_id, "status": "pending_human_approval",
                "reply": "I've sent this to a team member for review. You'll get an update shortly."}
    PENDING.pop(thread_id, None)
    return {"thread_id": thread_id, "status": "done", "reply": out["messages"][-1].content,
            "trace": out.get("trace", [])[-8:]}


@app.post("/api/chat")
async def chat(body: ChatIn, x_customer_id: int = Header(...)):
    thread_id = body.thread_id or f"{x_customer_id}:{uuid.uuid4().hex[:12]}"
    if not thread_id.startswith(f"{x_customer_id}:"):
        raise HTTPException(403, "thread does not belong to this customer")
    cfg = {"configurable": {"thread_id": thread_id}}
    out = await graph.ainvoke({"messages": [("user", body.message[:2000])], "customer_id": x_customer_id}, cfg)
    return _result(out, thread_id)


@app.get("/api/status/{thread_id}")
async def status(thread_id: str, x_customer_id: int = Header(...)):
    if not thread_id.startswith(f"{x_customer_id}:"):
        raise HTTPException(403, "thread does not belong to this customer")
    st = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    msgs = st.values.get("messages", [])
    return {"waiting_for_human": bool(st.next), "last_message": msgs[-1].content if msgs else None}


@app.get("/api/pending")
async def pending(x_agent_token: str | None = Header(None)):
    _agent_only(x_agent_token)
    return PENDING


@app.post("/api/approve")
async def approve(body: ApproveIn, x_agent_token: str | None = Header(None)):
    _agent_only(x_agent_token)
    if body.thread_id not in PENDING:
        raise HTTPException(404, "no pending review for this thread")
    cfg = {"configurable": {"thread_id": body.thread_id}}
    out = await graph.ainvoke(Command(resume={"approved": body.approved, "feedback": body.feedback}), cfg)
    return _result(out, body.thread_id)


@app.get("/metrics")
async def metrics():
    return gateway.report()


@app.get("/health")
async def health():
    return {"ok": True, "features": ["mcp", "gateway", "self_correcting_rag", "hitl", "evals"]}


# ---------- web UI (same origin as the API, so no CORS setup is needed) ----------
BASE = Path(__file__).parent
TEMPLATES = BASE / "templates"          # index.html
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")   # style.css, script.js


@app.get("/", include_in_schema=False)
async def customer_page():
    return FileResponse(TEMPLATES / "index.html")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")   # 0.0.0.0 is not browsable on Windows; use localhost
    port = int(os.getenv("PORT", "8000"))
    print(f"\n  Open http://localhost:{port}\n")
    uvicorn.run("app:app", host=host, port=port, reload=True)
