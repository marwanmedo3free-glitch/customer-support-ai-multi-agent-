"""
Self-correcting RAG (CRAG / Self-RAG style), built as a LangGraph subgraph.

  retrieve -> grade_docs --(none relevant)--> rewrite_query -> retrieve   (max MAX_RETRIEVALS)
                 |
                 v
              generate -> check_answer
                           |- grounded & answers question  -> finalize (confident)
                           |- not grounded                 -> generate again with feedback (max MAX_GENERATIONS)
                           |- doesn't answer the question  -> rewrite_query -> retrieve
                           '- out of retries                -> finalize (NOT confident -> caller hands off to a human)

Usage:
    python rag.py ingest            # build the vector index from data/policies/*.md
    python rag.py "what is the return window?"
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from gateway import gateway

MAX_RETRIEVALS = 3   # first try + 2 rewrites
MAX_GENERATIONS = 2
DATA_DIR = Path(__file__).parent / "data" / "policies"
CHROMA_DIR = os.getenv("CHROMA_DIR", str(Path(__file__).parent / ".chroma"))
FALLBACK = "I'm not certain about that, so I'd rather not guess. Let me connect you with a team member."


# ---------- vector store ----------
def _store():
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return Chroma(
        collection_name="policies",
        embedding_function=GoogleGenerativeAIEmbeddings(model=os.getenv("EMBED_MODEL", "gemini-embedding-001")),
        persist_directory=CHROMA_DIR,
    )


def ingest():
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = [Document(page_content=p.read_text(), metadata={"source": p.name}) for p in sorted(DATA_DIR.glob("*.md"))]
    chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80).split_documents(docs)
    ids = [f"{c.metadata['source']}-{i}" for i, c in enumerate(chunks)]  # deterministic -> re-ingest is an upsert
    _store().add_documents(chunks, ids=ids)
    print(f"indexed {len(chunks)} chunks from {len(docs)} files")


_retriever_cache = None


def _retriever():
    global _retriever_cache
    if _retriever_cache is None:
        _retriever_cache = _store().as_retriever(search_kwargs={"k": 4})
    return _retriever_cache


# ---------- structured outputs ----------
class Grades(BaseModel):
    relevant: list[bool]


class Check(BaseModel):
    grounded: bool
    answers_question: bool
    issues: str


# ---------- graph ----------
class RAGState(TypedDict, total=False):
    question: str
    query: str
    docs: list
    sources: list
    answer: str
    feedback: str
    grounded: bool
    answers: bool
    confident: bool
    retrieval_tries: int
    gen_tries: int


def _context(s: RAGState) -> str:
    return "\n\n".join(f"({src}) {d}" for src, d in zip(s["sources"], s["docs"]))


async def retrieve(s: RAGState):
    q = s.get("query") or s["question"]
    docs = await _retriever().ainvoke(q)
    return {
        "query": q,
        "docs": [d.page_content for d in docs],
        "sources": [d.metadata.get("source", "?") for d in docs],
        "retrieval_tries": s.get("retrieval_tries", 0) + 1,
        "feedback": "",
    }


async def grade_docs(s: RAGState):
    if not s["docs"]:
        return {}
    numbered = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(s["docs"]))
    g = await gateway.ainvoke(
        "fast",
        f"Question: {s['question']}\n\nFor each passage below, say whether it contains information useful "
        f"for answering the question. Return exactly one boolean per passage, in order.\n\n{numbered}",
        schema=Grades, tag="rag_grade",
    )
    if len(g.relevant) != len(s["docs"]):  # malformed grading -> keep everything rather than lose recall
        return {}
    keep = [i for i, ok in enumerate(g.relevant) if ok]
    return {"docs": [s["docs"][i] for i in keep], "sources": [s["sources"][i] for i in keep]}


def after_grade(s: RAGState) -> str:
    if s["docs"]:
        return "generate"
    return "rewrite_query" if s["retrieval_tries"] < MAX_RETRIEVALS else "finalize"


async def rewrite_query(s: RAGState):
    q = await gateway.ainvoke(
        "fast",
        "Rewrite this customer question as a short search query for a store-policy knowledge base. "
        "Use different keywords than the previous query. Output only the query.\n\n"
        f"Question: {s['question']}\nPrevious query: {s['query']}",
        tag="rag_rewrite",
    )
    return {"query": q.strip().strip('"')}


async def generate(s: RAGState):
    fix = f"\nYour previous draft had this problem; fix it: {s['feedback']}" if s.get("feedback") else ""
    ans = await gateway.ainvoke(
        "smart",
        "You are a customer-support assistant. Answer ONLY from the context. If the context does not contain "
        "the answer, say you don't know. Be concise and cite the source file name in parentheses."
        f"{fix}\n\nContext:\n{_context(s)}\n\nQuestion: {s['question']}",
        tag="rag_generate",
    )
    return {"answer": ans, "gen_tries": s.get("gen_tries", 0) + 1}


async def check_answer(s: RAGState):
    c = await gateway.ainvoke(
        "fast",
        f"Context:\n{_context(s)}\n\nQuestion: {s['question']}\n\nAnswer: {s['answer']}\n\n"
        "grounded = every claim in the answer is supported by the context. "
        "answers_question = the answer directly addresses the question (an honest \"I don't know\" does NOT). "
        "issues = a short description of what is wrong, or an empty string.",
        schema=Check, tag="rag_check",
    )
    return {"grounded": c.grounded, "answers": c.answers_question, "feedback": c.issues}


def after_check(s: RAGState) -> str:
    if s["grounded"] and s["answers"]:
        return "finalize"
    if not s["grounded"] and s["gen_tries"] < MAX_GENERATIONS:
        return "generate"
    if not s["answers"] and s["retrieval_tries"] < MAX_RETRIEVALS:
        return "rewrite_query"
    return "finalize"


async def finalize(s: RAGState):
    confident = bool(s.get("grounded") and s.get("answers"))
    return {"confident": confident, "answer": s["answer"] if confident else FALLBACK}


def build_rag_graph():
    g = StateGraph(RAGState)
    for name, fn in [("retrieve", retrieve), ("grade_docs", grade_docs), ("rewrite_query", rewrite_query),
                     ("generate", generate), ("check_answer", check_answer), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_docs")
    g.add_conditional_edges("grade_docs", after_grade, ["generate", "rewrite_query", "finalize"])
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", "check_answer")
    g.add_conditional_edges("check_answer", after_check, ["generate", "rewrite_query", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()


_graph = None


async def rag_answer(question: str) -> dict:
    global _graph
    _graph = _graph or build_rag_graph()
    out = await _graph.ainvoke({"question": question})
    return {
        "answer": out["answer"], "confident": out["confident"],
        "docs": out.get("docs", []), "sources": sorted(set(out.get("sources", []))),
        "retrieval_tries": out.get("retrieval_tries", 0), "gen_tries": out.get("gen_tries", 0),
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest()
    else:
        print(asyncio.run(rag_answer(" ".join(sys.argv[1:]) or "What is the return window?")))
