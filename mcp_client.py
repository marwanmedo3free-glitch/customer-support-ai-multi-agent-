"""
MCP client. The graph calls tools through call_tool() so that identity (customer_id)
is injected by code from the authenticated session, never chosen by the LLM.

Transport: stdio spawns mcp_server.py as a subprocess (fine for dev/eval).
Production: run the server separately and switch to
    {"transport": "streamable_http", "url": "http://mcp:8001/mcp"}
"""
import json
import os
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

_tools: dict | None = None


async def _load() -> dict:
    global _tools
    if _tools is None:
        client = MultiServerMCPClient({
            "support": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [os.path.join(os.path.dirname(__file__), "mcp_server.py")],
                "env": dict(os.environ),
            }
        })
        _tools = {t.name: t for t in await client.get_tools()}
    return _tools


def _parse(result):
    if isinstance(result, list):  # list of content blocks
        result = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in result)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"error": "bad_tool_output", "raw": result}
    return result


async def call_tool(name: str, **args) -> dict:
    tools = await _load()
    return _parse(await tools[name].ainvoke(args))
