"""
LLM gateway: every model call in the system goes through here.

  * tiers (fast / smart / judge) instead of hard-coded model names
  * provider fallback chain per tier, retries with backoff, per-model circuit breaker
  * timeout, response cache for structured (classification) calls, budget cap
  * card-number redaction on prompts, per-call metrics (tokens, cost, latency)

In-process on purpose (zero infra). To move it out of process later, point the same
tiers at a LiteLLM proxy / Portkey / Cloudflare AI Gateway and keep this interface.
"""
import asyncio
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from typing import Optional, Type

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

load_dotenv()

ROUTES = {
    "fast": [os.getenv("FAST_PRIMARY", "openai/gpt-oss-20b"), os.getenv("FAST_FALLBACK", "gemini-2.5-flash")],
    "smart": [os.getenv("SMART_PRIMARY", "openai/gpt-oss-20b"), os.getenv("SMART_FALLBACK", "gemini-2.5-flash")],
    # judge: ideally a different model family/strength than the generator, to reduce self-preference bias
    "judge": [os.getenv("JUDGE_PRIMARY", "openai/gpt-oss-20b"), os.getenv("JUDGE_FALLBACK", "gemini-2.5-flash")],
}


class GatewayError(Exception):
    pass


class BudgetExceeded(GatewayError):
    pass


_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")


def redact(text: str) -> str:
    return _CARD_RE.sub("[CARD]", text)


def _text(msg) -> str:
    c = msg.content
    return c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))


class LLMGateway:
    def __init__(self, routes=None, cache_ttl=300, retries=1, timeout=45.0, budget_usd: Optional[float] = None):
        self.routes = routes or ROUTES
        self.cache_ttl, self.retries, self.timeout, self.budget_usd = cache_ttl, retries, timeout, budget_usd
        self.prices = json.loads(os.getenv("MODEL_PRICES", "{}"))  # {"model": [usd_in_per_M, usd_out_per_M]}
        self.spent = 0.0
        self.calls: list[dict] = []
        self._models, self._cache = {}, {}
        self._breaker = defaultdict(lambda: {"fails": 0, "open_until": 0.0})

    # ---- internals ----
    def _model(self, spec: str):
        if spec not in self._models:
            # Tip: add temperature=0 here (if your models accept it) for more stable classifiers/evals.
            self._models[spec] = init_chat_model(spec)
        return self._models[spec]

    def _cost(self, spec: str, tin: int, tout: int) -> float:
        p = self.prices.get(spec.split(":", 1)[1])
        return (tin * p[0] + tout * p[1]) / 1e6 if p else 0.0

    async def _call(self, spec: str, prompt: str, schema: Optional[Type[BaseModel]]):
        model = self._model(spec)
        if schema:
            out = await model.with_structured_output(schema, include_raw=True).ainvoke(prompt)
            if out.get("parsed") is None:
                raise ValueError(f"unparseable structured output: {out.get('parsing_error')}")
            raw, result = out["raw"], out["parsed"]
        else:
            raw = await model.ainvoke(prompt)
            result = _text(raw)
        u = getattr(raw, "usage_metadata", None) or {}
        return result, u.get("input_tokens", 0), u.get("output_tokens", 0)

    def _log(self, **kw):
        self.calls.append(kw)

    # ---- public API ----
    async def ainvoke(self, tier: str, prompt: str, *, schema: Optional[Type[BaseModel]] = None,
                      cache: Optional[bool] = None, tag: str = ""):
        """Returns a pydantic object if schema is given, else a string."""
        if self.budget_usd is not None and self.spent >= self.budget_usd:
            raise BudgetExceeded(f"LLM budget of ${self.budget_usd} exhausted")
        prompt = redact(prompt)
        use_cache = (schema is not None) if cache is None else cache
        key = hashlib.sha256(json.dumps([tier, prompt, schema.__name__ if schema else None]).encode()).hexdigest()
        if use_cache and key in self._cache and time.time() - self._cache[key][0] < self.cache_ttl:
            self._log(tier=tier, model="cache", tag=tag, ms=0, tin=0, tout=0, cost=0.0, cached=True)
            return self._cache[key][1]

        errors = []
        for spec in self.routes[tier]:
            br = self._breaker[spec]
            if time.time() < br["open_until"]:
                errors.append(f"{spec}: circuit open")
                continue
            for attempt in range(self.retries + 1):
                t0 = time.perf_counter()
                try:
                    result, tin, tout = await asyncio.wait_for(self._call(spec, prompt, schema), self.timeout)
                except Exception as e:  # noqa: BLE001 - any provider error triggers retry/fallback
                    errors.append(f"{spec}: {type(e).__name__}: {str(e)[:120]}")
                    br["fails"] += 1
                    if br["fails"] >= 3:
                        br["open_until"] = time.time() + 60
                        break
                    await asyncio.sleep(0.5 * 2 ** attempt)
                    continue
                br["fails"] = 0
                cost = self._cost(spec, tin, tout)
                self.spent += cost
                self._log(tier=tier, model=spec, tag=tag, ms=int((time.perf_counter() - t0) * 1000),
                          tin=tin, tout=tout, cost=cost, cached=False)
                if use_cache:
                    self._cache[key] = (time.time(), result)
                return result
        raise GatewayError("all providers failed: " + " | ".join(errors))

    def report(self) -> dict:
        by = defaultdict(lambda: {"calls": 0, "cache_hits": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "lat": []})
        for c in self.calls:
            r = by[f"{c['tier']}/{c['model']}"]
            r["calls"] += 1
            r["cache_hits"] += c["cached"]
            r["tokens_in"] += c["tin"]
            r["tokens_out"] += c["tout"]
            r["cost_usd"] += c["cost"]
            if not c["cached"]:
                r["lat"].append(c["ms"])
        out = {}
        for k, r in by.items():
            lat = sorted(r.pop("lat"))
            r["p50_ms"] = lat[len(lat) // 2] if lat else 0
            r["p95_ms"] = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0
            r["cost_usd"] = round(r["cost_usd"], 6)
            out[k] = r
        return {"total_cost_usd": round(self.spent, 6), "models": out}


gateway = LLMGateway(budget_usd=float(os.environ["LLM_BUDGET_USD"]) if os.getenv("LLM_BUDGET_USD") else None)