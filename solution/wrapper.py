"""Phase 1 -- OBSERVABILITY ONLY.

This wrapper is a wiretap: it lets every request flow through to the agent
UNCHANGED, but on the way back it writes down everything the agent tried to
hide (latency, tokens, cost, tool calls, loops, PII). No mitigation yet --
that comes in Phase 3. The agent is silent and run_output.json is lean, so
this log file is the ONLY place these signals exist.

Output: one JSON line per request in logs/<date>.log  (via telemetry.logger).
Read it afterwards to compute P95 latency, total cost, loop/overuse/PII rates,
and quality drift across turn_index.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter

# The sim loads this file directly; make sure the lab root (which holds the
# telemetry/ package) is importable no matter what the working directory is.
_LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LAB_ROOT not in sys.path:
    sys.path.insert(0, _LAB_ROOT)

from telemetry.logger import logger          # one JSON line per event -> logs/<date>.log
from telemetry.cost import cost_from_usage   # token usage -> USD
from telemetry.redact import redact          # (text) -> (masked_text, num_pii_found)


def mitigate(call_next, question, config, context):
    # --- run the black box, timing it ourselves ---------------------------
    t0 = time.time()
    try:
        result = call_next(question, config)          # the ONLY door to the agent
    except Exception as exc:                           # never let our wiretap crash a run
        logger.log_event("CALL_EXCEPTION", {
            "qid": context.get("qid"),
            "session": context.get("session_id"),
            "turn": context.get("turn_index"),
            "error": repr(exc),
        })
        raise
    wall_ms = int((time.time() - t0) * 1000)

    # --- pull the hidden truth out of the result --------------------------
    meta   = result.get("meta", {}) or {}
    usage  = meta.get("usage", {}) or {}
    tools  = meta.get("tools_used", []) or []
    trace  = result.get("trace", []) or []
    answer = result.get("answer") or ""
    model  = meta.get("model", "") or ""

    # tool patterns: how many times was each tool called?
    tool_counts = Counter(tools)
    max_tool_repeat = max(tool_counts.values()) if tool_counts else 0   # loop signal
    distinct_tools = len(tool_counts)

    # error / tool-failure signals: this binary puts them inside step["observation"].
    #   observation.error == "upstream_unavailable" -> transient injected failure (error_spike, retryable)
    #   observation.error in ("item_not_found","destination_not_served") -> legitimate grounding case
    #   check_stock observation in_stock == False -> forced out-of-stock (catalog_override / tool_failure)
    transient_errors = 0      # error_spike (retryable)
    grounding_errors = 0      # not found / not served (agent should refuse, not fabricate)
    forced_oos = 0            # catalog_override tool_failure
    for step in trace:
        if not isinstance(step, dict):
            continue
        obs = step.get("observation")
        if isinstance(obs, dict):
            err = obs.get("error")
            if err == "upstream_unavailable":
                transient_errors += 1
            elif err in ("item_not_found", "destination_not_served"):
                grounding_errors += 1
            if step.get("tool") == "check_stock" and obs.get("found") and obs.get("in_stock") is False:
                forced_oos += 1
    trace_errors = transient_errors + grounding_errors

    # PII signal: did the agent echo an email/phone/card into the answer?
    _, n_pii = redact(answer)

    # --- write it all down (THIS is the observability) --------------------
    logger.log_event("CALL", {
        "qid":          context.get("qid"),
        "session":      context.get("session_id"),
        "turn":         context.get("turn_index"),
        "status":       result.get("status"),
        # latency
        "wall_ms":      wall_ms,
        "latency_ms":   meta.get("latency_ms"),
        # cost / tokens
        "prompt_tokens":     usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens":      usage.get("total_tokens"),
        "cost_usd":     cost_from_usage(model, usage),
        "model":        model,
        # agent effort
        "steps":        result.get("steps"),
        "n_tools":      len(tools),
        "distinct_tools": distinct_tools,
        "max_tool_repeat": max_tool_repeat,   # >1 = overuse; large = infinite_loop
        "tool_counts":  dict(tool_counts),
        "trace_errors":     trace_errors,     # total failed tool calls
        "transient_errors": transient_errors, # upstream_unavailable -> error_spike (retryable)
        "grounding_errors": grounding_errors, # not_found / not_served
        "forced_oos":       forced_oos,       # macbook OOS via catalog_override -> tool_failure
        # quality / safety
        "answer_len":   len(answer),
        "has_answer":   bool(answer.strip()),
        "n_pii":        n_pii,                # > 0 = pii_leak
    })

    # Phase 1 is a wiretap: return the result UNCHANGED.
    return result
