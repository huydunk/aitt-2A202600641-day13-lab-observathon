"""Observability + mitigation layer (Phase 1 + Phase 3).

OBSERVABILITY (Phase 1): every request is timed and one JSON line per request is
written to logs/<date>.log -- latency, tokens, cost, tool counts, loops, errors,
forced-OOS and PII. The agent is silent, so this log is the only place these
signals exist. Read it with scripts/analyze_logs.py.

MITIGATIONS (Phase 3), applied around the black box -- chosen to NOT duplicate the
config knobs (retry/cache/normalize are already on in config.json), so we add only
unique, high-value, low-risk fixes:
  1. injection note sanitization -- strip instruction/price-override clauses hidden
     in order notes BEFORE the agent sees them (private-phase defense).
  2. arithmetic verify-and-correct -- recompute the total deterministically from the
     tool data in the trace and fix the answer's total line (protects correctness).
     Fires ONLY when the order is unambiguously answerable (single in-stock product,
     valid coupon if any, shipping served); otherwise the answer is left untouched.
  3. PII redaction backstop -- mask any email/phone the model still echoes.
"""
from __future__ import annotations

import os
import re
import sys
import time
import unicodedata
from collections import Counter

# The sim loads this file directly; make sure the lab root (telemetry/ package)
# is importable regardless of the working directory.
_LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LAB_ROOT not in sys.path:
    sys.path.insert(0, _LAB_ROOT)

from telemetry.logger import logger          # one JSON line per event -> logs/<date>.log
from telemetry.cost import cost_from_usage   # token usage -> USD
from telemetry.redact import redact          # (text) -> (masked_text, num_pii_found)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


_QTY_RE = re.compile(r"\b(?:mua|x)\s*(\d+)", re.IGNORECASE)
# tolerate ":", "*", markdown and spaces between the label and the number
_TOTAL_RE = re.compile(r"(?:Tong\s*cong|Tổng\s*cộng)[^\d]{0,15}([\d.,\s]+?)\s*VND", re.IGNORECASE)

_NOTE_MARKERS = ("ghi chu", "note", "luu y", "chu thich")
# clauses in a note that try to override price/total or give instructions
_INJECT_KEYWORDS = ("gia", "price", "tong", "total", "giam", "discount", "free", "mien phi",
                    "ignore", "bo qua", "thuc te", "thay vi", "instead", "set ", "dat gia", "chi con")


def _sanitize_question(q: str):
    """Drop instruction/price-override clauses hidden after an order-note marker.
    Returns (clean_question, num_clauses_dropped). Only touches the note tail."""
    low = _strip_diacritics(q).lower()
    cut = -1
    for m in _NOTE_MARKERS:
        i = low.find(m)
        if i != -1:
            cut = i if cut == -1 else min(cut, i)
    if cut == -1:
        return q, 0
    head, note = q[:cut], q[cut:]
    kept, dropped = [], 0
    for clause in re.split(r"[;.\n]", note):
        cl = _strip_diacritics(clause).lower()
        if any(k in cl for k in _INJECT_KEYWORDS):
            dropped += 1
            continue
        kept.append(clause)
    if dropped == 0:
        return q, 0
    return (head + " ".join(kept)).strip(), dropped


def _recompute_total(question: str, trace):
    """Deterministically recompute the order total from tool observations.
    Returns an int total, or None if the order is NOT unambiguously answerable
    (out of stock / not found / destination not served / invalid coupon /
    multiple products / transient tool error) -- in which case we never touch
    the answer."""
    unit_price = None
    available_qty = None
    pct = 0
    shipping = 0
    disc_called = False
    items = set()
    for step in trace:
        obs = step.get("observation") if isinstance(step, dict) else None
        if not isinstance(obs, dict):
            continue
        tool = step.get("tool")
        if tool == "check_stock":
            if obs.get("error"):
                return None                       # transient or item_not_found
            if obs.get("item"):
                items.add(obs.get("item"))
            if obs.get("found") is False:
                return None                       # not found -> refusal
            if obs.get("found") and obs.get("in_stock") is False:
                return None                       # out of stock -> refusal
            if obs.get("found") and obs.get("in_stock") and obs.get("unit_price_vnd") is not None:
                unit_price = obs.get("unit_price_vnd")
                if obs.get("quantity") is not None:
                    available_qty = obs.get("quantity")
        elif tool == "get_discount":
            disc_called = True
            if obs.get("error"):
                return None
            if obs.get("valid"):
                pct = obs.get("percent") or 0
            else:
                return None                       # invalid coupon -> let model decide
        elif tool == "calc_shipping":
            if obs.get("error"):
                return None                       # destination not served -> refusal
            shipping = obs.get("cost_vnd") or 0
    if unit_price is None or len(items) != 1:
        return None                               # no clean single-product price
    m = _QTY_RE.search(question)
    if not m:
        return None                               # no explicit quantity -> a price inquiry,
                                                  # NOT an order; never fabricate a total
    qty = int(m.group(1))
    if available_qty is not None and qty > available_qty:
        return None                               # not enough stock -> refusal, no total
    subtotal = unit_price * qty
    discounted = subtotal * (100 - pct) // 100
    return discounted + shipping


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def mitigate(call_next, question, config, context):
    # --- (1) injection defense: sanitize order notes before the agent sees them ---
    clean_q, dropped = _sanitize_question(question)

    # --- run the black box, timing it ourselves -------------------------------
    t0 = time.time()
    try:
        result = call_next(clean_q, config)       # the ONLY door to the agent
    except Exception as exc:                       # never let our wiretap crash a run
        logger.log_event("CALL_EXCEPTION", {
            "qid": context.get("qid"), "session": context.get("session_id"),
            "turn": context.get("turn_index"), "error": repr(exc),
        })
        raise
    wall_ms = int((time.time() - t0) * 1000)

    # --- pull the hidden truth out of the result ------------------------------
    meta   = result.get("meta", {}) or {}
    usage  = meta.get("usage", {}) or {}
    tools  = meta.get("tools_used", []) or []
    trace  = result.get("trace", []) or []
    answer = result.get("answer") or ""
    model  = meta.get("model", "") or ""

    tool_counts = Counter(tools)
    max_tool_repeat = max(tool_counts.values()) if tool_counts else 0
    distinct_tools = len(tool_counts)

    transient_errors = grounding_errors = forced_oos = 0
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

    # PII the model emitted (measured BEFORE we redact, for observability)
    _, n_pii = redact(answer)

    # --- (2) arithmetic verify-and-correct ------------------------------------
    arith_fixed = None
    recomputed = _recompute_total(clean_q, trace)
    if recomputed is not None and answer:
        m = _TOTAL_RE.search(answer)
        if m:
            stated = re.sub(r"[.,\s]", "", m.group(1))
            stated_int = int(stated) if stated.isdigit() else None
            if stated_int != recomputed:
                answer = answer[:m.start()] + ("Tong cong: %d VND" % recomputed) + answer[m.end():]
                arith_fixed = [stated_int, recomputed]
        else:
            answer = answer.rstrip() + ("\nTong cong: %d VND" % recomputed)
            arith_fixed = [None, recomputed]

    # --- (3) PII redaction backstop -------------------------------------------
    answer, n_pii_redacted = redact(answer)

    # --- write observability --------------------------------------------------
    logger.log_event("CALL", {
        "qid": context.get("qid"), "session": context.get("session_id"),
        "turn": context.get("turn_index"), "status": result.get("status"),
        "wall_ms": wall_ms, "latency_ms": meta.get("latency_ms"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost_usd": cost_from_usage(model, usage), "model": model,
        "steps": result.get("steps"), "n_tools": len(tools),
        "distinct_tools": distinct_tools, "max_tool_repeat": max_tool_repeat,
        "tool_counts": dict(tool_counts), "trace_errors": trace_errors,
        "transient_errors": transient_errors, "grounding_errors": grounding_errors,
        "forced_oos": forced_oos, "answer_len": len(answer),
        "has_answer": bool(answer.strip()), "n_pii": n_pii,
        # mitigation telemetry
        "note_clauses_dropped": dropped,
        "arith_fixed": arith_fixed,
        "pii_redacted": n_pii_redacted,
    })

    # return the (possibly corrected + redacted) answer
    result["answer"] = answer
    return result
