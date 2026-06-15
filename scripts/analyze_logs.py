"""Read logs/<date>.log (CALL events from solution/wrapper.py) and print a
Phase-1 diagnostic summary: latency percentiles, cost, and per-fault rates.

Usage:  python scripts/analyze_logs.py [path/to/logfile]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict


def percentile(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[i]


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        logs = sorted(glob.glob(os.path.join("logs", "*.log")))
        if not logs:
            print("no log files in logs/ -- run the sim first")
            return
        path = logs[-1]

    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("event") == "CALL":
                rows.append(obj["data"])

    n = len(rows)
    print(f"\n=== analyzed {n} CALL events from {path} ===\n")
    if not n:
        return

    # ---- latency ----
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    print("LATENCY (ms):  P50={}  P95={}  P99={}  max={}".format(
        percentile(lat, 50), percentile(lat, 95), percentile(lat, 99), max(lat) if lat else None))

    # ---- cost / tokens ----
    total_cost = sum(r.get("cost_usd") or 0 for r in rows)
    total_tok = sum(r.get("total_tokens") or 0 for r in rows)
    print("COST:          total=${:.4f}   tokens={}   avg_tokens/req={:.0f}".format(
        total_cost, total_tok, total_tok / n))

    # ---- status distribution ----
    status = defaultdict(int)
    for r in rows:
        status[r.get("status")] += 1
    print("STATUS:        " + "  ".join(f"{k}={v}" for k, v in sorted(status.items())))

    # ---- fault rates ----
    def rate(pred):
        c = sum(1 for r in rows if pred(r))
        return c, 100.0 * c / n

    loops = rate(lambda r: (r.get("max_tool_repeat") or 0) >= 3)
    overuse = rate(lambda r: (r.get("n_tools") or 0) > 3 or (r.get("max_tool_repeat") or 0) >= 2)
    transient = rate(lambda r: (r.get("transient_errors") or 0) > 0)
    toolfail = rate(lambda r: (r.get("forced_oos") or 0) > 0)
    pii = rate(lambda r: (r.get("n_pii") or 0) > 0)
    noans = rate(lambda r: not r.get("has_answer"))
    maxsteps = rate(lambda r: r.get("status") == "max_steps")
    # error_spike rate measured per tool-call, not per request
    tot_tool = sum(r.get("n_tools") or 0 for r in rows)
    tot_transient = sum(r.get("transient_errors") or 0 for r in rows)
    print("INFINITE_LOOP: {} ({:.1f}%)  (max_tool_repeat>=3)".format(*loops))
    print("MAX_STEPS:     {} ({:.1f}%)".format(*maxsteps))
    print("TOOL_OVERUSE:  {} ({:.1f}%)".format(*overuse))
    print("ERROR_SPIKE:   {}/{} tool calls failed transiently = {:.1f}%  ({} reqs hit one)".format(
        tot_transient, tot_tool, 100.0 * tot_transient / max(1, tot_tool), transient[0]))
    print("TOOL_FAILURE:  {} ({:.1f}%)  (forced OOS via catalog_override)".format(*toolfail))
    print("PII_LEAK:      {} ({:.1f}%)".format(*pii))
    print("NO_ANSWER:     {} ({:.1f}%)".format(*noans))

    # ---- drift across turns ----
    by_turn = defaultdict(list)
    for r in rows:
        by_turn[r.get("turn")].append(r)
    print("\nDRIFT by turn (avg tokens, %no-answer, %loop):")
    for t in sorted(k for k in by_turn if k is not None):
        g = by_turn[t]
        avg_tok = sum(r.get("total_tokens") or 0 for r in g) / len(g)
        pna = 100.0 * sum(1 for r in g if not r.get("has_answer")) / len(g)
        plp = 100.0 * sum(1 for r in g if (r.get("max_tool_repeat") or 0) >= 3) / len(g)
        print(f"  turn {t:>2}: n={len(g):<3} avg_tokens={avg_tok:>8.0f}  no_answer={pna:4.0f}%  loop={plp:4.0f}%")


if __name__ == "__main__":
    main()
