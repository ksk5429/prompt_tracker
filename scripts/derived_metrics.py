"""
Tier 2 Intelligence Layer: compute derived metrics from raw extractions.
- Session fingerprints (tool distribution vectors → clusters)
- Cache efficiency timeseries
- Prompt analysis (length, complexity trends)
- Conversation flow patterns (tool sequences)
- Session quality scores (git output correlation)
- Attention allocation vs plan
"""
import csv
import math
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

OUT_DIR = Path(__file__).parent.parent / "data" / "processed"


def compute_session_fingerprints():
    """Cluster sessions by tool usage distribution."""
    tool_calls_path = OUT_DIR / "jsonl_tool_calls.csv"
    if not tool_calls_path.exists():
        print("[derived] no tool calls data, skipping fingerprints")
        return

    # build tool distribution per session
    session_tools = defaultdict(Counter)
    with open(tool_calls_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            session_tools[row["session_id"]][row["tool_name"]] += 1

    # define feature categories
    CATEGORIES = {
        "read_pct": {"Read", "Grep", "Glob"},
        "write_pct": {"Write", "Edit"},
        "shell_pct": {"Bash"},
        "orchestrate_pct": {"Agent", "TodoWrite", "ToolSearch"},
        "web_pct": {"WebFetch", "WebSearch"},
        "mcp_pct": set(),  # anything with mcp__
    }

    rows = []
    for sid, tools in session_tools.items():
        total = sum(tools.values())
        if total < 3:
            continue

        vec = {}
        for cat, names in CATEGORIES.items():
            if cat == "mcp_pct":
                count = sum(v for k, v in tools.items() if k.startswith("mcp__"))
            else:
                count = sum(tools.get(n, 0) for n in names)
            vec[cat] = round(count / total * 100, 1)

        # classify session type
        if vec["write_pct"] > 30:
            session_type = "coding"
        elif vec["read_pct"] > 40:
            session_type = "research"
        elif vec["orchestrate_pct"] > 25:
            session_type = "orchestration"
        elif vec["shell_pct"] > 50:
            session_type = "devops"
        elif vec["web_pct"] > 15:
            session_type = "web_research"
        else:
            session_type = "mixed"

        # dominant tool
        dominant = tools.most_common(1)[0][0] if tools else "none"

        rows.append({
            "session_id": sid,
            "total_tool_calls": total,
            "unique_tools": len(tools),
            "dominant_tool": dominant,
            "session_type": session_type,
            **vec,
        })

    fields = ["session_id", "total_tool_calls", "unique_tools", "dominant_tool",
              "session_type", "read_pct", "write_pct", "shell_pct",
              "orchestrate_pct", "web_pct", "mcp_pct"]
    _write(OUT_DIR / "session_fingerprints.csv", rows, fields)
    type_counts = Counter(r["session_type"] for r in rows)
    print(f"[derived] fingerprints: {len(rows)} sessions — {dict(type_counts)}")


def compute_cache_efficiency():
    """Cache hit rate over time."""
    tokens_path = OUT_DIR / "jsonl_tokens.csv"
    if not tokens_path.exists():
        print("[derived] no token data, skipping cache efficiency")
        return

    daily = defaultdict(lambda: {"cache_read": 0, "cache_create": 0, "input": 0, "output": 0, "requests": 0})
    with open(tokens_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            date = row["timestamp"][:10]
            if not date:
                continue
            d = daily[date]
            d["cache_read"] += int(row.get("cache_read_tokens", 0) or 0)
            d["cache_create"] += int(row.get("cache_creation_tokens", 0) or 0)
            d["input"] += int(row.get("input_tokens", 0) or 0)
            d["output"] += int(row.get("output_tokens", 0) or 0)
            d["requests"] += 1

    rows = []
    for date in sorted(daily):
        d = daily[date]
        total_input = d["cache_read"] + d["cache_create"] + d["input"]
        hit_rate = round(d["cache_read"] / total_input * 100, 1) if total_input > 0 else 0
        rows.append({
            "date": date,
            "cache_read_tokens": d["cache_read"],
            "cache_creation_tokens": d["cache_create"],
            "input_tokens": d["input"],
            "output_tokens": d["output"],
            "total_input_tokens": total_input,
            "cache_hit_rate": hit_rate,
            "requests": d["requests"],
        })

    fields = ["date", "cache_read_tokens", "cache_creation_tokens", "input_tokens",
              "output_tokens", "total_input_tokens", "cache_hit_rate", "requests"]
    _write(OUT_DIR / "cache_efficiency.csv", rows, fields)
    if rows:
        avg_hit = sum(r["cache_hit_rate"] for r in rows) / len(rows)
        print(f"[derived] cache efficiency: {len(rows)} days, avg hit rate {avg_hit:.1f}%")


def compute_prompt_analysis():
    """Analyze prompt patterns over time."""
    msgs_path = OUT_DIR / "jsonl_messages.csv"
    if not msgs_path.exists():
        print("[derived] no messages data, skipping prompt analysis")
        return

    daily = defaultdict(lambda: {"lengths": [], "words": [], "count": 0, "images": 0})
    with open(msgs_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            date = row["timestamp"][:10]
            if not date:
                continue
            d = daily[date]
            d["count"] += 1
            length = int(row.get("prompt_length", 0) or 0)
            words = int(row.get("prompt_words", 0) or 0)
            d["lengths"].append(length)
            d["words"].append(words)
            if row.get("has_image") == "True":
                d["images"] += 1

    rows = []
    for date in sorted(daily):
        d = daily[date]
        lengths = d["lengths"]
        words = d["words"]
        # filter out empty prompts (IDE context, system messages)
        nonzero_lengths = [l for l in lengths if l > 0]
        nonzero_words = [w for w in words if w > 0]

        rows.append({
            "date": date,
            "prompt_count": d["count"],
            "avg_prompt_length": round(sum(nonzero_lengths) / max(len(nonzero_lengths), 1), 0),
            "median_prompt_length": _median(nonzero_lengths),
            "max_prompt_length": max(nonzero_lengths) if nonzero_lengths else 0,
            "avg_prompt_words": round(sum(nonzero_words) / max(len(nonzero_words), 1), 1),
            "images_sent": d["images"],
            "empty_prompt_pct": round((len(lengths) - len(nonzero_lengths)) / max(len(lengths), 1) * 100, 1),
        })

    fields = ["date", "prompt_count", "avg_prompt_length", "median_prompt_length",
              "max_prompt_length", "avg_prompt_words", "images_sent", "empty_prompt_pct"]
    _write(OUT_DIR / "prompt_analysis.csv", rows, fields)
    print(f"[derived] prompt analysis: {len(rows)} days")


def compute_conversation_flows():
    """Extract tool call sequences (bigrams) per session."""
    tool_calls_path = OUT_DIR / "jsonl_tool_calls.csv"
    if not tool_calls_path.exists():
        return

    # load tool calls sorted by session + timestamp
    session_tools = defaultdict(list)
    with open(tool_calls_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            session_tools[row["session_id"]].append({
                "tool": row["tool_name"],
                "ts": row["timestamp"],
            })

    # compute bigrams and sequences
    bigram_counts = Counter()
    session_sequences = []

    for sid, calls in session_tools.items():
        calls.sort(key=lambda x: x["ts"])
        tools = [c["tool"] for c in calls]

        # bigrams
        for i in range(len(tools) - 1):
            bigram_counts[(tools[i], tools[i+1])] += 1

        # session opening pattern (first 5 tools)
        opener = "→".join(tools[:5]) if len(tools) >= 5 else "→".join(tools)
        session_sequences.append({
            "session_id": sid,
            "tool_count": len(tools),
            "opener_pattern": opener,
            "unique_tools": len(set(tools)),
            "most_repeated": Counter(tools).most_common(1)[0][0] if tools else "",
        })

    # write bigrams
    bigram_rows = [{"from_tool": k[0], "to_tool": k[1], "count": v}
                   for k, v in bigram_counts.most_common(100)]
    _write(OUT_DIR / "tool_bigrams.csv", bigram_rows, ["from_tool", "to_tool", "count"])

    # write session sequences
    _write(OUT_DIR / "session_sequences.csv", session_sequences,
           ["session_id", "tool_count", "opener_pattern", "unique_tools", "most_repeated"])

    print(f"[derived] flows: {len(bigram_rows)} bigrams, {len(session_sequences)} sequences")


def compute_session_quality():
    """
    Score sessions using 6 practical, meaningful components:

    1. OUTPUT (25pts) — Did the session produce artifacts?
       Write/Edit calls as fraction of total. A session that writes is more
       productive than one that only reads, regardless of git commits.

    2. IMPACT (20pts) — Did it produce lasting changes?
       Git commits + net lines added. Rewards sessions that ship code,
       but doesn't dominate the score like the old 40pt weight did.

    3. FOCUS (15pts) — Did the session stay on task?
       Single-project sessions score higher. Context-switching between
       5 projects in one session suggests thrashing, not productivity.

    4. CRAFT (15pts) — Was the work deliberate?
       Output/Input ratio (Write+Edit vs Read+Grep). High ratio = generating.
       Low ratio = consuming. Both are valid, but craft measures creation.
       Also rewards planning (TodoWrite) and orchestration (Agent).

    5. DEPTH (15pts) — How substantial was the session?
       Log-scaled tool calls. A 10-call session and a 200-call session both
       score, but the marginal value of the 500th call is near zero.

    6. PROMPT QUALITY (10pts) — Did the user give clear instructions?
       Average prompt length (nonzero). Longer, more structured prompts
       correlate with fewer correction cycles and better outcomes.
    """
    git_path = OUT_DIR / "git_session_join.csv"
    fp_path = OUT_DIR / "session_fingerprints.csv"
    meta_path = OUT_DIR / "jsonl_session_meta.csv"
    tool_path = OUT_DIR / "jsonl_tool_calls.csv"
    msgs_path = OUT_DIR / "jsonl_messages.csv"

    # load git commits per session
    git_by_session = defaultdict(lambda: {"commits": 0, "additions": 0, "deletions": 0})
    if git_path.exists():
        with open(git_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row.get("session_id", "")
                if sid:
                    g = git_by_session[sid]
                    g["commits"] += 1
                    g["additions"] += int(row.get("additions", 0) or 0)
                    g["deletions"] += int(row.get("deletions", 0) or 0)

    # load fingerprints
    fingerprints = {}
    if fp_path.exists():
        with open(fp_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fingerprints[row["session_id"]] = row

    # load session meta
    metas = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                metas[row["session_id"]] = row

    # load tool calls per session for output/input ratio
    session_tool_counts = defaultdict(Counter)
    if tool_path.exists():
        with open(tool_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                session_tool_counts[row["session_id"]][row["tool_name"]] += 1

    # load prompt data per session
    session_prompts = defaultdict(list)
    if msgs_path.exists():
        with open(msgs_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                length = int(row.get("prompt_length", 0) or 0)
                if length > 0:
                    session_prompts[row["session_id"]].append(length)

    OUTPUT_TOOLS = {"Write", "Edit"}
    INPUT_TOOLS = {"Read", "Grep", "Glob"}
    PLANNING_TOOLS = {"TodoWrite", "Agent", "ToolSearch"}

    rows = []
    for sid, fp in fingerprints.items():
        git = git_by_session.get(sid, {"commits": 0, "additions": 0, "deletions": 0})
        meta = metas.get(sid, {})
        tools = session_tool_counts.get(sid, Counter())
        prompts = session_prompts.get(sid, [])

        # duration
        duration_min = 0
        first_ts = meta.get("first_timestamp", "")
        last_ts = meta.get("last_timestamp", "")
        date = first_ts[:10] if first_ts else ""
        if first_ts and last_ts:
            try:
                t0 = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                duration_min = round((t1 - t0).total_seconds() / 60, 1)
            except ValueError:
                pass

        total_calls = int(fp.get("total_tool_calls", 0))
        unique_tools = int(fp.get("unique_tools", 0))

        # ── Component 1: OUTPUT (0-25) ──
        # What fraction of tool calls produced artifacts?
        output_calls = sum(tools.get(t, 0) for t in OUTPUT_TOOLS)
        output_pct = output_calls / max(total_calls, 1)
        # 50%+ output = max score. Scale linearly.
        output_score = round(min(output_pct / 0.50 * 25, 25), 1)

        # ── Component 2: IMPACT (0-20) ──
        # Git commits (10pts) + net lines (10pts)
        commit_pts = min(git["commits"] * 5, 10)  # 2 commits = max
        net_lines = git["additions"] + git["deletions"]
        lines_pts = min(math.log10(max(net_lines, 1)) * 3, 10) if net_lines > 0 else 0
        impact_score = round(commit_pts + lines_pts, 1)

        # ── Component 3: FOCUS (0-15) ──
        # Single-project focus. Penalize multi-project thrashing.
        # This uses the session's own project (always 1 from fingerprint).
        # But also check if tools operated on many different paths.
        # Simple proxy: unique_tools < 5 = focused, > 10 = scattered
        focus_score = round(min(15, 15 - max(unique_tools - 5, 0) * 1.5), 1)
        focus_score = max(focus_score, 3)  # floor at 3

        # ── Component 4: CRAFT (0-15) ──
        # Output/Input ratio + planning bonus
        input_calls = sum(tools.get(t, 0) for t in INPUT_TOOLS)
        planning_calls = sum(tools.get(t, 0) for t in PLANNING_TOOLS)
        # Craft = (output + planning) / (input + 1), scaled
        craft_ratio = (output_calls + planning_calls) / max(input_calls, 1)
        craft_score = round(min(craft_ratio * 5, 12), 1)
        # bonus for using Agent (orchestration sophistication)
        if tools.get("Agent", 0) > 0:
            craft_score = min(craft_score + 3, 15)
        craft_score = round(craft_score, 1)

        # ── Component 5: DEPTH (0-15) ──
        # Log-scaled tool calls. Diminishing returns past ~100 calls.
        if total_calls > 0:
            depth_score = round(min(math.log2(total_calls) * 2, 15), 1)
        else:
            depth_score = 0

        # ── Component 6: PROMPT QUALITY (0-10) ──
        # Average prompt length. Longer = more structured instructions.
        # Median of 102 chars, 90th pctile of 927 chars.
        avg_prompt = sum(prompts) / max(len(prompts), 1) if prompts else 0
        # Scale: 500+ chars avg = full score
        prompt_score = round(min(avg_prompt / 500 * 10, 10), 1)

        quality = round(output_score + impact_score + focus_score +
                        craft_score + depth_score + prompt_score, 1)

        rows.append({
            "session_id": sid,
            "date": date,
            "project": meta.get("project", ""),
            "session_type": fp.get("session_type", ""),
            "duration_min": duration_min,
            "total_tool_calls": total_calls,
            "commits": git["commits"],
            "additions": git["additions"],
            "deletions": git["deletions"],
            "quality_score": quality,
            "output_score": output_score,
            "impact_score": impact_score,
            "focus_score": focus_score,
            "craft_score": craft_score,
            "depth_score": depth_score,
            "prompt_score": prompt_score,
        })

    rows.sort(key=lambda x: x.get("date", ""))
    fields = ["session_id", "date", "project", "session_type", "duration_min",
              "total_tool_calls", "commits", "additions", "deletions",
              "quality_score", "output_score", "impact_score", "focus_score",
              "craft_score", "depth_score", "prompt_score"]
    _write(OUT_DIR / "session_quality.csv", rows, fields)
    if rows:
        avg_q = sum(r["quality_score"] for r in rows) / len(rows)
        # breakdown
        avg_o = sum(r["output_score"] for r in rows) / len(rows)
        avg_i = sum(r["impact_score"] for r in rows) / len(rows)
        avg_f = sum(r["focus_score"] for r in rows) / len(rows)
        avg_c = sum(r["craft_score"] for r in rows) / len(rows)
        avg_d = sum(r["depth_score"] for r in rows) / len(rows)
        avg_p = sum(r["prompt_score"] for r in rows) / len(rows)
        print(f"[derived] quality: {len(rows)} sessions, avg {avg_q:.1f}/100")
        print(f"  output={avg_o:.1f}/25  impact={avg_i:.1f}/20  focus={avg_f:.1f}/15  "
              f"craft={avg_c:.1f}/15  depth={avg_d:.1f}/15  prompt={avg_p:.1f}/10")


def compute_model_routing():
    """Analyze which models are used when and for what."""
    tokens_path = OUT_DIR / "jsonl_tokens.csv"
    fp_path = OUT_DIR / "session_fingerprints.csv"
    if not tokens_path.exists():
        return

    # load fingerprints
    fp_map = {}
    if fp_path.exists():
        with open(fp_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fp_map[row["session_id"]] = row.get("session_type", "unknown")

    # aggregate by model x session_type
    model_type = defaultdict(lambda: {"requests": 0, "output_tokens": 0})
    daily_model = defaultdict(lambda: {"requests": 0, "output_tokens": 0, "cache_read": 0})

    with open(tokens_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = row.get("model", "unknown")
            if model == "<synthetic>" or not model:
                continue
            sid = row["session_id"]
            stype = fp_map.get(sid, "unknown")
            date = row["timestamp"][:10]

            mt = model_type[(model, stype)]
            mt["requests"] += 1
            mt["output_tokens"] += int(row.get("output_tokens", 0) or 0)

            dm = daily_model[(date, model)]
            dm["requests"] += 1
            dm["output_tokens"] += int(row.get("output_tokens", 0) or 0)
            dm["cache_read"] += int(row.get("cache_read_tokens", 0) or 0)

    # model x type
    rows1 = [{"model": k[0], "session_type": k[1], **v} for k, v in model_type.items()]
    _write(OUT_DIR / "model_by_type.csv", rows1, ["model", "session_type", "requests", "output_tokens"])

    # daily model
    rows2 = [{"date": k[0], "model": k[1], **v} for k, v in sorted(daily_model.items())]
    _write(OUT_DIR / "daily_model_detail.csv", rows2, ["date", "model", "requests", "output_tokens", "cache_read"])

    print(f"[derived] model routing: {len(rows1)} model-type pairs, {len(rows2)} daily records")


def run():
    print("=" * 60)
    print("Derived Metrics — Tier 2 Intelligence")
    print("=" * 60)

    compute_session_fingerprints()
    compute_cache_efficiency()
    compute_prompt_analysis()
    compute_conversation_flows()
    compute_session_quality()
    compute_model_routing()

    print("=" * 60)
    print("Derived metrics complete!")
    print("=" * 60)


def _write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _median(lst):
    if not lst:
        return 0
    s = sorted(lst)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return round((s[n // 2 - 1] + s[n // 2]) / 2)


if __name__ == "__main__":
    run()
