"""Parse ~/.claude/stats-cache.json into processed CSVs."""
import json
import csv
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"


def run(raw_dir=None):
    src = (raw_dir or RAW_DIR) / "stats_cache.json"
    if not src.exists():
        print(f"[stats_cache] {src} not found, skipping")
        return

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # daily activity
    rows = data.get("dailyActivity", [])
    with open(OUT_DIR / "stats_cache_daily.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "message_count", "session_count", "tool_call_count"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "date": r["date"],
                "message_count": r["messageCount"],
                "session_count": r["sessionCount"],
                "tool_call_count": r["toolCallCount"],
            })
    print(f"[stats_cache] wrote {len(rows)} rows to stats_cache_daily.csv")

    # daily model tokens
    token_rows = []
    for entry in data.get("dailyModelTokens", []):
        for model, tokens in entry.get("tokensByModel", {}).items():
            token_rows.append({"date": entry["date"], "model": model, "tokens": tokens})
    with open(OUT_DIR / "stats_cache_tokens.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "model", "tokens"])
        w.writeheader()
        w.writerows(token_rows)
    print(f"[stats_cache] wrote {len(token_rows)} rows to stats_cache_tokens.csv")

    # hour counts
    hour_rows = [{"hour": int(h), "count": c} for h, c in data.get("hourCounts", {}).items()]
    hour_rows.sort(key=lambda x: x["hour"])
    with open(OUT_DIR / "stats_cache_hours.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["hour", "count"])
        w.writeheader()
        w.writerows(hour_rows)

    # model usage summary
    model_rows = []
    for model, usage in data.get("modelUsage", {}).items():
        model_rows.append({
            "model": model,
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "cache_read_tokens": usage.get("cacheReadInputTokens", 0),
            "cache_creation_tokens": usage.get("cacheCreationInputTokens", 0),
        })
    with open(OUT_DIR / "stats_cache_models.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens"])
        w.writeheader()
        w.writerows(model_rows)

    # summary stats
    summary = {
        "total_sessions": data.get("totalSessions", 0),
        "total_messages": data.get("totalMessages", 0),
        "first_session": data.get("firstSessionDate", ""),
        "last_computed": data.get("lastComputedDate", ""),
    }
    longest = data.get("longestSession", {})
    if longest:
        summary["longest_session_id"] = longest.get("sessionId", "")
        summary["longest_session_duration_h"] = round(longest.get("duration", 0) / 3600000, 1)
        summary["longest_session_messages"] = longest.get("messageCount", 0)

    with open(OUT_DIR / "stats_cache_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print(f"[stats_cache] summary: {summary['total_sessions']} sessions, {summary['total_messages']} messages")


if __name__ == "__main__":
    run()
