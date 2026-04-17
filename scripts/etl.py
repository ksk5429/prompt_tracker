"""ETL orchestrator: sync raw data from ~/.claude, run all parsers, merge into unified CSVs."""
import shutil
import csv
from pathlib import Path
from collections import defaultdict

CLAUDE_DIR = Path.home() / ".claude"
PROJECT_DIR = Path(__file__).parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUT_DIR = PROJECT_DIR / "data" / "processed"


def sync_raw_data():
    """Copy data from ~/.claude to data/raw/ for local processing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # stats-cache.json
    src = CLAUDE_DIR / "stats-cache.json"
    if src.exists():
        shutil.copy2(src, RAW_DIR / "stats_cache.json")
        print(f"[sync] copied stats-cache.json")

    # costs.jsonl
    src = CLAUDE_DIR / "metrics" / "costs.jsonl"
    if src.exists():
        shutil.copy2(src, RAW_DIR / "costs.jsonl")
        print(f"[sync] copied costs.jsonl")

    # session-data/*.tmp
    session_dir = RAW_DIR / "session_tmp"
    session_dir.mkdir(exist_ok=True)
    src_dir = CLAUDE_DIR / "session-data"
    if src_dir.exists():
        count = 0
        for f in src_dir.glob("*.tmp"):
            shutil.copy2(f, session_dir / f.name)
            count += 1
        print(f"[sync] copied {count} .tmp session files")

    # sessions-index.json from each project
    idx_dir = RAW_DIR / "sessions_index"
    idx_dir.mkdir(exist_ok=True)
    projects_dir = CLAUDE_DIR / "projects"
    if projects_dir.exists():
        count = 0
        for pdir in projects_dir.iterdir():
            if pdir.is_dir():
                idx = pdir / "sessions-index.json"
                if idx.exists():
                    shutil.copy2(idx, idx_dir / f"{pdir.name}.json")
                    count += 1
        print(f"[sync] copied {count} sessions-index.json files")


def merge_daily():
    """Merge stats_cache_daily + session_tmp_daily into merged_daily.csv."""
    daily = {}

    # stats_cache_daily (Jan-Feb, has message_count, session_count, tool_call_count)
    sc_path = OUT_DIR / "stats_cache_daily.csv"
    if sc_path.exists():
        with open(sc_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                daily[row["date"]] = {
                    "date": row["date"],
                    "message_count": int(row["message_count"]),
                    "session_count": int(row["session_count"]),
                    "tool_call_count": int(row["tool_call_count"]),
                    "user_msg_count": 0,
                    "duration_min": 0,
                    "files_modified_count": 0,
                    "source": "stats_cache",
                }

    # session_tmp_daily (Mar-Apr, has user_msg_count, duration, files, tools)
    st_path = OUT_DIR / "session_tmp_daily.csv"
    if st_path.exists():
        # aggregate by date (multiple sessions per day)
        day_agg = defaultdict(lambda: {
            "user_msg_count": 0, "session_count": 0,
            "duration_min": 0, "files_modified_count": 0,
            "tools": set(),
        })
        with open(st_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d = row["date"]
                agg = day_agg[d]
                agg["session_count"] += 1
                agg["user_msg_count"] += int(row["user_msg_count"] or 0)
                agg["duration_min"] += int(row["duration_min"] or 0)
                agg["files_modified_count"] += int(row["files_modified_count"] or 0)
                if row["tools_used"]:
                    for t in row["tools_used"].split(","):
                        agg["tools"].add(t.strip())

        for d, agg in day_agg.items():
            daily[d] = {
                "date": d,
                "message_count": agg["user_msg_count"],  # best proxy
                "session_count": agg["session_count"],
                "tool_call_count": len(agg["tools"]),  # unique tools, not calls
                "user_msg_count": agg["user_msg_count"],
                "duration_min": agg["duration_min"],
                "files_modified_count": agg["files_modified_count"],
                "source": "session_tmp",
            }

    rows = [daily[d] for d in sorted(daily)]
    fields = ["date", "message_count", "session_count", "tool_call_count",
              "user_msg_count", "duration_min", "files_modified_count", "source"]
    with open(OUT_DIR / "merged_daily.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[merge] wrote {len(rows)} rows to merged_daily.csv")


def merge_sessions():
    """Merge sessions_index + session_tmp into unified sessions.csv."""
    rows = []

    # from sessions_index (Jan-Feb)
    si_path = OUT_DIR / "sessions_index.csv"
    if si_path.exists():
        with open(si_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "session_id": row["session_id"],
                    "project": row["project"],
                    "date": row["date"],
                    "duration_min": float(row["duration_min"] or 0),
                    "message_count": int(row["message_count"] or 0),
                    "source": "sessions_index",
                })

    # from session_tmp (Mar-Apr) — each file is one session
    st_path = OUT_DIR / "session_tmp_daily.csv"
    if st_path.exists():
        with open(st_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "session_id": f"{row['date']}_{row['project']}",
                    "project": row["project"],
                    "date": row["date"],
                    "duration_min": float(row["duration_min"] or 0),
                    "message_count": int(row["user_msg_count"] or 0),
                    "source": "session_tmp",
                })

    rows.sort(key=lambda x: x["date"])
    fields = ["session_id", "project", "date", "duration_min", "message_count", "source"]
    with open(OUT_DIR / "sessions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[merge] wrote {len(rows)} rows to sessions.csv")


def merge_tokens():
    """Merge stats_cache_tokens + costs_clean into model_tokens.csv."""
    rows = []

    sc_path = OUT_DIR / "stats_cache_tokens.csv"
    if sc_path.exists():
        with open(sc_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "date": row["date"],
                    "model": row["model"],
                    "tokens": int(row["tokens"]),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "source": "stats_cache",
                })

    cc_path = OUT_DIR / "costs_clean.csv"
    if cc_path.exists():
        with open(cc_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "date": row["date"],
                    "model": row["model"],
                    "tokens": int(row["input_tokens"]) + int(row["output_tokens"]),
                    "input_tokens": int(row["input_tokens"]),
                    "output_tokens": int(row["output_tokens"]),
                    "source": "costs",
                })

    rows.sort(key=lambda x: x["date"])
    fields = ["date", "model", "tokens", "input_tokens", "output_tokens", "source"]
    with open(OUT_DIR / "model_tokens.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[merge] wrote {len(rows)} rows to model_tokens.csv")


def main():
    print("=" * 60)
    print("Prompt Tracker ETL Pipeline")
    print("=" * 60)

    # Step 1: sync raw data
    print("\n--- Syncing raw data from ~/.claude ---")
    sync_raw_data()

    # Step 2: run individual ETL scripts
    print("\n--- Running ETL: stats_cache ---")
    from etl_stats_cache import run as run_stats
    run_stats(RAW_DIR)

    print("\n--- Running ETL: costs ---")
    from etl_costs import run as run_costs
    run_costs(RAW_DIR)

    print("\n--- Running ETL: session_tmp ---")
    from etl_session_tmp import run as run_tmp
    run_tmp(RAW_DIR / "session_tmp")

    print("\n--- Running ETL: sessions_index ---")
    from etl_sessions_index import run as run_idx
    run_idx(RAW_DIR / "sessions_index")

    # Step 3: merge
    print("\n--- Merging datasets ---")
    merge_daily()
    merge_sessions()
    merge_tokens()

    print("\n" + "=" * 60)
    print("ETL complete! Processed CSVs in data/processed/")
    print("=" * 60)


if __name__ == "__main__":
    main()
