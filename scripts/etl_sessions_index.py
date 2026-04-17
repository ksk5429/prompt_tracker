"""Parse ~/.claude/projects/*/sessions-index.json into processed CSV."""
import json
import csv
import re
from pathlib import Path
from datetime import datetime

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "sessions_index"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"


def normalize_project(dirname):
    """f--GITHUB3-docs -> GITHUB3/docs, strip drive prefix."""
    name = dirname
    name = re.sub(r"^[a-zA-Z]--", "", name)
    name = re.sub(r"^C--Users-\w+", "HOME", name)
    parts = name.split("-")
    # top-level is first part
    return parts[0] if parts else name


def run(raw_dir=None):
    src_dir = raw_dir or RAW_DIR
    if not src_dir.exists():
        print(f"[sessions_index] {src_dir} not found, skipping")
        return

    files = sorted(src_dir.glob("*.json"))
    if not files:
        print("[sessions_index] no JSON files found, skipping")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[sessions_index] skipping {fp.name}: {e}")
            continue

        project_dir = fp.stem  # filename without .json
        project = normalize_project(project_dir)

        for entry in data.get("entries", []):
            created = entry.get("created", "")
            modified = entry.get("modified", "")
            duration_min = 0
            if created and modified:
                try:
                    t0 = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(modified.replace("Z", "+00:00"))
                    duration_min = round((t1 - t0).total_seconds() / 60, 1)
                except ValueError:
                    pass

            rows.append({
                "session_id": entry.get("sessionId", ""),
                "project": project,
                "date": created[:10] if created else "",
                "created": created,
                "modified": modified,
                "duration_min": duration_min,
                "message_count": entry.get("messageCount", 0),
                "is_sidechain": entry.get("isSidechain", False),
            })

    fields = ["session_id", "project", "date", "created", "modified",
              "duration_min", "message_count", "is_sidechain"]
    with open(OUT_DIR / "sessions_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[sessions_index] wrote {len(rows)} rows to sessions_index.csv")


if __name__ == "__main__":
    run()
