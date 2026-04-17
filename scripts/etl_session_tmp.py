"""Parse ~/.claude/session-data/*.tmp markdown journals into processed CSVs."""
import re
import csv
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "session_tmp"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"


def parse_tmp(filepath):
    """Parse a single .tmp session journal file."""
    text = filepath.read_text(encoding="utf-8", errors="replace")

    def extract(pattern, default=""):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else default

    date = extract(r"\*\*Date:\*\*\s*(\S+)")
    start = extract(r"\*\*Started:\*\*\s*(\d{1,2}:\d{2})")
    end = extract(r"\*\*Last Updated:\*\*\s*(\d{1,2}:\d{2})")
    project = extract(r"\*\*Project:\*\*\s*(.+)")
    branch = extract(r"\*\*Branch:\*\*\s*(.+)")

    # duration
    duration_min = 0
    if start and end:
        try:
            sh, sm = map(int, start.split(":"))
            eh, em = map(int, end.split(":"))
            dur = (eh * 60 + em) - (sh * 60 + sm)
            if dur < 0:
                dur += 24 * 60
            duration_min = dur
        except ValueError:
            pass

    # user messages
    user_msg_count = 0
    m = re.search(r"Total user messages:\s*(\d+)", text)
    if m:
        user_msg_count = int(m.group(1))

    # tools used
    tools_used = ""
    m = re.search(r"### Tools Used\s*\n(.+)", text)
    if m:
        tools_used = m.group(1).strip()

    # files modified (count lines starting with -)
    files_modified = []
    m = re.search(r"### Files Modified\s*\n(.*?)(?=\n###|\n<!-- )", text, re.DOTALL)
    if m:
        for line in m.group(1).strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                files_modified.append(line[2:].strip())

    # tasks (count)
    task_count = 0
    m = re.search(r"### Tasks\s*\n(.*?)(?=\n###)", text, re.DOTALL)
    if m:
        for line in m.group(1).strip().split("\n"):
            if line.strip().startswith("- "):
                task_count += 1

    return {
        "date": date,
        "project": project,
        "branch": branch,
        "start_time": start,
        "end_time": end,
        "duration_min": duration_min,
        "user_msg_count": user_msg_count,
        "task_count": task_count,
        "files_modified_count": len(files_modified),
        "tools_used": tools_used,
    }


def run(raw_dir=None):
    src_dir = raw_dir or RAW_DIR
    if not src_dir.exists():
        print(f"[session_tmp] {src_dir} not found, skipping")
        return

    files = sorted(src_dir.glob("*.tmp"))
    if not files:
        print("[session_tmp] no .tmp files found, skipping")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sessions = []
    tool_rows = []
    for fp in files:
        rec = parse_tmp(fp)
        sessions.append(rec)
        # explode tools
        if rec["tools_used"]:
            for tool in rec["tools_used"].split(","):
                tool = tool.strip()
                if tool:
                    tool_rows.append({
                        "date": rec["date"],
                        "project": rec["project"],
                        "tool_name": tool,
                    })

    # sessions
    fields = ["date", "project", "branch", "start_time", "end_time",
              "duration_min", "user_msg_count", "task_count",
              "files_modified_count", "tools_used"]
    with open(OUT_DIR / "session_tmp_daily.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sessions)
    print(f"[session_tmp] wrote {len(sessions)} rows to session_tmp_daily.csv")

    # tool usage
    with open(OUT_DIR / "tool_usage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "project", "tool_name"])
        w.writeheader()
        w.writerows(tool_rows)
    print(f"[session_tmp] wrote {len(tool_rows)} rows to tool_usage.csv")


if __name__ == "__main__":
    run()
