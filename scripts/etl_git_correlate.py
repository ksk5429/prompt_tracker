"""
Correlate git commits with Claude Code sessions.
Matches commit timestamps to session time windows.
"""
import csv
import subprocess
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

OUT_DIR = Path(__file__).parent.parent / "data" / "processed"

# Known git repos the user works in
GIT_REPOS = [
    Path(r"F:\TREE_OF_THOUGHT"),
    Path(r"F:\GITHUB3"),
    Path(r"F:\GITHUB3\numerical_model_fresh"),
    Path(r"F:\health_tracker"),
]


def get_commits(repo_path, since="2026-01-01"):
    """Extract git log from a repo."""
    if not (repo_path / ".git").exists():
        return []
    try:
        result = subprocess.run(
            ["git", "log", f"--since={since}", "--format=%H|%aI|%s|%an", "--all"],
            cwd=str(repo_path), capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line or "|" not in line:
                continue
            parts = line.split("|", 3)
            if len(parts) >= 4:
                commits.append({
                    "hash": parts[0][:12],
                    "timestamp": parts[1],
                    "subject": parts[2][:200],
                    "author": parts[3],
                    "repo": repo_path.name,
                })
        return commits
    except Exception as e:
        print(f"[git] error reading {repo_path}: {e}")
        return []


def get_commit_stats(repo_path, commit_hash):
    """Get files changed and lines for a commit."""
    try:
        result = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--numstat", "-r", commit_hash],
            cwd=str(repo_path), capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace",
        )
        additions = 0
        deletions = 0
        files_changed = 0
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    additions += int(parts[0]) if parts[0] != "-" else 0
                    deletions += int(parts[1]) if parts[1] != "-" else 0
                    files_changed += 1
                except ValueError:
                    pass
        return additions, deletions, files_changed
    except Exception:
        return 0, 0, 0


def load_sessions():
    """Load session time windows from jsonl_session_meta.csv."""
    path = OUT_DIR / "jsonl_session_meta.csv"
    if not path.exists():
        return []
    sessions = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            first = row.get("first_timestamp", "")
            last = row.get("last_timestamp", "")
            if first and last:
                try:
                    t0 = datetime.fromisoformat(first.replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(last.replace("Z", "+00:00"))
                    sessions.append({
                        "session_id": row["session_id"],
                        "project": row["project"],
                        "start": t0,
                        "end": t1,
                    })
                except ValueError:
                    pass
    return sessions


def match_commit_to_session(commit_ts_str, sessions, tolerance_min=30):
    """Find which session a commit belongs to (within tolerance)."""
    try:
        ct = datetime.fromisoformat(commit_ts_str)
        if ct.tzinfo is None:
            ct = ct.replace(tzinfo=timezone.utc)
    except ValueError:
        return None, None

    tolerance = timedelta(minutes=tolerance_min)
    best = None
    best_dist = timedelta.max
    for s in sessions:
        # commit within session window (with tolerance)
        if s["start"] - tolerance <= ct <= s["end"] + tolerance:
            dist = abs(ct - s["start"])
            if dist < best_dist:
                best = s
                best_dist = dist

    if best:
        return best["session_id"], best["project"]
    return None, None


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions()
    print(f"[git] loaded {len(sessions)} sessions with time windows")

    all_commits = []
    for repo in GIT_REPOS:
        if repo.exists():
            commits = get_commits(repo)
            print(f"[git] {repo.name}: {len(commits)} commits")
            all_commits.extend(commits)

    print(f"[git] total commits: {len(all_commits)}")

    # match and enrich
    rows = []
    matched = 0
    for c in all_commits:
        sid, proj = match_commit_to_session(c["timestamp"], sessions)
        # get stats for matched commits
        repo_path = None
        for r in GIT_REPOS:
            if r.name == c["repo"]:
                repo_path = r
                break

        additions, deletions, files_changed = 0, 0, 0
        if repo_path and sid:
            additions, deletions, files_changed = get_commit_stats(repo_path, c["hash"])

        rows.append({
            "commit_hash": c["hash"],
            "commit_timestamp": c["timestamp"],
            "commit_subject": c["subject"],
            "commit_author": c["author"],
            "repo": c["repo"],
            "session_id": sid or "",
            "session_project": proj or "",
            "additions": additions,
            "deletions": deletions,
            "files_changed": files_changed,
        })
        if sid:
            matched += 1

    fields = ["commit_hash", "commit_timestamp", "commit_subject", "commit_author",
              "repo", "session_id", "session_project", "additions", "deletions", "files_changed"]
    with open(OUT_DIR / "git_session_join.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"[git] wrote {len(rows)} commits, {matched} matched to sessions ({matched*100//max(len(rows),1)}%)")


if __name__ == "__main__":
    run()
