"""
Streaming JSONL parser with content-hash caching.
Extracts per-message data from ~/.claude/projects/*/*.jsonl files.
Only re-parses files whose content hash has changed.
"""
import json
import csv
import hashlib
import os
from pathlib import Path
from collections import defaultdict

CLAUDE_DIR = Path.home() / ".claude" / "projects"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"
CACHE_DIR = Path(__file__).parent.parent / "data" / "raw" / "jsonl_cache"
HASH_FILE = CACHE_DIR / "file_hashes.json"


def file_hash(path):
    """SHA-256 of first 64KB + file size (fast proxy for full hash)."""
    h = hashlib.sha256()
    size = os.path.getsize(path)
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


def load_hash_cache():
    if HASH_FILE.exists():
        with open(HASH_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_hash_cache(cache):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def normalize_project(dirname):
    """Extract top-level project name from directory name."""
    import re
    name = dirname
    name = re.sub(r"^[a-zA-Z]--", "", name)
    name = re.sub(r"^C--Users-\w+", "HOME", name)
    return name.split("-")[0] if "-" in name else name


def parse_jsonl(filepath, project):
    """Stream-parse a single JSONL file, extract structured data."""
    messages = []       # per-user-message records
    tool_calls = []     # per-tool-call records
    token_records = []  # per-final-assistant token usage
    session_meta = {}   # session-level metadata

    session_id = filepath.stem
    current_model = "unknown"

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = rec.get("type")

            if rtype == "user":
                ts = rec.get("timestamp", "")
                content = rec.get("message", {}).get("content", [])
                text = ""
                has_image = False
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text += block.get("text", "") + " "
                            elif block.get("type") in ("image", "image_url"):
                                has_image = True
                        elif isinstance(block, str):
                            text += block + " "
                text = text.strip()

                # detect if this is a correction (follows a tool failure or short re-prompt)
                messages.append({
                    "session_id": session_id,
                    "project": project,
                    "timestamp": ts,
                    "prompt_length": len(text),
                    "prompt_words": len(text.split()) if text else 0,
                    "has_image": has_image,
                })

                # session metadata from first user record
                if not session_meta:
                    session_meta = {
                        "session_id": session_id,
                        "project": project,
                        "entrypoint": rec.get("entrypoint", "unknown"),
                        "git_branch": rec.get("gitBranch", ""),
                        "version": rec.get("version", ""),
                        "cwd": rec.get("cwd", ""),
                        "first_timestamp": ts,
                    }

            elif rtype == "assistant":
                msg = rec.get("message", {})
                model = msg.get("model", "")
                if model and model != "<synthetic>":
                    current_model = model

                usage = msg.get("usage", {})
                stop_reason = msg.get("stop_reason")

                # extract tool calls from content
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_name = block.get("name", "unknown")
                        tool_input = block.get("input", {})
                        # extract key info from tool input
                        tool_detail = ""
                        if tool_name in ("Read", "Write", "Edit"):
                            tool_detail = tool_input.get("file_path", "")
                        elif tool_name == "Bash":
                            cmd = tool_input.get("command", "")
                            tool_detail = cmd[:150]
                        elif tool_name == "Grep":
                            tool_detail = tool_input.get("pattern", "")
                        elif tool_name == "Agent":
                            tool_detail = tool_input.get("description", "")

                        tool_calls.append({
                            "session_id": session_id,
                            "project": project,
                            "timestamp": rec.get("timestamp", ""),
                            "tool_name": tool_name,
                            "model": current_model,
                        })

                # only record tokens from final responses (stop_reason set)
                if stop_reason:
                    ts = rec.get("timestamp", "")
                    if session_meta:
                        session_meta["last_timestamp"] = ts

                    token_records.append({
                        "session_id": session_id,
                        "project": project,
                        "timestamp": ts,
                        "model": current_model,
                        "stop_reason": stop_reason,
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                        "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
                    })

    return messages, tool_calls, token_records, session_meta


def run():
    if not CLAUDE_DIR.exists():
        print("[jsonl_deep] ~/.claude/projects not found")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    hash_cache = load_hash_cache()
    all_messages = []
    all_tool_calls = []
    all_token_records = []
    all_session_metas = []
    files_parsed = 0
    files_skipped = 0

    # find all jsonl files
    jsonl_files = []
    for pdir in CLAUDE_DIR.iterdir():
        if pdir.is_dir():
            project = normalize_project(pdir.name)
            for jf in pdir.glob("*.jsonl"):
                jsonl_files.append((jf, project))

    print(f"[jsonl_deep] found {len(jsonl_files)} JSONL files ({sum(os.path.getsize(f) for f, _ in jsonl_files) / 1024 / 1024:.0f} MB)")

    for jf, project in jsonl_files:
        fpath = str(jf)
        fhash = file_hash(jf)

        if hash_cache.get(fpath) == fhash:
            files_skipped += 1
            continue

        try:
            msgs, tools, tokens, meta = parse_jsonl(jf, project)
            all_messages.extend(msgs)
            all_tool_calls.extend(tools)
            all_token_records.extend(tokens)
            if meta:
                all_session_metas.append(meta)
            hash_cache[fpath] = fhash
            files_parsed += 1
        except Exception as e:
            print(f"[jsonl_deep] ERROR parsing {jf.name}: {e}")

    print(f"[jsonl_deep] parsed {files_parsed} files, skipped {files_skipped} (cached)")

    # load existing data if incremental
    existing_msgs = _load_existing(OUT_DIR / "jsonl_messages.csv",
                                   ["session_id", "project", "timestamp", "prompt_length",
                                    "prompt_words", "has_image"])
    existing_tools = _load_existing(OUT_DIR / "jsonl_tool_calls.csv",
                                    ["session_id", "project", "timestamp", "tool_name",
                                     "model"])
    existing_tokens = _load_existing(OUT_DIR / "jsonl_tokens.csv",
                                     ["session_id", "project", "timestamp", "model",
                                      "stop_reason", "input_tokens", "output_tokens",
                                      "cache_read_tokens", "cache_creation_tokens"])
    existing_metas = _load_existing(OUT_DIR / "jsonl_session_meta.csv",
                                    ["session_id", "project", "entrypoint", "git_branch",
                                     "version", "cwd", "first_timestamp", "last_timestamp"])

    # merge: remove old entries for re-parsed sessions, add new
    reparsed_ids = {m["session_id"] for m in all_session_metas}
    existing_msgs = [r for r in existing_msgs if r["session_id"] not in reparsed_ids]
    existing_tools = [r for r in existing_tools if r["session_id"] not in reparsed_ids]
    existing_tokens = [r for r in existing_tokens if r["session_id"] not in reparsed_ids]
    existing_metas = [r for r in existing_metas if r["session_id"] not in reparsed_ids]

    all_messages = existing_msgs + all_messages
    all_tool_calls = existing_tools + all_tool_calls
    all_token_records = existing_tokens + all_token_records
    all_session_metas = existing_metas + all_session_metas

    # write CSVs
    _write_csv(OUT_DIR / "jsonl_messages.csv", all_messages,
               ["session_id", "project", "timestamp", "prompt_length",
                "prompt_words", "has_image"])
    _write_csv(OUT_DIR / "jsonl_tool_calls.csv", all_tool_calls,
               ["session_id", "project", "timestamp", "tool_name",
                "model"])
    _write_csv(OUT_DIR / "jsonl_tokens.csv", all_token_records,
               ["session_id", "project", "timestamp", "model",
                "stop_reason", "input_tokens", "output_tokens",
                "cache_read_tokens", "cache_creation_tokens"])
    _write_csv(OUT_DIR / "jsonl_session_meta.csv", all_session_metas,
               ["session_id", "project", "entrypoint", "git_branch",
                "version", "cwd", "first_timestamp", "last_timestamp"])

    save_hash_cache(hash_cache)

    print(f"[jsonl_deep] total: {len(all_messages)} messages, {len(all_tool_calls)} tool calls, "
          f"{len(all_token_records)} token records, {len(all_session_metas)} sessions")


def _load_existing(path, fields):
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[jsonl_deep] wrote {len(rows)} rows to {path.name}")


if __name__ == "__main__":
    run()
