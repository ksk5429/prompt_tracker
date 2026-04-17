"""Parse ~/.claude/metrics/costs.jsonl into processed CSV."""
import json
import csv
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"


def run(raw_dir=None):
    src = (raw_dir or RAW_DIR) / "costs.jsonl"
    if not src.exists():
        print(f"[costs] {src} not found, skipping")
        return

    daily = defaultdict(lambda: {"request_count": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})

    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            model = rec.get("model", "unknown")
            if model == "unknown":
                continue
            date = rec.get("timestamp", "")[:10]
            if not date:
                continue
            key = (date, model)
            daily[key]["request_count"] += 1
            daily[key]["input_tokens"] += rec.get("input_tokens", 0)
            daily[key]["output_tokens"] += rec.get("output_tokens", 0)
            daily[key]["cost_usd"] += rec.get("estimated_cost_usd", 0.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for (date, model), vals in sorted(daily.items()):
        rows.append({
            "date": date,
            "model": model,
            "request_count": vals["request_count"],
            "input_tokens": vals["input_tokens"],
            "output_tokens": vals["output_tokens"],
            "cost_usd": round(vals["cost_usd"], 6),
        })

    with open(OUT_DIR / "costs_clean.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "model", "request_count", "input_tokens", "output_tokens", "cost_usd"])
        w.writeheader()
        w.writerows(rows)
    print(f"[costs] wrote {len(rows)} rows to costs_clean.csv")


if __name__ == "__main__":
    run()
