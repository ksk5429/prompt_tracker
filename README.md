# Prompt Tracker

Interactive dashboard tracking Claude Code usage analytics.

**Live:** [ksk5429.github.io/prompt_tracker](https://ksk5429.github.io/prompt_tracker/)

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| Timeline | Daily message volume, session count, duration |
| Sessions | Duration/message histograms, trends over time |
| Projects | Messages by project, stacked area, weekly breakdown |
| Tools | Top tools, category breakdown, tool x project heatmap |
| Heatmaps | Contribution calendar, hour-of-day, day-of-week |
| Models | Token usage by model, cache efficiency |
| Intelligence | Cumulative growth, project diversity, intensity |

## Data Sources

- `~/.claude/stats-cache.json` (Jan-Feb 2026)
- `~/.claude/session-data/*.tmp` (Mar-Apr 2026)
- `~/.claude/projects/*/sessions-index.json`
- `~/.claude/metrics/costs.jsonl`

## Local Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run ETL (syncs from ~/.claude, generates CSVs)
cd scripts && python etl.py

# Generate dashboard
python generate_interactive.py
```

## Architecture

```
scripts/etl.py          → data/processed/*.csv
scripts/generate_interactive.py → dashboard/interactive/index.html
```

GitHub Actions auto-regenerates the dashboard and deploys to Pages on push.
