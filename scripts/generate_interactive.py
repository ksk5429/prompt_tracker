"""
Generate interactive Plotly HTML dashboard for Claude Code usage analytics.
Run: python scripts/generate_interactive.py
Outputs: dashboard/interactive/index.html
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "dashboard" / "interactive"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "bg": "#0d1117", "card": "#161b22", "border": "#30363d",
    "text": "#c9d1d9", "dim": "#8b949e",
    "accent": "#58a6ff", "green": "#3fb950", "red": "#f85149",
    "orange": "#d29922", "purple": "#bc8cff", "pink": "#f778ba", "cyan": "#39d2c0",
}
SEQ = [COLORS["accent"], COLORS["green"], COLORS["red"],
       COLORS["orange"], COLORS["purple"], COLORS["pink"], COLORS["cyan"]]

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
    font=dict(color=COLORS["text"], family="Inter, -apple-system, sans-serif", size=12),
    hovermode="x unified",
    xaxis=dict(gridcolor=COLORS["border"], gridwidth=0.5),
    yaxis=dict(gridcolor=COLORS["border"], gridwidth=0.5),
    margin=dict(l=50, r=30, t=50, b=40),
)


def fig_to_div(fig, div_id):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=div_id,
                       config={"displayModeBar": True, "responsive": True,
                               "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                               "displaylogo": False})


# ── Load data ──────────────────────────────────────────────
def load_data():
    data = {}

    # merged daily
    p = PROCESSED / "merged_daily.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"])
        data["daily"] = df
    else:
        data["daily"] = pd.DataFrame()

    # sessions
    p = PROCESSED / "sessions.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"])
        data["sessions"] = df
    else:
        data["sessions"] = pd.DataFrame()

    # tool usage
    p = PROCESSED / "tool_usage.csv"
    if p.exists():
        data["tools"] = pd.read_csv(p)
    else:
        data["tools"] = pd.DataFrame()

    # model tokens
    p = PROCESSED / "model_tokens.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"])
        data["tokens"] = df
    else:
        data["tokens"] = pd.DataFrame()

    # stats cache hours
    p = PROCESSED / "stats_cache_hours.csv"
    if p.exists():
        data["hours"] = pd.read_csv(p)
    else:
        data["hours"] = pd.DataFrame()

    # stats cache models
    p = PROCESSED / "stats_cache_models.csv"
    if p.exists():
        data["models"] = pd.read_csv(p)
    else:
        data["models"] = pd.DataFrame()

    # session_tmp_daily (for project-level detail)
    p = PROCESSED / "session_tmp_daily.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"])
        data["session_detail"] = df
    else:
        data["session_detail"] = pd.DataFrame()

    return data


# ── KPI computation ────────────────────────────────────────
def compute_kpis(data):
    daily = data["daily"]
    sessions = data["sessions"]
    models = data.get("models", pd.DataFrame())

    total_sessions = int(sessions["message_count"].count()) if len(sessions) else 0
    total_msgs = int(daily["message_count"].sum()) if len(daily) else 0

    total_tokens = 0
    if len(models):
        total_tokens = int(models["input_tokens"].sum() + models["output_tokens"].sum())
    # also add from model_tokens
    tokens_df = data.get("tokens", pd.DataFrame())
    if len(tokens_df):
        total_tokens = max(total_tokens, int(tokens_df["tokens"].sum()))

    active_projects = 0
    if len(sessions):
        active_projects = sessions["project"].nunique()

    avg_duration = 0
    if len(sessions) and "duration_min" in sessions.columns:
        valid = sessions[sessions["duration_min"] > 0]
        if len(valid):
            avg_duration = round(valid["duration_min"].mean(), 0)

    # longest streak
    streak = 0
    if len(daily):
        dates = sorted(daily["date"].dropna().dt.date.unique())
        if dates:
            current_streak = 1
            max_streak = 1
            for i in range(1, len(dates)):
                if (dates[i] - dates[i-1]).days == 1:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 1
            streak = max_streak

    # most active project
    top_project = "N/A"
    if len(sessions):
        top = sessions.groupby("project")["message_count"].sum().idxmax()
        top_project = top

    return [
        ("Total Sessions", f"{total_sessions:,}"),
        ("Total Messages", f"{total_msgs:,}"),
        ("Total Tokens", f"{total_tokens:,.0f}"),
        ("Active Projects", f"{active_projects}"),
        ("Avg Duration", f"{avg_duration:.0f} min"),
        ("Longest Streak", f"{streak} days"),
        ("Top Project", top_project),
    ]


# ── Tab 1: Timeline ───────────────────────────────────────
def build_timeline_fig(data):
    df = data["daily"]
    if df.empty:
        return go.Figure().update_layout(title="No daily data", **LAYOUT_DEFAULTS)

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("Daily Messages", "Sessions per Day", "Duration (minutes)"),
                        vertical_spacing=0.08, shared_xaxes=True)

    # messages bar + 7d rolling
    fig.add_trace(go.Bar(x=df["date"], y=df["message_count"], name="Messages",
                         marker_color=SEQ[0], opacity=0.4), row=1, col=1)
    roll = df["message_count"].rolling(7, min_periods=3).mean()
    fig.add_trace(go.Scatter(x=df["date"], y=roll, name="7d avg",
                             line=dict(color=SEQ[0], width=3)), row=1, col=1)

    # sessions bar
    fig.add_trace(go.Bar(x=df["date"], y=df["session_count"], name="Sessions",
                         marker_color=SEQ[1], opacity=0.6), row=2, col=1)

    # duration
    if "duration_min" in df.columns:
        fig.add_trace(go.Bar(x=df["date"], y=df["duration_min"], name="Duration",
                             marker_color=SEQ[3], opacity=0.5), row=3, col=1)

    # data source boundary annotation
    sc = df[df["source"] == "stats_cache"]
    st = df[df["source"] == "session_tmp"]
    if len(sc) and len(st):
        boundary = sc["date"].max()
        boundary_str = boundary.strftime("%Y-%m-%d")
        for i in range(1, 4):
            fig.add_shape(type="line", x0=boundary_str, x1=boundary_str,
                          y0=0, y1=1, yref=f"y{'' if i==1 else i} domain",
                          xref=f"x{'' if i==1 else i}",
                          line=dict(dash="dash", color=COLORS["dim"], width=1),
                          row=i, col=1)
        fig.add_annotation(x=boundary_str, y=1, yref="y domain",
                           text="data source boundary", showarrow=False,
                           font=dict(color=COLORS["dim"], size=10))

    fig.update_layout(height=800, showlegend=True,
                      legend=dict(orientation="h", y=1.02), **LAYOUT_DEFAULTS)
    return fig


# ── Tab 2: Sessions ───────────────────────────────────────
def build_sessions_fig(data):
    df = data["sessions"]
    if df.empty:
        return go.Figure().update_layout(title="No session data", **LAYOUT_DEFAULTS)

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Session Duration Distribution",
                                        "Messages per Session Distribution",
                                        "Session Duration Over Time",
                                        "Messages per Session Over Time"),
                        vertical_spacing=0.12, horizontal_spacing=0.08)

    valid_dur = df[df["duration_min"] > 0]["duration_min"]
    valid_msg = df[df["message_count"] > 0]["message_count"]

    # histogram: duration
    fig.add_trace(go.Histogram(x=valid_dur, nbinsx=30, name="Duration",
                               marker_color=SEQ[0], opacity=0.7), row=1, col=1)

    # histogram: messages
    fig.add_trace(go.Histogram(x=valid_msg, nbinsx=30, name="Messages",
                               marker_color=SEQ[1], opacity=0.7), row=1, col=2)

    # scatter: duration over time
    valid = df[df["duration_min"] > 0].copy()
    if len(valid):
        fig.add_trace(go.Scatter(x=valid["date"], y=valid["duration_min"],
                                 mode="markers", name="Duration",
                                 marker=dict(color=SEQ[0], size=5, opacity=0.5)),
                      row=2, col=1)
        roll = valid.set_index("date")["duration_min"].rolling("14D").mean()
        fig.add_trace(go.Scatter(x=roll.index, y=roll.values, name="14d trend",
                                 line=dict(color=SEQ[2], width=3)), row=2, col=1)

    # scatter: messages over time
    valid = df[df["message_count"] > 0].copy()
    if len(valid):
        fig.add_trace(go.Scatter(x=valid["date"], y=valid["message_count"],
                                 mode="markers", name="Msgs",
                                 marker=dict(color=SEQ[1], size=5, opacity=0.5)),
                      row=2, col=2)
        roll = valid.set_index("date")["message_count"].rolling("14D").mean()
        fig.add_trace(go.Scatter(x=roll.index, y=roll.values, name="14d trend",
                                 line=dict(color=SEQ[2], width=3)), row=2, col=2)

    fig.update_layout(height=700, showlegend=False, **LAYOUT_DEFAULTS)
    return fig


# ── Tab 3: Projects ───────────────────────────────────────
def build_projects_fig(data):
    sessions = data["sessions"]
    if sessions.empty:
        return go.Figure().update_layout(title="No session data", **LAYOUT_DEFAULTS)

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Total Messages by Project",
                                        "Project Activity Over Time (weekly)"),
                        vertical_spacing=0.12)

    # horizontal bar
    proj_msgs = sessions.groupby("project")["message_count"].sum().sort_values()
    top_n = proj_msgs.tail(15)
    colors = [SEQ[i % len(SEQ)] for i in range(len(top_n))]
    fig.add_trace(go.Bar(x=top_n.values, y=top_n.index, orientation="h",
                         name="Messages", marker_color=colors), row=1, col=1)

    # stacked area: top 6 projects weekly
    df = sessions.copy()
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    top6 = proj_msgs.tail(6).index.tolist()
    df["proj_group"] = df["project"].apply(lambda x: x if x in top6 else "Other")
    weekly = df.groupby(["week", "proj_group"])["message_count"].sum().unstack(fill_value=0)
    for i, col in enumerate(weekly.columns):
        fig.add_trace(go.Scatter(x=weekly.index, y=weekly[col], name=col,
                                 stackgroup="one", mode="lines",
                                 line=dict(color=SEQ[i % len(SEQ)], width=0.5)),
                      row=2, col=1)

    fig.update_layout(height=750, **LAYOUT_DEFAULTS)
    fig.update_yaxes(autorange="reversed", row=1, col=1)  # fix bar direction
    return fig


# ── Tab 4: Tools ──────────────────────────────────────────
def build_tools_fig(data):
    tools = data["tools"]
    if tools.empty:
        return go.Figure().update_layout(title="No tool data", **LAYOUT_DEFAULTS)

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Top 20 Tools by Usage",
                                        "Tool Category Breakdown",
                                        "Tool x Project Heatmap", ""),
                        specs=[[{}, {"type": "pie"}], [{"colspan": 2}, None]],
                        vertical_spacing=0.12, horizontal_spacing=0.08)

    # top 20 bar
    tool_counts = tools["tool_name"].value_counts().head(20).sort_values()
    colors = [SEQ[i % len(SEQ)] for i in range(len(tool_counts))]
    fig.add_trace(go.Bar(x=tool_counts.values, y=tool_counts.index, orientation="h",
                         name="Calls", marker_color=colors), row=1, col=1)

    # category breakdown
    categories = {
        "Read": "File Ops", "Write": "File Ops", "Edit": "File Ops", "Glob": "File Ops",
        "Grep": "Search", "Agent": "Orchestration", "Bash": "Shell",
        "TodoWrite": "Planning", "ToolSearch": "Planning",
        "WebFetch": "Web", "WebSearch": "Web",
    }
    tools_cat = tools.copy()
    tools_cat["category"] = tools_cat["tool_name"].map(categories).fillna("Other")
    cat_counts = tools_cat["category"].value_counts()
    fig.add_trace(go.Pie(labels=cat_counts.index, values=cat_counts.values,
                         marker=dict(colors=SEQ[:len(cat_counts)]),
                         hole=0.4, textinfo="label+percent"), row=1, col=2)

    # heatmap: tool x project
    top_tools = tools["tool_name"].value_counts().head(10).index.tolist()
    top_projs = tools["project"].value_counts().head(8).index.tolist()
    filt = tools[tools["tool_name"].isin(top_tools) & tools["project"].isin(top_projs)]
    pivot = filt.groupby(["tool_name", "project"]).size().unstack(fill_value=0)
    if not pivot.empty:
        fig.add_trace(go.Heatmap(z=pivot.values, x=pivot.columns.tolist(),
                                 y=pivot.index.tolist(),
                                 colorscale="Blues", showscale=True), row=2, col=1)

    fig.update_layout(height=850, showlegend=False, **LAYOUT_DEFAULTS)
    return fig


# ── Tab 5: Heatmaps ──────────────────────────────────────
def build_heatmap_fig(data):
    daily = data["daily"]
    hours = data.get("hours", pd.DataFrame())

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("Contribution Calendar",
                                        "Hour of Day Activity (Jan-Feb historical)",
                                        "Day of Week Activity"),
                        vertical_spacing=0.1)

    # GitHub contribution calendar
    if not daily.empty:
        df = daily.copy()
        df["dow"] = df["date"].dt.dayofweek  # 0=Mon
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
        # adjust week to be continuous
        df["week_offset"] = ((df["date"] - df["date"].min()).dt.days // 7)
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        fig.add_trace(go.Heatmap(
            x=df["week_offset"], y=df["dow"],
            z=df["message_count"],
            colorscale=[[0, "#161b22"], [0.25, "#0e4429"], [0.5, "#006d32"],
                        [0.75, "#26a641"], [1.0, "#39d353"]],
            showscale=True, colorbar=dict(title="msgs", len=0.25, y=0.88),
            hovertemplate="Week %{x}<br>%{text}<br>Messages: %{z}<extra></extra>",
            text=[d.strftime("%Y-%m-%d") for d in df["date"]],
        ), row=1, col=1)
        fig.update_yaxes(tickvals=list(range(7)), ticktext=day_names, row=1, col=1)

    # hour of day
    if not hours.empty:
        all_hours = pd.DataFrame({"hour": range(24)})
        hours_full = all_hours.merge(hours, on="hour", how="left").fillna(0)
        fig.add_trace(go.Bar(x=hours_full["hour"], y=hours_full["count"],
                             name="Sessions started",
                             marker_color=[SEQ[0] if 9 <= h <= 22 else COLORS["dim"]
                                          for h in hours_full["hour"]]),
                      row=2, col=1)
        fig.update_xaxes(tickvals=list(range(24)),
                         ticktext=[f"{h:02d}" for h in range(24)], row=2, col=1)

    # day of week
    if not daily.empty:
        df = daily.copy()
        df["dow"] = df["date"].dt.dayofweek
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_msgs = df.groupby("dow")["message_count"].sum()
        dow_full = pd.Series(0, index=range(7))
        dow_full.update(dow_msgs)
        fig.add_trace(go.Bar(x=[day_names[i] for i in range(7)],
                             y=dow_full.values,
                             name="Messages by day",
                             marker_color=[SEQ[1] if i < 5 else SEQ[2] for i in range(7)]),
                      row=3, col=1)

    fig.update_layout(height=900, showlegend=False, **LAYOUT_DEFAULTS)
    return fig


# ── Tab 6: Models ─────────────────────────────────────────
def build_models_fig(data):
    tokens = data.get("tokens", pd.DataFrame())
    models = data.get("models", pd.DataFrame())

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Tokens by Model Over Time",
                                        "Total Tokens by Model",
                                        "Cache Efficiency by Model", ""),
                        specs=[[{}, {"type": "pie"}], [{"colspan": 2}, None]],
                        vertical_spacing=0.12, horizontal_spacing=0.08)

    # tokens over time
    if not tokens.empty:
        model_names = {"claude-opus-4-5-20251101": "Opus 4.5",
                       "claude-sonnet-4-5-20250929": "Sonnet 4.5",
                       "claude-opus-4-6": "Opus 4.6"}
        for i, (model_id, label) in enumerate(model_names.items()):
            subset = tokens[tokens["model"] == model_id]
            if not subset.empty:
                fig.add_trace(go.Bar(x=subset["date"], y=subset["tokens"],
                                     name=label, marker_color=SEQ[i], opacity=0.7),
                              row=1, col=1)
        fig.update_layout(barmode="stack")

    # pie: total tokens by model
    if not models.empty:
        totals = models["input_tokens"] + models["output_tokens"]
        labels = models["model"].apply(lambda x: x.split("-")[1].title() + " " + x.split("-")[2]).tolist()
        fig.add_trace(go.Pie(labels=labels, values=totals.values,
                             marker=dict(colors=SEQ[:len(labels)]),
                             hole=0.4, textinfo="label+percent"), row=1, col=2)

    # cache efficiency
    if not models.empty:
        fig.add_trace(go.Bar(
            x=models["model"].apply(lambda x: x.split("-")[1].title()),
            y=models["cache_read_tokens"],
            name="Cache Read", marker_color=SEQ[1], opacity=0.7,
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=models["model"].apply(lambda x: x.split("-")[1].title()),
            y=models["cache_creation_tokens"],
            name="Cache Write", marker_color=SEQ[3], opacity=0.7,
        ), row=2, col=1)

    fig.update_layout(height=700, **LAYOUT_DEFAULTS)
    return fig


# ── Tab 7: Intelligence ───────────────────────────────────
def build_intelligence_fig(data):
    daily = data["daily"]
    sessions = data["sessions"]

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Cumulative Messages Over Time",
                                        "Project Diversity (14d rolling unique projects)",
                                        "Messages per Session Trend",
                                        "Session Intensity Distribution"),
                        vertical_spacing=0.12, horizontal_spacing=0.08)

    # cumulative
    if not daily.empty:
        df = daily.sort_values("date")
        cumsum = df["message_count"].cumsum()
        fig.add_trace(go.Scatter(x=df["date"], y=cumsum, name="Cumulative",
                                 fill="tozeroy", fillcolor="rgba(88,166,255,0.15)",
                                 line=dict(color=SEQ[0], width=2)), row=1, col=1)

    # project diversity
    if not sessions.empty:
        df = sessions.sort_values("date")
        diversity = []
        for d in df["date"].unique():
            window = df[(df["date"] >= d - pd.Timedelta(days=14)) & (df["date"] <= d)]
            diversity.append({"date": d, "unique_projects": window["project"].nunique()})
        if diversity:
            div_df = pd.DataFrame(diversity)
            fig.add_trace(go.Scatter(x=div_df["date"], y=div_df["unique_projects"],
                                     name="Projects (14d)",
                                     line=dict(color=SEQ[4], width=2),
                                     fill="tozeroy", fillcolor="rgba(188,140,255,0.15)"),
                          row=1, col=2)

    # messages per session trend
    if not sessions.empty:
        valid = sessions[sessions["message_count"] > 0].copy()
        if len(valid):
            fig.add_trace(go.Scatter(x=valid["date"], y=valid["message_count"],
                                     mode="markers", name="Msgs/session",
                                     marker=dict(color=SEQ[1], size=5, opacity=0.4)),
                          row=2, col=1)
            # trend line
            valid_sorted = valid.sort_values("date")
            if len(valid_sorted) >= 5:
                roll = valid_sorted.set_index("date")["message_count"].rolling("14D").mean()
                fig.add_trace(go.Scatter(x=roll.index, y=roll.values,
                                         name="14d avg",
                                         line=dict(color=SEQ[2], width=3)),
                              row=2, col=1)

    # session intensity
    if not sessions.empty:
        valid = sessions[(sessions["duration_min"] > 0) & (sessions["message_count"] > 0)].copy()
        if len(valid):
            valid["intensity"] = valid["message_count"] / valid["duration_min"]
            fig.add_trace(go.Histogram(x=valid["intensity"], nbinsx=25,
                                       name="msg/min",
                                       marker_color=SEQ[3], opacity=0.7),
                          row=2, col=2)

    fig.update_layout(height=700, showlegend=False, **LAYOUT_DEFAULTS)
    return fig


# ── Assemble page ─────────────────────────────────────────
def build_single_page():
    data = load_data()
    kpis = compute_kpis(data)

    # build all figures
    print("  Building figures...")
    divs = {
        "timeline": fig_to_div(build_timeline_fig(data), "fig-timeline"),
        "sessions": fig_to_div(build_sessions_fig(data), "fig-sessions"),
        "projects": fig_to_div(build_projects_fig(data), "fig-projects"),
        "tools": fig_to_div(build_tools_fig(data), "fig-tools"),
        "heatmaps": fig_to_div(build_heatmap_fig(data), "fig-heatmaps"),
        "models": fig_to_div(build_models_fig(data), "fig-models"),
        "intelligence": fig_to_div(build_intelligence_fig(data), "fig-intelligence"),
    }

    # KPI HTML
    kpi_html = ""
    for label, value in kpis:
        kpi_html += f'<div class="kpi"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>\n'

    daily = data["daily"]
    sessions = data["sessions"]
    date_min = daily["date"].min().strftime("%b %Y") if len(daily) else "N/A"
    date_max = daily["date"].max().strftime("%b %Y") if len(daily) else "N/A"
    n_days = int(daily["date"].nunique()) if len(daily) else 0
    n_sessions = len(sessions) if len(sessions) else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claude Code Usage Tracker</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{ --bg:{COLORS["bg"]}; --card:{COLORS["card"]}; --border:{COLORS["border"]};
         --text:{COLORS["text"]}; --dim:{COLORS["dim"]}; --accent:{COLORS["accent"]}; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:Inter,-apple-system,sans-serif; }}
.container {{ max-width:1400px; margin:0 auto; padding:1rem; }}
header {{ display:flex; align-items:baseline; gap:1rem; margin-bottom:1rem; flex-wrap:wrap; }}
header h1 {{ font-size:1.5rem; font-weight:700; }}
.date-range {{ color:var(--dim); font-size:0.85rem; }}

/* KPI grid */
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); gap:0.75rem; margin-bottom:1.25rem; }}
.kpi {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:0.75rem; text-align:center; }}
.kpi-value {{ font-size:1.3rem; font-weight:700; color:var(--accent); }}
.kpi-label {{ font-size:0.7rem; color:var(--dim); margin-top:0.2rem; text-transform:uppercase; letter-spacing:0.05em; }}

/* Tabs */
.tabs {{ display:flex; gap:0.5rem; margin-bottom:1rem; overflow-x:auto; padding-bottom:0.25rem; }}
.tab {{ padding:0.4rem 1rem; border-radius:8px; cursor:pointer; font-size:0.85rem; color:var(--dim);
        border:1px solid var(--border); background:transparent; white-space:nowrap; transition:all 0.15s; }}
.tab:hover {{ border-color:var(--accent); color:var(--text); }}
.tab.active {{ background:var(--accent); color:var(--bg); border-color:var(--accent); font-weight:600; }}

/* Panels */
.panel {{ display:none; }}
.panel.active {{ display:block; }}
.chart-card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:0.5rem; margin-bottom:1rem; }}

footer {{ text-align:center; color:var(--dim); font-size:0.75rem; margin-top:2rem; padding:1rem 0; border-top:1px solid var(--border); }}
footer a {{ color:var(--accent); }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Claude Code Usage Tracker</h1>
        <span class="date-range">{date_min} &ndash; {date_max} | Updated {pd.Timestamp.now().strftime('%Y-%m-%d')}</span>
    </header>

    <div class="kpi-grid">{kpi_html}</div>

    <div class="tabs">
        <div class="tab active" data-tab="timeline">Timeline</div>
        <div class="tab" data-tab="sessions">Sessions</div>
        <div class="tab" data-tab="projects">Projects</div>
        <div class="tab" data-tab="tools">Tools</div>
        <div class="tab" data-tab="heatmaps">Heatmaps</div>
        <div class="tab" data-tab="models">Models</div>
        <div class="tab" data-tab="intelligence">Intelligence</div>
    </div>

    <div class="panel active" id="panel-timeline">
        <div class="chart-card">{divs["timeline"]}</div>
    </div>
    <div class="panel" id="panel-sessions">
        <div class="chart-card">{divs["sessions"]}</div>
    </div>
    <div class="panel" id="panel-projects">
        <div class="chart-card">{divs["projects"]}</div>
    </div>
    <div class="panel" id="panel-tools">
        <div class="chart-card">{divs["tools"]}</div>
    </div>
    <div class="panel" id="panel-heatmaps">
        <div class="chart-card">{divs["heatmaps"]}</div>
    </div>
    <div class="panel" id="panel-models">
        <div class="chart-card">{divs["models"]}</div>
    </div>
    <div class="panel" id="panel-intelligence">
        <div class="chart-card">{divs["intelligence"]}</div>
    </div>

    <footer>
        {n_days} active days &middot; {n_sessions} sessions &middot;
        Auto-generated by <a href="https://github.com/ksk5429/prompt_tracker">prompt_tracker</a>
    </footer>
</div>

<script>
document.querySelectorAll('.tab').forEach(tab => {{
    tab.addEventListener('click', () => {{
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
        window.dispatchEvent(new Event('resize'));
    }});
}});
</script>
</body>
</html>"""

    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"  Written: {OUT / 'index.html'}")


def main():
    print("=== Generating Interactive Dashboard ===")
    build_single_page()
    print("=== Done ===")


if __name__ == "__main__":
    main()
