from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .analyze import analyze_pair  # noqa: F401 (re-exported for convenience)


PLOT_COLORS = {
    "base": "#2f6fed",
    "test": "#19a974",
    "hit": "#19a974",
    "miss": "#e5484d",
}


def _pretty_model_name(model_path: str) -> str:
    parts = Path(str(model_path)).parts
    if "transformers" in parts:
        idx = parts.index("transformers")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if len(parts) >= 2 and parts[-1].isdigit():
        return parts[-2]
    return Path(str(model_path)).name or str(model_path)


def _build_markdown(
    result,
    totals: Dict[str, Any],
    question_summaries: List[Dict[str, Any]],
    max_questions: int,
    preview_chars: int,
) -> str:
    base_model = _pretty_model_name(totals["base"]["model"])
    test_model = _pretty_model_name(totals["test"]["model"])
    hw_plan = getattr(result, "hardware_plan", None) or "auto"
    n_questions = getattr(result, "n_questions", len(question_summaries))
    samples_start = getattr(result, "samples_start", 0)

    lines: List[str] = [
        f"# Gemma 4 comparison — `{result.run_name}`",
        "",
        f"Task `{result.task}` · slice `{n_questions} × {result.rounds}` · starts at `{samples_start}` · hardware `{hw_plan}`",
        "",
        "## Scores",
        "",
        "| Side | Model | Per-round | Majority | Correct |",
        "|---|---|---:|---:|---:|",
        f"| base | `{base_model}` | {totals['base']['per_round_accuracy_pct']:.2f}% | "
        f"{totals['base']['majority_accuracy_pct']:.2f}% | "
        f"{totals['base']['correct']}/{totals['base']['samples']} |",
        f"| test | `{test_model}` | {totals['test']['per_round_accuracy_pct']:.2f}% | "
        f"{totals['test']['majority_accuracy_pct']:.2f}% | "
        f"{totals['test']['correct']}/{totals['test']['samples']} |",
        "",
        f"**Per-round Δ** {totals['delta_pp']:+.2f} pp · "
        f"**Majority Δ** {totals['majority_delta_pp']:+.2f} pp",
        "",
    ]

    shown = question_summaries[:max_questions]
    if shown:
        lines.append("## Per-question breakdown")
        lines.append("")
    for q in shown:
        gold_letter = q["gold_letter"]
        gold_text = q["gold_text"] or "—"
        body = q["question"][:preview_chars].strip()
        lines.append(f"### Question {q['question_index']}")
        lines.append("")
        lines.append(f"**Gold** `{gold_letter}` — {gold_text}")
        lines.append("")
        lines.append(f"> {body}")
        lines.append("")
        lines.append("| Side | Majority | Matches gold | Hits | Answers |")
        lines.append("|---|:---:|:---:|---:|---|")
        for side in ("base", "test"):
            data = q[side]
            majority = data["majority_answer"]
            matches = (
                "yes" if majority == gold_letter and gold_letter != "?" else "no"
            )
            answers = " ".join(data["answers"]) or "—"
            lines.append(
                f"| {side} | `{majority}` | {matches} | "
                f"{sum(data['hits'])}/{data['total']} | {answers} |"
            )
        lines.append("")

    hidden = len(question_summaries) - len(shown)
    if hidden > 0:
        lines.append(f"*…{hidden} more question(s) saved in the result files.*")
        lines.append("")

    return "\n".join(lines)


def _build_score_figure(totals: Dict[str, Any]):
    import plotly.express as px

    score_df = pd.DataFrame(
        [
            {"model": "base", "metric": "per-round accuracy", "pct": totals["base"]["per_round_accuracy_pct"]},
            {"model": "test", "metric": "per-round accuracy", "pct": totals["test"]["per_round_accuracy_pct"]},
            {"model": "base", "metric": "majority accuracy", "pct": totals["base"]["majority_accuracy_pct"]},
            {"model": "test", "metric": "majority accuracy", "pct": totals["test"]["majority_accuracy_pct"]},
        ]
    )
    score_df["label"] = score_df["pct"].map(lambda v: f"{v:.1f}%")
    fig = px.bar(
        score_df,
        x="metric",
        y="pct",
        color="model",
        barmode="group",
        text="label",
        color_discrete_map={"base": PLOT_COLORS["base"], "test": PLOT_COLORS["test"]},
        title="Score snapshot: base vs test",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        yaxis_title="accuracy percent",
        xaxis_title="",
        yaxis_range=[0, 105],
        legend_title_text="model side",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=70, r=30, b=50, l=50),
    )
    fig.update_yaxes(gridcolor="#e8eef5")
    return fig


def _build_hit_heatmap(detail_df: pd.DataFrame):
    if detail_df.empty:
        return None
    import plotly.graph_objects as go

    heat_df = detail_df.copy()
    heat_df["row_label"] = heat_df["model_side"] + " R" + heat_df["round"].astype(int).astype(str)
    row_order: List[str] = []
    for side in ("base", "test"):
        rounds = sorted(
            heat_df.loc[heat_df["model_side"] == side, "round"].dropna().astype(int).unique()
        )
        for round_idx in rounds:
            row_order.append(f"{side} R{round_idx}")
    question_order = sorted(heat_df["question_index"].dropna().astype(int).unique())
    pivot = (
        heat_df.pivot_table(index="row_label", columns="question_index", values="hit", aggfunc="max")
        .reindex(row_order)
        .reindex(question_order, axis=1)
    )
    answer_pivot = (
        heat_df.pivot_table(
            index="row_label", columns="question_index", values="extracted_answer", aggfunc="first"
        )
        .reindex(row_order)
        .reindex(question_order, axis=1)
    )
    hover = []
    for row in pivot.index:
        hover_row = []
        for col in pivot.columns:
            hit = pivot.loc[row, col]
            ans = answer_pivot.loc[row, col]
            status = "correct" if hit == 1 else "miss"
            hover_row.append(f"{row}<br>Q{int(col)}<br>answer: {ans}<br>{status}")
        hover.append(hover_row)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.fillna(0).values,
            x=[f"Q{int(q)}" for q in pivot.columns],
            y=list(pivot.index),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            zmin=0,
            zmax=1,
            colorscale=[
                [0, PLOT_COLORS["miss"]],
                [0.49, PLOT_COLORS["miss"]],
                [0.5, PLOT_COLORS["hit"]],
                [1, PLOT_COLORS["hit"]],
            ],
            colorbar=dict(tickvals=[0, 1], ticktext=["miss", "hit"]),
        )
    )
    fig.update_layout(
        title="Round-by-round hit map",
        xaxis_title="question slice",
        yaxis_title="model and round",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=70, r=30, b=50, l=90),
    )
    return fig


def render(
    result,
    *,
    max_questions: int = 8,
    preview_chars: int = 420,
    show_plotly: bool = True,
    display_inline: bool = True,
) -> Optional[str]:
    """Render the comparison dashboard as markdown + plotly figures.

    Returns the markdown string when `display_inline=False`, otherwise None
    so notebook cells don't echo the raw source below the rendered output.
    """
    detail_df, question_summaries, totals = result.analyze()

    md = _build_markdown(
        result=result,
        totals=totals,
        question_summaries=question_summaries,
        max_questions=max_questions,
        preview_chars=preview_chars,
    )

    figures = []
    if show_plotly:
        try:
            score_fig = _build_score_figure(totals)
            heat_fig = _build_hit_heatmap(detail_df)
            figures = [fig for fig in (score_fig, heat_fig) if fig is not None]
        except ImportError:
            figures = []

    if display_inline:
        try:
            from IPython.display import Markdown, display

            display(Markdown(md))
            for fig in figures:
                fig.show()
        except ImportError:
            pass

    plotly_html_sections: List[str] = []
    for idx, fig in enumerate(figures):
        plotly_html_sections.append(fig.to_html(full_html=False, include_plotlyjs=(idx == 0)))

    try:
        setattr(result, "_dashboard_markdown", md)
        setattr(result, "_dashboard_plotly_html", plotly_html_sections)
    except Exception:
        pass

    return None if display_inline else md
