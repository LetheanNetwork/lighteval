# Render the Gemma 4 comparison dashboard.
#
# Produces an HTML + plotly dashboard suitable for Jupyter / Kaggle
# notebooks. The dashboard walks a reader through one evaluation slice
# from prompt window -> rounds -> extracted answers -> majority vote ->
# delta, using colour-coded cards and a heatmap.
#
# Called via Gemma4EvalResult.dashboard(). Can also be called directly
# with a Gemma4EvalResult instance:
#
#     from lighteval.kaggle.dashboard import render
#     html = render(result)
from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .analyze import analyze_pair


PLOT_COLORS = {
    "base": "#2f6fed",
    "test": "#19a974",
    "right": "#19a974",
    "miss": "#e5484d",
    "unknown": "#d5a100",
}


# ----------------------------------------------------------------------
# Small presentation helpers.
# ----------------------------------------------------------------------
def _pretty_model_name(model_path: str) -> str:
    """Try to surface a readable model name from a filesystem path."""
    parts = Path(str(model_path)).parts
    if "transformers" in parts:
        idx = parts.index("transformers")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if len(parts) >= 2 and parts[-1].isdigit():
        return parts[-2]
    return Path(str(model_path)).name or str(model_path)


def _clamp_pct(value: Any) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except Exception:
        return 0.0


# ----------------------------------------------------------------------
# HTML building blocks.
# ----------------------------------------------------------------------
def _score_card(side: str, data: Dict[str, Any]) -> str:
    name = _pretty_model_name(data["model"])
    per_round = _clamp_pct(data["per_round_accuracy_pct"])
    majority = _clamp_pct(data["majority_accuracy_pct"])
    return f'''
    <div class="ge-card">
      <div class="ge-kicker">{escape(side)}</div>
      <h3>{escape(name)}</h3>
      <div class="ge-muted">{escape(str(data["model"]))}</div>
      <div class="ge-score">{per_round:.1f}%</div>
      <div class="ge-label">per-round accuracy</div>
      <div class="ge-bar"><span style="width:{per_round:.2f}%"></span></div>
      <div class="ge-mini-row"><span>correct samples</span><b>{int(data["correct"])}/{int(data["samples"])}</b></div>
      <div class="ge-mini-row"><span>majority vote</span><b>{majority:.1f}%</b></div>
      <div class="ge-bar ge-bar-alt"><span style="width:{majority:.2f}%"></span></div>
    </div>'''


def _delta_card(label: str, delta: float) -> str:
    delta = float(delta)
    tone = "pos" if delta > 0 else "neg" if delta < 0 else "flat"
    sign = "+" if delta > 0 else ""
    return f'''
    <div class="ge-card ge-delta ge-{tone}">
      <div class="ge-kicker">delta</div>
      <h3>{escape(label)}</h3>
      <div class="ge-score">{sign}{delta:.2f} pp</div>
      <div class="ge-muted">test model minus base model</div>
    </div>'''


def _answer_distribution(answers: List[str], gold_letter: str) -> str:
    total = max(1, len(answers))
    chunks = []
    for answer, count in sorted(Counter(answers).items(), key=lambda kv: (-kv[1], str(kv[0]))):
        width = 100 * count / total
        cls = "right" if answer == gold_letter else "miss" if answer != "?" else "unknown"
        chunks.append(
            f'<div class="ge-dist-row"><span class="ge-dist-answer ge-{cls}">{escape(str(answer))}</span>'
            f'<div class="ge-dist-track"><span class="ge-{cls}" style="width:{width:.2f}%"></span></div>'
            f"<b>{count}/{total}</b></div>"
        )
    return "".join(chunks) or '<div class="ge-muted">No answers captured.</div>'


def _round_pills(answers: List[str], hits: List[int], gold_letter: str) -> str:
    pills = []
    for idx, (answer, hit) in enumerate(zip(answers, hits), start=1):
        cls = "right" if hit else "miss" if answer != "?" else "unknown"
        title = f"round {idx}: answer {answer}, gold {gold_letter}"
        pills.append(
            f'<span class="ge-pill ge-{cls}" title="{escape(title)}">'
            f'R{idx}: {escape(str(answer))}</span>'
        )
    return "".join(pills) or '<span class="ge-muted">No rounds captured.</span>'


def _model_slice(side: str, q: Dict[str, Any]) -> str:
    data = q[side]
    majority = data["majority_answer"]
    majority_cls = (
        "right" if majority == q["gold_letter"] else "miss" if majority != "?" else "unknown"
    )
    return f'''
    <div class="ge-model-slice">
      <div class="ge-model-head">
        <b>{escape(side)}</b>
        <span>majority: <b class="ge-{majority_cls}">{escape(str(majority))}</b></span>
        <span>hits: <b>{sum(data["hits"])}/{data["total"]}</b></span>
      </div>
      <div class="ge-pills">{_round_pills(data["answers"], data["hits"], q["gold_letter"])}</div>
      {_answer_distribution(data["answers"], q["gold_letter"])}
    </div>'''


def _render_question_slice(q: Dict[str, Any], preview_chars: int) -> str:
    question = escape(q["question"][:preview_chars])
    gold = escape(f"{q['gold_letter']}: {q['gold_text']}")
    return f'''
    <section class="ge-question">
      <div class="ge-question-head">
        <div class="ge-kicker">question slice {int(q["question_index"])}</div>
        <div class="ge-gold">gold answer: <b>{gold}</b></div>
      </div>
      <p>{question}</p>
      <div class="ge-model-grid">
        {_model_slice("base", q)}
        {_model_slice("test", q)}
      </div>
    </section>'''


# ----------------------------------------------------------------------
# Plotly helpers.
# ----------------------------------------------------------------------
def _build_score_figure(totals: Dict[str, Any]):
    import plotly.express as px  # lazy import; plotly is optional

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
                [0.5, PLOT_COLORS["right"]],
                [1, PLOT_COLORS["right"]],
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


# ----------------------------------------------------------------------
# Public entrypoint.
# ----------------------------------------------------------------------
_DASHBOARD_CSS = """
<style>
  .ge-wrap { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #18212f; }
  .ge-wrap * { box-sizing: border-box; }
  .ge-hero { border: 1px solid #d5dde8; border-radius: 8px; padding: 18px; background: linear-gradient(135deg, #f7fbff, #f7fff9); margin: 14px 0; }
  .ge-hero h2 { margin: 0 0 8px; font-size: 24px; }
  .ge-hero p { margin: 0; color: #44546a; }
  .ge-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 12px 0; }
  .ge-card { border: 1px solid #d5dde8; border-radius: 8px; padding: 14px; background: #ffffff; box-shadow: 0 1px 2px rgba(24, 33, 47, 0.06); }
  .ge-card h3 { margin: 4px 0 6px; font-size: 16px; overflow-wrap: anywhere; }
  .ge-kicker { color: #52647a; text-transform: uppercase; font-size: 11px; letter-spacing: .08em; font-weight: 700; }
  .ge-muted { color: #66758a; font-size: 12px; overflow-wrap: anywhere; }
  .ge-score { margin-top: 12px; font-size: 34px; line-height: 1; font-weight: 800; }
  .ge-label { color: #52647a; font-size: 12px; margin: 2px 0 8px; }
  .ge-bar, .ge-dist-track { height: 10px; border-radius: 6px; overflow: hidden; background: #e8eef5; }
  .ge-bar span, .ge-dist-track span { display: block; height: 100%; background: #2f6fed; border-radius: 6px; }
  .ge-bar-alt span { background: #19a974; }
  .ge-mini-row { display: flex; justify-content: space-between; gap: 10px; margin-top: 8px; color: #334155; font-size: 13px; }
  .ge-delta.ge-pos { border-color: #8fd8b0; background: #f2fff7; }
  .ge-delta.ge-neg { border-color: #f1a1a1; background: #fff6f6; }
  .ge-delta.ge-flat { border-color: #d5dde8; background: #fafcff; }
  .ge-flow { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin: 12px 0; }
  .ge-step { border: 1px dashed #b9c6d6; border-radius: 8px; padding: 10px; background: #fbfdff; }
  .ge-step b { display: block; margin-bottom: 4px; }
  .ge-question { border: 1px solid #d5dde8; border-radius: 8px; padding: 14px; background: #ffffff; margin: 12px 0; }
  .ge-question p { margin: 10px 0; line-height: 1.45; color: #243244; }
  .ge-question-head { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
  .ge-gold { color: #243244; }
  .ge-model-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
  .ge-model-slice { border: 1px solid #e0e7ef; border-radius: 8px; padding: 12px; background: #fbfdff; }
  .ge-model-head { display: flex; gap: 12px; justify-content: space-between; flex-wrap: wrap; font-size: 13px; margin-bottom: 8px; }
  .ge-pills { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
  .ge-pill { border-radius: 8px; padding: 4px 8px; font-size: 12px; font-weight: 700; border: 1px solid transparent; }
  .ge-right { color: #047857; }
  .ge-miss { color: #b42318; }
  .ge-unknown { color: #8a6100; }
  .ge-pill.ge-right { background: #e9fbf1; border-color: #8fd8b0; }
  .ge-pill.ge-miss { background: #fff0f0; border-color: #f1a1a1; }
  .ge-pill.ge-unknown { background: #fff8db; border-color: #edd37a; }
  .ge-dist-row { display: grid; grid-template-columns: 34px 1fr 42px; align-items: center; gap: 8px; margin-top: 6px; font-size: 12px; }
  .ge-dist-answer { font-weight: 800; }
  .ge-dist-track .ge-right { background: #19a974; }
  .ge-dist-track .ge-miss { background: #e5484d; }
  .ge-dist-track .ge-unknown { background: #d5a100; }
  .ge-note { color: #52647a; font-size: 13px; margin-top: 10px; }
</style>
""".strip()


def render(
    result,
    *,
    max_questions: int = 8,
    preview_chars: int = 420,
    show_plotly: bool = True,
    display_inline: bool = True,
) -> str:
    """Render the Gemma 4 comparison dashboard for a Gemma4EvalResult.

    Parameters
    ----------
    result : Gemma4EvalResult
        The outcome object returned by `Gemma4Eval().run()`.
    max_questions : int, default 8
        Cap question slices shown inline to keep the dashboard readable.
    preview_chars : int, default 420
        Max chars of each question body to show inline.
    show_plotly : bool, default True
        If True, render the score bar chart + hit heatmap.
    display_inline : bool, default True
        If True, call IPython.display.HTML — suitable for notebooks. Set
        False if you just want the HTML string (e.g. to write to file).

    Returns
    -------
    str : the dashboard HTML (caller can save to disk if desired).
    """

    detail_df, question_summaries, totals = result.analyze()

    hw_plan = getattr(result, "hardware_plan", None) or "auto"
    run_name = escape(str(result.run_name))
    task = escape(str(result.task))
    rounds = int(result.rounds)
    # n_questions is the per-round sample count from Gemma4Eval
    n_questions = getattr(result, "n_questions", len(question_summaries))
    samples_start = getattr(result, "samples_start", 0)

    shown_questions = question_summaries[:max_questions]
    hidden = max(0, len(question_summaries) - len(shown_questions))
    hidden_note = ""
    if hidden:
        hidden_note = (
            f'<div class="ge-note">Showing the first {len(shown_questions)} '
            f"question slices. {hidden} more are saved in the result files.</div>"
        )

    html = f"""
{_DASHBOARD_CSS}
<div class="ge-wrap">
  <div class="ge-hero">
    <h2>Gemma 4 comparison dashboard — {run_name}</h2>
    <p>This view follows one evaluation slice from prompt window to model answers, extracted labels, majority vote, and final delta.</p>
  </div>
  <div class="ge-grid">
    <div class="ge-card"><div class="ge-kicker">task</div><h3>{task}</h3><div class="ge-muted">sample window starts at {int(samples_start)}</div></div>
    <div class="ge-card"><div class="ge-kicker">slice size</div><h3>{int(n_questions)} questions x {rounds} rounds</h3><div class="ge-muted">{int(n_questions) * rounds} samples per model</div></div>
    <div class="ge-card"><div class="ge-kicker">hardware plan</div><h3>{escape(str(hw_plan))}</h3><div class="ge-muted">same questions, same sampling settings</div></div>
  </div>
  <div class="ge-grid">
    {_score_card("base", totals["base"])}
    {_score_card("test", totals["test"])}
    {_delta_card("per-round accuracy", totals["delta_pp"])}
    {_delta_card("majority accuracy", totals["majority_delta_pp"])}
  </div>
  <div class="ge-card">
    <div class="ge-kicker">how the data moves</div>
    <div class="ge-flow">
      <div class="ge-step"><b>1. Prompt slice</b><span>A small task window is selected via samples_start and N_QUESTIONS.</span></div>
      <div class="ge-step"><b>2. Repeated rounds</b><span>Each model answers the same prompt ROUNDS times with sampled decoding.</span></div>
      <div class="ge-step"><b>3. Extracted answer</b><span>The evaluator reads the final A-J answer from each model response.</span></div>
      <div class="ge-step"><b>4. Vote and delta</b><span>Correct samples and majority votes become the comparison scores.</span></div>
    </div>
  </div>
  {"".join(_render_question_slice(q, preview_chars) for q in shown_questions)}
  {hidden_note}
</div>
"""

    plotly_html_sections: List[str] = []
    figures = []
    if show_plotly:
        try:
            score_fig = _build_score_figure(totals)
            heat_fig = _build_hit_heatmap(detail_df)
            figures = [fig for fig in (score_fig, heat_fig) if fig is not None]
        except ImportError:
            # plotly not installed — dashboard HTML still renders
            figures = []

    if display_inline:
        try:
            from IPython.display import HTML, display  # type: ignore

            display(HTML(html))
            for fig in figures:
                fig.show()
        except ImportError:
            pass

    for idx, fig in enumerate(figures):
        plotly_html_sections.append(fig.to_html(full_html=False, include_plotlyjs=(idx == 0)))

    # Stash the plotly HTML on the result so save_report can embed it.
    try:
        setattr(result, "_dashboard_plotly_html", plotly_html_sections)
        setattr(result, "_dashboard_html", html)
    except Exception:
        pass

    return html
