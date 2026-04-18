from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List


_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def _slug(value: str) -> str:
    return _SLUG_RE.sub("-", value.strip()).strip("-").lower() or "item"


def _question_dir(root: Path, index: int, task: str) -> Path:
    return root / f"q{index:04d}_{_slug(task)}"


def _verdict(hit: int, answer: str) -> str:
    if hit:
        return "hit"
    if answer == "?":
        return "unknown"
    return "miss"


def dump(result, *, include_responses: bool = True) -> Path:
    """Write a research-grade per-question dump under `<output_dir>/research`.

    Each question gets its own folder with:
      - question.md  (prompt, gold, per-round summary table)
      - {side}_round{N}.md  (prompt + full model response + extracted answer)

    Returns the path to the research root directory.
    """
    if result.output_dir is None:
        raise RuntimeError("result.output_dir is unset — run the pipeline first.")

    detail_df, question_summaries, totals = result.analyze()
    research_root = Path(result.output_dir) / "research"
    research_root.mkdir(parents=True, exist_ok=True)

    _write_index(research_root, result, totals, question_summaries)

    for q in question_summaries:
        qdir = _question_dir(research_root, int(q["question_index"]), result.task)
        qdir.mkdir(parents=True, exist_ok=True)
        _write_question_summary(qdir, result, q)
        if include_responses:
            _write_round_files(qdir, result, detail_df, q)

    print(f"Research dump written to {research_root}")
    return research_root


def _write_index(root: Path, result, totals: Dict[str, Any], questions: List[Dict[str, Any]]) -> None:
    lines = [
        f"# Research index — `{result.run_name}`",
        "",
        f"Task `{result.task}` · rounds `{result.rounds}` · "
        f"slice starts at `{result.samples_start}` · "
        f"questions `{len(questions)}`",
        "",
        f"Base: `{result.base_source}`  · Test: `{result.test_source}`",
        "",
        "## Totals",
        "",
        "| Side | Per-round | Majority | Correct |",
        "|---|---:|---:|---:|",
    ]
    for side in ("base", "test"):
        lines.append(
            f"| {side} | {totals[side]['per_round_accuracy_pct']:.2f}% | "
            f"{totals[side]['majority_accuracy_pct']:.2f}% | "
            f"{totals[side]['correct']}/{totals[side]['samples']} |"
        )
    lines += [
        "",
        f"Per-round Δ **{totals['delta_pp']:+.2f} pp** · "
        f"Majority Δ **{totals['majority_delta_pp']:+.2f} pp**",
        "",
        "## Questions",
        "",
        "| Index | Gold | Base majority | Test majority | Folder |",
        "|---:|:---:|:---:|:---:|---|",
    ]
    for q in questions:
        idx = int(q["question_index"])
        folder = _question_dir(root, idx, result.task).name
        lines.append(
            f"| {idx} | `{q['gold_letter']}` | "
            f"`{q['base']['majority_answer']}` | "
            f"`{q['test']['majority_answer']}` | [{folder}]({folder}/question.md) |"
        )
    (root / "README.md").write_text("\n".join(lines) + "\n")


def _write_question_summary(qdir: Path, result, q: Dict[str, Any]) -> None:
    idx = int(q["question_index"])
    lines = [
        f"# Question {idx} — {result.task}",
        "",
        f"**Gold** `{q['gold_letter']}` — {q['gold_text'] or '—'}",
        "",
        "## Prompt",
        "",
        q["question"],
        "",
        "## Per-round summary",
        "",
        "| Side | Round | Extracted | Hit |",
        "|---|---:|:---:|:---:|",
    ]
    for side in ("base", "test"):
        data = q[side]
        for round_idx, (answer, hit) in enumerate(zip(data["answers"], data["hits"]), start=1):
            lines.append(f"| {side} | {round_idx} | `{answer}` | {_verdict(hit, answer)} |")
    lines += [
        "",
        f"Base majority: `{q['base']['majority_answer']}` "
        f"({sum(q['base']['hits'])}/{q['base']['total']} hits)",
        "",
        f"Test majority: `{q['test']['majority_answer']}` "
        f"({sum(q['test']['hits'])}/{q['test']['total']} hits)",
        "",
    ]
    (qdir / "question.md").write_text("\n".join(lines))


def _write_round_files(qdir: Path, result, detail_df, q: Dict[str, Any]) -> None:
    idx = int(q["question_index"])
    rows = detail_df[detail_df["question_index"] == idx]
    for _, row in rows.iterrows():
        side = row["model_side"]
        round_idx = int(row["round"])
        hit = int(row.get("hit", 0))
        answer = str(row.get("extracted_answer", "?"))
        filename = f"{side}_round{round_idx}.md"
        body = [
            f"# {side.title()} · Round {round_idx}",
            "",
            f"Model: `{row['model_name']}`",
            f"Question: {idx} ({result.task})",
            f"Gold: `{q['gold_letter']}` — {q['gold_text'] or '—'}",
            f"Extracted: `{answer}` · {_verdict(hit, answer)}",
            "",
            "## Prompt",
            "",
            str(row.get("question_body", q["question"])),
            "",
            "## Response",
            "",
            str(row.get("full_text", "")),
            "",
        ]
        (qdir / filename).write_text("\n".join(body))
