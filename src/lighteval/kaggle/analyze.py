from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def _first_existing(paths: List[str]):
    for p in paths:
        if p and Path(p).exists():
            return p
    return None


def _extract_text(resp: Any) -> str:
    try:
        text = resp["text"]
    except Exception:
        text = getattr(resp, "text", resp)
    if isinstance(text, (list, tuple)):
        return str(text[0]) if text else ""
    try:
        values = list(text)
        return str(values[0]) if values else ""
    except Exception:
        return str(text)


def _extract_answer(text: str) -> str:
    m = re.search(r"Answer:\s*([A-Z])", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-J])\b", text)
    return m.group(1) if m else "?"


def _metric_hit(metric: Any) -> int:
    if isinstance(metric, dict):
        for key in ("extractive_match", "exact_match", "accuracy"):
            if key in metric:
                return int(metric[key])
    return 0


def _choice_map_from_query(query: str) -> Dict[str, str]:
    return dict(re.findall(r"^([A-Z]):\s*(.+?)$", str(query), re.MULTILINE))


def analyze_pair(
    base_paths: List[str],
    test_paths: List[str],
    base_model_name: str,
    test_model_name: str,
    run_name: str,
    task: str,
    samples_start: int = 0,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """Parse paired details parquets into (detail_df, question_summaries, totals)."""

    ref_path = _first_existing(base_paths + test_paths)
    if not ref_path:
        raise RuntimeError(
            "No lighteval detail parquet files were produced. "
            "Check the pipeline output for errors."
        )

    ref_df = pd.read_parquet(ref_path)
    base_dfs = [pd.read_parquet(p) if p and Path(p).exists() else None for p in base_paths]
    test_dfs = [pd.read_parquet(p) if p and Path(p).exists() else None for p in test_paths]

    rows: List[Dict[str, Any]] = []
    question_summaries: List[Dict[str, Any]] = []

    for q_idx in range(len(ref_df)):
        doc = ref_df.iloc[q_idx].get("doc", {})
        query = doc.get("query", "") if isinstance(doc, dict) else str(doc)
        gold_idx = doc.get("gold_index", None) if isinstance(doc, dict) else None
        gold_letter = chr(ord("A") + int(gold_idx)) if gold_idx is not None else "?"
        choices = _choice_map_from_query(query)
        gold_text = choices.get(gold_letter, "")
        qbody_match = re.search(r"question\.\s*\n+(.*?)(?=\n[A-Z]:)", query, re.DOTALL)
        qbody = qbody_match.group(1).strip() if qbody_match else str(query)[:500]

        per_side: Dict[str, Dict[str, Any]] = {}
        for side, dfs, model_name in [
            ("base", base_dfs, base_model_name),
            ("test", test_dfs, test_model_name),
        ]:
            answers: List[str] = []
            hits: List[int] = []
            for round_num, df in enumerate(dfs, start=1):
                if df is None or q_idx >= len(df):
                    continue
                row = df.iloc[q_idx]
                text = _extract_text(row.get("model_response", {}))
                answer = _extract_answer(text)
                hit = _metric_hit(row.get("metric", {}))
                answers.append(answer)
                hits.append(hit)
                rows.append(
                    {
                        "run_name": run_name,
                        "task": task,
                        "model_side": side,
                        "model_name": model_name,
                        "round": round_num,
                        "samples_start": samples_start,
                        "question_index": samples_start + q_idx,
                        "question_body": qbody,
                        "gold_letter": gold_letter,
                        "gold_text": gold_text,
                        "extracted_answer": answer,
                        "hit": hit,
                        "full_text": text,
                    }
                )
            total = len(hits)
            correct = sum(hits)
            majority = Counter(answers).most_common(1)[0][0] if answers else "?"
            per_side[side] = {
                "answers": answers,
                "hits": hits,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
                "majority_answer": majority,
                "majority_hit": int(majority == gold_letter) if gold_letter != "?" else 0,
            }

        question_summaries.append(
            {
                "question_index": samples_start + q_idx,
                "question": qbody,
                "gold_letter": gold_letter,
                "gold_text": gold_text,
                "base": per_side["base"],
                "test": per_side["test"],
            }
        )

    detail_df = pd.DataFrame(rows)
    totals: Dict[str, Any] = {}
    for side in ("base", "test"):
        side_df = detail_df[detail_df.model_side == side]
        total = len(side_df)
        correct = int(side_df.hit.sum()) if total else 0
        q_majority = sum(q[side]["majority_hit"] for q in question_summaries)
        totals[side] = {
            "model": base_model_name if side == "base" else test_model_name,
            "samples": total,
            "correct": correct,
            "per_round_accuracy_pct": round(100 * correct / total, 2) if total else 0.0,
            "majority_correct": int(q_majority),
            "questions": len(question_summaries),
            "majority_accuracy_pct": round(100 * q_majority / len(question_summaries), 2)
            if question_summaries
            else 0.0,
        }
    totals["delta_pp"] = round(
        totals["test"]["per_round_accuracy_pct"] - totals["base"]["per_round_accuracy_pct"], 2
    )
    totals["majority_delta_pp"] = round(
        totals["test"]["majority_accuracy_pct"] - totals["base"]["majority_accuracy_pct"], 2
    )

    return detail_df, question_summaries, totals
