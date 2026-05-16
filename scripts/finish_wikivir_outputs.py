#!/usr/bin/env python3
"""Finish plots/report from an existing Wikivir analysis output directory.

Use this when the long analysis run has already written TSV tables but crashed late
in plotting/report generation.  It intentionally does *not* read the CoNLL-U file
or rebuild millions of segments.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]


def read_table(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path, sep="\t", low_memory=False)
        except Exception as exc:
            print(f"[warning] could not read {path}: {exc}", file=sys.stderr)
    return pd.DataFrame()


def numeric_time_key(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown", "_"}:
        return float("nan")
    m = re.search(r"(?<!\d)(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})(?!\d)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(?<!\d)([0-9]{1,2})(?:\.|\s|$)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"-?\d+(?:[\.,]\d+)?", text)
    if m:
        return float(m.group(0).replace(",", "."))
    return float("nan")


def sort_time_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out["__label"] = out[col].astype(str).replace({"nan": "", "None": ""})
    out["__num"] = [numeric_time_key(v) for v in out[col].tolist()]
    out["__num_missing"] = pd.to_numeric(out["__num"], errors="coerce").isna()
    return out.sort_values(["__num_missing", "__num", "__label"], kind="mergesort")


def weighted_group_summary(df: pd.DataFrame, group_col: str, weight_col: str = "lexical_token_count") -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    metric_cols = []
    for c in df.columns:
        if c.startswith("emotion_") and (c.endswith("_per_1k") or c.endswith("_prop") or c.endswith("_count")):
            metric_cols.append(c)
        elif c.startswith("sentiment_") and (c.endswith("_per_1k") or c.endswith("_count")):
            metric_cols.append(c)
        elif c.startswith("vad_") and c.endswith("_mean"):
            metric_cols.append(c)
        elif c in {"sentiment_balance", "lexicon_coverage", "lexicon_hit_count", "lexical_token_count", "token_count"}:
            metric_cols.append(c)
    rows = []
    work = df.copy()
    work[group_col] = work[group_col].astype(str).replace({"nan": "", "None": ""})
    work = work[work[group_col].str.len() > 0]
    for value, g in work.groupby(group_col, dropna=False):
        weights = pd.to_numeric(g.get(weight_col, pd.Series(np.ones(len(g)))), errors="coerce").fillna(0).to_numpy(dtype=float)
        if not weights.any():
            weights = np.ones(len(g), dtype=float)
        row = {group_col: value, "n_documents": len(g), "weight_sum": float(weights.sum())}
        for c in metric_cols:
            vals = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            mask = ~np.isnan(vals)
            if not mask.any():
                row[c] = np.nan
            elif c.endswith("_count") or c in {"lexicon_hit_count", "lexical_token_count", "token_count"}:
                row[c] = float(np.nansum(vals))
            else:
                w = weights[mask]
                row[c] = float(np.average(vals[mask], weights=w)) if w.any() else float(np.nanmean(vals[mask]))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_outputs(outdir: Path, tables: dict[str, pd.DataFrame]) -> list[str]:
    import matplotlib.pyplot as plt

    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    warnings: list[str] = []
    doc_scores = tables.get("document_emotion_scores", pd.DataFrame())
    lda_topic_summary = tables.get("lda_topic_summary", pd.DataFrame())
    transformer_predictions = tables.get("transformer_topic_predictions", pd.DataFrame())
    transformer_clusters = tables.get("transformer_embedding_clusters", pd.DataFrame())
    bertopic_topic_summary = tables.get("bertopic_topic_summary", pd.DataFrame())

    def safe(name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:
            warnings.append(f"{name}: {exc}")
            try:
                plt.close("all")
            except Exception:
                pass

    def emotion_profile() -> None:
        if doc_scores.empty:
            return
        vals = [float(pd.to_numeric(doc_scores.get(f"emotion_{e}_count", pd.Series(dtype=float)), errors="coerce").sum()) for e in EMOTIONS]
        if sum(vals) <= 0:
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(EMOTIONS, vals)
        ax.set_title("Corpus emotion profile: SloEmoLex counts")
        ax.set_ylabel("Weighted lexicon count")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        path = plots_dir / "emotion_profile_counts.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    def vad_by_time() -> None:
        if doc_scores.empty:
            return
        for time_col in ("year", "century", "period"):
            if time_col in doc_scores.columns and doc_scores[time_col].astype(str).replace("", np.nan).dropna().nunique() > 1:
                group = weighted_group_summary(doc_scores, time_col)
                if group.empty or "vad_valence_mean" not in group.columns:
                    continue
                group = sort_time_df(group, time_col)
                x = group["__label"].astype(str)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(x, group["vad_valence_mean"], marker="o", label="valence")
                if "vad_arousal_mean" in group.columns:
                    ax.plot(x, group["vad_arousal_mean"], marker="o", label="arousal")
                if "vad_dominance_mean" in group.columns:
                    ax.plot(x, group["vad_dominance_mean"], marker="o", label="dominance")
                ax.set_title(f"VAD by {time_col}")
                ax.set_ylabel("Mean score over lexicon hits")
                ax.tick_params(axis="x", rotation=35)
                ax.legend()
                fig.tight_layout()
                path = plots_dir / f"vad_by_{time_col}.png"
                fig.savefig(path, dpi=160)
                plt.close(fig)
                written.append(str(path))
                break

    def lda_prevalence() -> None:
        if lda_topic_summary.empty or "mean_probability" not in lda_topic_summary.columns:
            return
        tmp = lda_topic_summary.copy().head(30)
        labels = tmp.get("topic_label", pd.Series([""] * len(tmp))).astype(str).str.slice(0, 35)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(tmp)), pd.to_numeric(tmp["mean_probability"], errors="coerce"))
        ax.set_xticks(range(len(tmp)))
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_title("LDA topic prevalence")
        ax.set_ylabel("Mean topic probability")
        fig.tight_layout()
        path = plots_dir / "lda_topic_prevalence.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    def transformer_labels() -> None:
        if transformer_predictions.empty or "rank" not in transformer_predictions.columns or "label" not in transformer_predictions.columns:
            return
        top = transformer_predictions[pd.to_numeric(transformer_predictions["rank"], errors="coerce") == 1]
        counts = top["label"].value_counts().head(20)
        if counts.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(counts)), counts.values)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index.astype(str), rotation=60, ha="right")
        ax.set_title("Transformer topic classifier: top labels")
        ax.set_ylabel("Segment count")
        fig.tight_layout()
        path = plots_dir / "transformer_topic_labels.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    def transformer_pca() -> None:
        if transformer_clusters.empty or not {"pca_x", "pca_y", "transformer_cluster"}.issubset(transformer_clusters.columns):
            return
        tmp = transformer_clusters.copy()
        codes = pd.Categorical(tmp["transformer_cluster"]).codes
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(pd.to_numeric(tmp["pca_x"], errors="coerce"), pd.to_numeric(tmp["pca_y"], errors="coerce"), c=codes, s=18)
        ax.set_title("Transformer embedding clusters, PCA projection")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, label="cluster")
        fig.tight_layout()
        path = plots_dir / "transformer_cluster_pca.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    def bertopic_prevalence() -> None:
        if bertopic_topic_summary.empty or "segment_count" not in bertopic_topic_summary.columns:
            return
        tmp = bertopic_topic_summary.copy()
        if "is_outlier_topic" in tmp.columns:
            tmp = tmp[~tmp["is_outlier_topic"].astype(bool)]
        tmp = tmp.sort_values("segment_count", ascending=False).head(30)
        if tmp.empty:
            return
        labels = tmp.get("topic_label", pd.Series([""] * len(tmp))).astype(str).str.slice(0, 35)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(tmp)), pd.to_numeric(tmp["segment_count"], errors="coerce"))
        ax.set_xticks(range(len(tmp)))
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_title("BERTopic topic prevalence")
        ax.set_ylabel("Segment count")
        fig.tight_layout()
        path = plots_dir / "bertopic_topic_prevalence.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    for name, fn in [
        ("emotion profile", emotion_profile),
        ("VAD by time", vad_by_time),
        ("LDA prevalence", lda_prevalence),
        ("transformer labels", transformer_labels),
        ("transformer PCA", transformer_pca),
        ("BERTopic prevalence", bertopic_prevalence),
    ]:
        safe(name, fn)

    if warnings:
        (outdir / "finish_warnings.txt").write_text("\n".join(warnings), encoding="utf-8")
        for w in warnings:
            print(f"[warning] {w}", file=sys.stderr)
    return written



def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return ""
    work = df.head(max_rows) if max_rows else df
    cols = list(work.columns)
    lines = []
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in work.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.4g}")
            else:
                vals.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def write_report(outdir: Path, tables: dict[str, pd.DataFrame], plots: list[str]) -> Path:
    doc_scores = tables.get("document_emotion_scores", pd.DataFrame())
    lda_topic_summary = tables.get("lda_topic_summary", pd.DataFrame())
    transformer_predictions = tables.get("transformer_topic_predictions", pd.DataFrame())
    transformer_clusters = tables.get("transformer_embedding_clusters", pd.DataFrame())
    bertopic_topic_summary = tables.get("bertopic_topic_summary", pd.DataFrame())

    lines: list[str] = []
    lines.append("# Wikivir emotion and topic analysis report")
    lines.append("")
    lines.append(f"Finished from existing TSV outputs: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Outputs used")
    lines.append("")
    for name, df in tables.items():
        lines.append(f"- `{name}.tsv`: {len(df):,} rows, {len(df.columns):,} columns")
    lines.append("")

    if not doc_scores.empty:
        lines.append("## Corpus-level emotion profile")
        lines.append("")
        totals = []
        for emo in EMOTIONS:
            val = float(pd.to_numeric(doc_scores.get(f"emotion_{emo}_count", pd.Series(dtype=float)), errors="coerce").sum())
            totals.append((emo, val))
        total_sum = sum(v for _, v in totals)
        lines.append("| emotion | count | proportion |")
        lines.append("|---|---:|---:|")
        for emo, val in sorted(totals, key=lambda x: x[1], reverse=True):
            lines.append(f"| {emo} | {val:.2f} | {(val / total_sum if total_sum else 0):.3f} |")
        if "lexicon_coverage" in doc_scores.columns:
            cov = pd.to_numeric(doc_scores["lexicon_coverage"], errors="coerce").mean()
            lines.append("")
            lines.append(f"Mean document lexicon coverage: **{cov:.3f}**")
        lines.append("")

    if not lda_topic_summary.empty:
        lines.append("## LDA topics")
        lines.append("")
        cols = [c for c in ["topic_id", "topic_label", "mean_probability", "segment_count", "doc_count"] if c in lda_topic_summary.columns]
        lines.append(df_to_markdown(lda_topic_summary[cols], max_rows=15) if cols else f"{len(lda_topic_summary):,} topic summary rows.")
        lines.append("")

    if not bertopic_topic_summary.empty:
        lines.append("## BERTopic topics")
        lines.append("")
        cols = [c for c in ["topic_id", "topic_label", "segment_count", "doc_count", "mean_probability"] if c in bertopic_topic_summary.columns]
        lines.append(df_to_markdown(bertopic_topic_summary[cols], max_rows=15) if cols else f"{len(bertopic_topic_summary):,} BERTopic summary rows.")
        lines.append("")
    else:
        lines.append("## BERTopic topics")
        lines.append("")
        lines.append("BERTopic outputs are absent. In the logged run this was because the `bertopic` Python package was not installed.")
        lines.append("")

    if not transformer_predictions.empty and "label" in transformer_predictions.columns:
        lines.append("## Transformer topic classifier")
        lines.append("")
        top = transformer_predictions
        if "rank" in top.columns:
            top = top[pd.to_numeric(top["rank"], errors="coerce") == 1]
        counts = top["label"].value_counts().head(20)
        lines.append(df_to_markdown(counts.rename_axis("label").reset_index(name="segments")))
        lines.append("")

    if not transformer_clusters.empty:
        lines.append("## Transformer embedding clusters")
        lines.append("")
        lines.append(f"Embedding-cluster table: **{len(transformer_clusters):,} rows**.")
        if "transformer_cluster" in transformer_clusters.columns:
            counts = transformer_clusters["transformer_cluster"].value_counts().head(20)
            lines.append(df_to_markdown(counts.rename_axis("cluster").reset_index(name="segments")))
        lines.append("")

    if plots:
        lines.append("## Plots")
        lines.append("")
        for p in plots:
            lines.append(f"- `{Path(p).relative_to(outdir) if Path(p).is_relative_to(outdir) else p}`")
        lines.append("")

    report = outdir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Finish Wikivir plots/report from existing analysis TSV outputs.")
    ap.add_argument("--output-dir", type=Path, required=True, help="Existing analysis output directory, e.g. analysis/wikivir_full_stack")
    args = ap.parse_args()
    outdir = args.output_dir
    tables_dir = outdir / "tables"
    if not tables_dir.exists():
        raise SystemExit(f"Missing tables directory: {tables_dir}")
    names = [
        "document_emotion_scores",
        "segment_emotion_scores",
        "lda_topic_summary",
        "transformer_topic_predictions",
        "transformer_embedding_clusters",
        "bertopic_topic_summary",
    ]
    tables = {name: read_table(tables_dir / f"{name}.tsv") for name in names}
    plots = plot_outputs(outdir, tables)
    report = write_report(outdir, tables, plots)
    manifest = outdir / "finish_manifest.json"
    manifest.write_text(json.dumps({"report": str(report), "plots": plots, "finished_at": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    print(f"Finished. Report: {report}", file=sys.stderr)
    print(f"Plots written: {len(plots)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
