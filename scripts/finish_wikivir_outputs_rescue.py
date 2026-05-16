#!/usr/bin/env python3
"""Rescue/finish Wikivir analysis outputs after a late crash.

This script is deliberately defensive:
- finds non-empty TSV outputs even if the wrong output directory was passed;
- reconstructs final TSVs from checkpoint shards where possible;
- builds document_emotion_scores.tsv from segment_emotion_scores shards without loading
  the whole segment table into memory;
- writes plots, report.md, and a detailed rescue_manifest.json.

It does not read the CoNLL-U and does not redo CLASSLA, lexicon scoring, LDA, or transformers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
SENTIMENTS = ["positive", "negative"]
TABLE_NAMES = [
    "document_emotion_scores",
    "segment_emotion_scores",
    "lda_topics",
    "lda_segment_topics",
    "lda_topic_summary",
    "transformer_topic_predictions",
    "transformer_topic_document_summary",
    "transformer_embedding_clusters",
    "transformer_cluster_summary",
    "bertopic_topics",
    "bertopic_segment_topics",
    "bertopic_topic_summary",
    "bertopic_emotion_crosswalk",
]


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def human_size(n: int | float) -> str:
    n = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:,.1f}{unit}"
        n /= 1024.0
    return f"{n:,.1f}PB"


def ensure_dirs(outdir: Path) -> dict[str, Path]:
    dirs = {
        "base": outdir,
        "tables": outdir / "tables",
        "plots": outdir / "plots",
        "checkpoints": outdir / "checkpoints",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def count_tsv_rows(path: Path) -> tuple[int, int]:
    """Return rows, columns without using pandas."""
    if not path.exists() or path.stat().st_size == 0:
        return 0, 0
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            header = f.readline()
            if not header:
                return 0, 0
            cols = len(header.rstrip("\n\r").split("\t"))
            rows = sum(1 for _ in f)
            return rows, cols
    except Exception:
        return 0, 0


def read_table(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t", low_memory=False, nrows=max_rows)
    except Exception as exc:
        eprint(f"[warning] could not read {path}: {exc}")
        return pd.DataFrame()


def concat_tsv_parts(parts: list[Path], output: Path, force: bool = False) -> int:
    parts = [p for p in parts if p.exists() and p.stat().st_size > 0]
    if not parts:
        return 0
    if output.exists() and output.stat().st_size > 0 and not force:
        rows, _ = count_tsv_rows(output)
        return rows
    output.parent.mkdir(parents=True, exist_ok=True)
    header: str | None = None
    rows = 0
    with output.open("w", encoding="utf-8", newline="") as out:
        for part in sorted(parts):
            with part.open("r", encoding="utf-8", errors="replace", newline="") as f:
                first = f.readline()
                if not first:
                    continue
                if header is None:
                    header = first
                    out.write(first)
                for line in f:
                    out.write(line)
                    rows += 1
    return rows


def default_search_roots(outdir: Path) -> list[Path]:
    roots = [outdir, outdir.parent]
    cwd = Path.cwd()
    if (cwd / "analysis").exists():
        roots.append(cwd / "analysis")
    roots.append(cwd)
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for r in roots:
        try:
            rp = r.resolve()
        except Exception:
            rp = r
        if rp not in seen and r.exists():
            unique.append(r)
            seen.add(rp)
    return unique


def find_nonempty_file(filename: str, roots: list[Path], exclude: set[Path] | None = None) -> Path | None:
    exclude = exclude or set()
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob(filename):
            if p in exclude:
                continue
            try:
                if p.exists() and p.stat().st_size > 0:
                    candidates.append(p)
            except OSError:
                continue
    if not candidates:
        return None
    # prefer files inside a tables directory, then biggest/newest
    candidates.sort(key=lambda p: ("/tables/" in str(p), p.stat().st_size, p.stat().st_mtime), reverse=True)
    return candidates[0]


def copy_or_link(src: Path, dst: Path, force: bool = False) -> bool:
    if src.resolve() == dst.resolve():
        return True
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Hardlink if possible, copy otherwise. Hardlink saves disk but behaves like a normal file.
    try:
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
        return True
    except Exception:
        try:
            import shutil
            shutil.copy2(src, dst)
            return True
        except Exception as exc:
            eprint(f"[warning] could not copy {src} -> {dst}: {exc}")
            return False


def stage_existing_and_shards(outdir: Path, force: bool = False, search_roots: list[Path] | None = None, include_token_matches: bool = False) -> dict[str, Any]:
    dirs = ensure_dirs(outdir)
    tables = dirs["tables"]
    ckpt = dirs["checkpoints"]
    roots = search_roots or default_search_roots(outdir)
    manifest: dict[str, Any] = {"staged": {}, "merged_shards": {}, "search_roots": [str(r) for r in roots]}

    # First, find/copy non-empty final table files.
    for name in TABLE_NAMES:
        dst = tables / f"{name}.tsv"
        if dst.exists() and dst.stat().st_size > 0 and not force:
            rows, cols = count_tsv_rows(dst)
            manifest["staged"][name] = {"path": str(dst), "rows": rows, "columns": cols, "bytes": dst.stat().st_size, "source": "already_present"}
            continue
        src = find_nonempty_file(f"{name}.tsv", roots, exclude={dst})
        if src:
            copy_or_link(src, dst, force=True)
            rows, cols = count_tsv_rows(dst)
            manifest["staged"][name] = {"path": str(dst), "rows": rows, "columns": cols, "bytes": dst.stat().st_size, "source": str(src)}

    # Merge known checkpoint shards when final tables are absent.
    shard_specs = {
        "segment_emotion_scores": [ckpt / "lexicon"],
        "transformer_topic_predictions": [ckpt / "transformer_classifier"],
    }
    shard_patterns = {
        "segment_emotion_scores": "segment_emotion_scores.part-*.tsv",
        "transformer_topic_predictions": "transformer_topic_predictions.part-*.tsv",
    }
    if include_token_matches:
        shard_specs["token_emotion_matches"] = [ckpt / "lexicon"]
        shard_patterns["token_emotion_matches"] = "token_emotion_matches.part-*.tsv"
    # Also search more broadly because users sometimes pass the wrong output dir.
    for root in roots:
        for name in shard_specs:
            shard_specs[name].append(root)

    for name, roots_or_dirs in shard_specs.items():
        dst = tables / f"{name}.tsv"
        if dst.exists() and dst.stat().st_size > 0 and not force:
            continue
        parts: list[Path] = []
        seen = set()
        pattern = shard_patterns[name]
        for root in roots_or_dirs:
            if not root.exists():
                continue
            # Search recursively from each root; this handles both the expected
            # checkpoint directory and the "wrong output dir" case.
            try:
                candidates = root.rglob(pattern)
            except Exception:
                continue
            for p in candidates:
                try:
                    rp = p.resolve()
                    if p.exists() and p.stat().st_size > 0 and rp not in seen:
                        parts.append(p)
                        seen.add(rp)
                except Exception:
                    pass
        if parts:
            rows = concat_tsv_parts(sorted(parts), dst, force=force)
            _, cols = count_tsv_rows(dst)
            manifest["merged_shards"][name] = {"output": str(dst), "parts": len(parts), "rows": rows, "columns": cols, "bytes": dst.stat().st_size}
            manifest["staged"][name] = {"path": str(dst), "rows": rows, "columns": cols, "bytes": dst.stat().st_size, "source": "checkpoint_shards"}

    return manifest


def build_document_scores_from_segments(outdir: Path, force: bool = False) -> dict[str, Any]:
    tables = outdir / "tables"
    seg_path = tables / "segment_emotion_scores.tsv"
    doc_path = tables / "document_emotion_scores.tsv"
    result = {"built": False, "reason": ""}
    if doc_path.exists() and doc_path.stat().st_size > 0 and not force:
        rows, cols = count_tsv_rows(doc_path)
        return {"built": False, "reason": "already_present", "rows": rows, "columns": cols, "path": str(doc_path)}
    if not seg_path.exists() or seg_path.stat().st_size == 0:
        return {"built": False, "reason": "missing_segment_emotion_scores"}

    # Prefer actual document-level segment rows. Read in chunks; do not load all segment rows.
    chunks = []
    try:
        for chunk in pd.read_csv(seg_path, sep="\t", low_memory=False, chunksize=100_000):
            if "unit" in chunk.columns:
                docs = chunk[chunk["unit"].astype(str) == "document"].copy()
                if not docs.empty:
                    chunks.append(docs)
    except Exception as exc:
        return {"built": False, "reason": f"failed_reading_segments: {exc}"}

    if chunks:
        doc_df = pd.concat(chunks, ignore_index=True)
        doc_df.to_csv(doc_path, sep="\t", index=False)
        return {"built": True, "method": "document_unit_rows", "rows": len(doc_df), "columns": len(doc_df.columns), "path": str(doc_path)}

    # Fallback aggregation by doc_id. This is less pretty but enough for report/plots.
    try:
        agg: dict[str, dict[str, float | str]] = {}
        meta_first: dict[str, dict[str, Any]] = {}
        numeric_re = re.compile(r"(_count|_per_1k|_prop|_mean|_score|coverage|token_count)$")
        for chunk in pd.read_csv(seg_path, sep="\t", low_memory=False, chunksize=100_000):
            if "doc_id" not in chunk.columns:
                continue
            numeric_cols = [c for c in chunk.columns if numeric_re.search(c) and c not in {"dominant_emotion_score"}]
            for doc_id, g in chunk.groupby("doc_id", dropna=False):
                key = str(doc_id)
                if key not in agg:
                    agg[key] = {"doc_id": key}
                    meta_cols = [c for c in ["title", "author", "century", "year", "genre", "publication"] if c in chunk.columns]
                    meta_first[key] = {c: g.iloc[0].get(c, "") for c in meta_cols}
                for c in numeric_cols:
                    vals = pd.to_numeric(g[c], errors="coerce")
                    old = float(agg[key].get(c, 0.0) or 0.0)  # type: ignore[arg-type]
                    agg[key][c] = old + float(vals.sum(skipna=True))
        rows = []
        for doc_id, row in agg.items():
            row.update(meta_first.get(doc_id, {}))
            rows.append(row)
        doc_df = pd.DataFrame(rows)
        doc_df.to_csv(doc_path, sep="\t", index=False)
        return {"built": True, "method": "fallback_sum_by_doc_id", "rows": len(doc_df), "columns": len(doc_df.columns), "path": str(doc_path)}
    except Exception as exc:
        return {"built": False, "reason": f"fallback_failed: {exc}"}


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


def plot_outputs(outdir: Path, tables: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
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
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(EMOTIONS, vals)
        ax.set_title("Corpus emotion profile: SloEmoLex counts")
        ax.set_ylabel("Lexicon count")
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
                fig, ax = plt.subplots(figsize=(11, 5.5))
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
        tmp = lda_topic_summary.copy().sort_values("mean_probability", ascending=False).head(30)
        labels = tmp.get("topic_label", pd.Series([""] * len(tmp))).astype(str).str.slice(0, 40)
        fig, ax = plt.subplots(figsize=(12, 5.8))
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
        if transformer_predictions.empty or "label" not in transformer_predictions.columns:
            return
        top = transformer_predictions
        if "rank" in top.columns:
            top = top[pd.to_numeric(top["rank"], errors="coerce") == 1]
        counts = top["label"].value_counts().head(20)
        if counts.empty:
            return
        fig, ax = plt.subplots(figsize=(11, 6))
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
        required = {"pca_x", "pca_y", "transformer_cluster"}
        if transformer_clusters.empty or not required.issubset(transformer_clusters.columns):
            return
        tmp = transformer_clusters.copy()
        if len(tmp) > 25_000:
            tmp = tmp.sample(25_000, random_state=13)
        codes = pd.Categorical(tmp["transformer_cluster"]).codes
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        sc = ax.scatter(pd.to_numeric(tmp["pca_x"], errors="coerce"), pd.to_numeric(tmp["pca_y"], errors="coerce"), c=codes, s=15)
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
        labels = tmp.get("topic_label", pd.Series([""] * len(tmp))).astype(str).str.slice(0, 40)
        fig, ax = plt.subplots(figsize=(12, 6))
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
    return written, warnings


def df_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return ""
    work = df.head(max_rows) if max_rows else df
    cols = list(work.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
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


def write_report(outdir: Path, tables: dict[str, pd.DataFrame], table_stats: dict[str, Any], plots: list[str], warnings: list[str], staging_manifest: dict[str, Any]) -> Path:
    doc_scores = tables.get("document_emotion_scores", pd.DataFrame())
    lda_topic_summary = tables.get("lda_topic_summary", pd.DataFrame())
    transformer_predictions = tables.get("transformer_topic_predictions", pd.DataFrame())
    transformer_clusters = tables.get("transformer_embedding_clusters", pd.DataFrame())
    bertopic_topic_summary = tables.get("bertopic_topic_summary", pd.DataFrame())

    lines: list[str] = []
    lines.append("# Wikivir emotion and topic analysis report")
    lines.append("")
    lines.append(f"Finished/rescued from existing outputs: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Outputs used")
    lines.append("")
    for name in TABLE_NAMES:
        stat = table_stats.get(name, {"rows": 0, "columns": 0, "bytes": 0, "path": str(outdir / "tables" / f"{name}.tsv")})
        if stat.get("rows", 0) or stat.get("bytes", 0):
            lines.append(f"- `{name}.tsv`: {int(stat.get('rows', 0)):,} rows, {int(stat.get('columns', 0)):,} columns, {human_size(stat.get('bytes', 0))}")
        else:
            lines.append(f"- `{name}.tsv`: absent or empty")
    lines.append("")

    if not doc_scores.empty:
        lines.append("## Corpus-level emotion profile")
        lines.append("")
        totals = []
        for emo in EMOTIONS:
            totals.append((emo, float(pd.to_numeric(doc_scores.get(f"emotion_{emo}_count", pd.Series(dtype=float)), errors="coerce").sum())))
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
        sort_col = "mean_probability" if "mean_probability" in lda_topic_summary.columns else None
        tmp = lda_topic_summary.sort_values(sort_col, ascending=False) if sort_col else lda_topic_summary
        cols = [c for c in ["topic_id", "topic_label", "mean_probability", "segment_count", "doc_count"] if c in tmp.columns]
        lines.append(df_to_markdown(tmp[cols], max_rows=20) if cols else f"{len(tmp):,} topic summary rows.")
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

    lines.append("## BERTopic topics")
    lines.append("")
    if not bertopic_topic_summary.empty:
        cols = [c for c in ["topic_id", "topic_label", "segment_count", "doc_count", "mean_probability"] if c in bertopic_topic_summary.columns]
        lines.append(df_to_markdown(bertopic_topic_summary[cols], max_rows=20) if cols else f"{len(bertopic_topic_summary):,} BERTopic summary rows.")
    else:
        lines.append("BERTopic outputs are absent. Install `bertopic` and rerun the main analysis with `--resume --run-bertopic` to add this layer without redoing completed stages.")
    lines.append("")

    if plots:
        lines.append("## Plots")
        lines.append("")
        for p in plots:
            pp = Path(p)
            try:
                rel = pp.relative_to(outdir)
            except Exception:
                rel = pp
            lines.append(f"- `{rel}`")
        lines.append("")

    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    if staging_manifest.get("merged_shards"):
        lines.append("## Rescue actions")
        lines.append("")
        for name, info in staging_manifest["merged_shards"].items():
            lines.append(f"- merged `{name}` from {info.get('parts', 0)} checkpoint shards into `{Path(info.get('output', '')).name}`")
        lines.append("")

    report = outdir / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def load_report_tables(outdir: Path) -> dict[str, pd.DataFrame]:
    tables = outdir / "tables"
    # Keep memory down: do not read huge segment_emotion_scores unless needed.
    load_names = [
        "document_emotion_scores",
        "lda_topic_summary",
        "transformer_topic_predictions",
        "transformer_embedding_clusters",
        "bertopic_topic_summary",
    ]
    result = {}
    for name in load_names:
        path = tables / f"{name}.tsv"
        # For transformer predictions, 25k*top_k is OK; for clusters 25k OK. Still cap absurd files.
        result[name] = read_table(path)
    # Provide an empty placeholder for segment stats/report listing.
    result["segment_emotion_scores"] = pd.DataFrame()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Rescue/finish Wikivir outputs from existing tables and checkpoint shards.")
    ap.add_argument("--output-dir", type=Path, required=True, help="Analysis output dir, e.g. analysis/wikivir_full_stack")
    ap.add_argument("--search-root", type=Path, action="append", help="Additional root to search for misplaced outputs. Can be repeated.")
    ap.add_argument("--force", action="store_true", help="Overwrite/merge staged outputs even if final TSVs already exist.")
    ap.add_argument("--include-token-matches", action="store_true", help="Also merge token_emotion_matches checkpoint shards. This can be huge and is not needed for the report.")
    args = ap.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    roots = default_search_roots(outdir)
    for r in args.search_root or []:
        if r.exists() and r not in roots:
            roots.insert(0, r)

    eprint(f"[rescue] output dir: {outdir}")
    eprint("[rescue] search roots:")
    for r in roots:
        eprint(f"  - {r}")

    staging_manifest = stage_existing_and_shards(outdir, force=args.force, search_roots=roots, include_token_matches=args.include_token_matches)
    doc_build = build_document_scores_from_segments(outdir, force=args.force)
    staging_manifest["document_scores_build"] = doc_build
    if doc_build.get("built"):
        eprint(f"[rescue] built document_emotion_scores.tsv: {doc_build.get('rows')} rows")

    table_stats: dict[str, Any] = {}
    for name in TABLE_NAMES:
        path = outdir / "tables" / f"{name}.tsv"
        rows, cols = count_tsv_rows(path)
        table_stats[name] = {"path": str(path), "rows": rows, "columns": cols, "bytes": path.stat().st_size if path.exists() else 0}
        eprint(f"[rescue] {name}.tsv: {rows:,} rows, {cols:,} cols, {human_size(table_stats[name]['bytes'])}")

    tables = load_report_tables(outdir)
    plots, plot_warnings = plot_outputs(outdir, tables)
    report = write_report(outdir, tables, table_stats, plots, plot_warnings, staging_manifest)

    manifest = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(outdir),
        "report": str(report),
        "plots": plots,
        "plot_warnings": plot_warnings,
        "table_stats": table_stats,
        "rescue": staging_manifest,
    }
    manifest_path = outdir / "rescue_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if plot_warnings:
        (outdir / "finish_warnings.txt").write_text("\n".join(plot_warnings), encoding="utf-8")

    eprint(f"[rescue] report: {report}")
    eprint(f"[rescue] plots written: {len(plots)}")
    eprint(f"[rescue] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
