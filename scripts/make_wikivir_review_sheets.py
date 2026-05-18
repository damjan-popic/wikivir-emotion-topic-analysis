#!/usr/bin/env python3
"""Generate publication-grade validation/review sheets from Wikivir analysis outputs.

No plots are produced. This script is about QA and human review:
- topic examples ranked by actual centroid similarity when BERTopic/KMeans embeddings are available;
- random topic examples for bias checking;
- emotion extremes with minimum-length and coverage filters;
- metadata/lexicon/OOV review copies with useful filters.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

BUILTIN_GENERIC_OOV = {
    "biti", "imeti", "tako", "priti", "vedeti", "videti", "moči", "hoteti", "iti", "dati", "reči", "delati",
    "kako", "ali", "kaj", "kdo", "ta", "tisti", "nek", "nekaj", "vse", "ves", "vsak", "on", "ona", "ono",
    "jaz", "ti", "mi", "vi", "se", "pa", "tudi", "že", "še", "samo", "le", "zelo", "dobro", "slabo",
}


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", keep_default_na=False, low_memory=False)


def readable_preview(text: Any, limit: int = 500) -> str:
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"(?<=[.!?;:])(?=[A-ZČŠŽ])", " ", s)
    s = re.sub(r"(?<=[a-zčšž])(?=[A-ZČŠŽ])", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def find_embedding_stage(analysis_dir: Path, preferred: str = "bertopic") -> Path | None:
    candidates = [analysis_dir / "checkpoints" / preferred, analysis_dir / "checkpoints" / "embeddings"]
    for c in candidates:
        if (c / "segments.tsv").exists() and sorted(c.glob("embeddings.part-*.npy")):
            return c
    return None


def iter_embedding_shards(stage: Path):
    for p in sorted(stage.glob("embeddings.part-*.npy")):
        yield p, np.load(p, mmap_mode="r")


def load_embeddings_for_segments(stage: Path) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = read_tsv(stage / "segments.tsv")
    arrays = []
    for _, arr in iter_embedding_shards(stage):
        arrays.append(np.asarray(arr, dtype="float32"))
    if not arrays:
        return manifest, np.empty((0, 0), dtype="float32")
    emb = np.vstack(arrays).astype("float32", copy=False)
    if len(manifest) != emb.shape[0]:
        raise ValueError(f"Embedding manifest row count {len(manifest)} != embedding rows {emb.shape[0]} in {stage}")
    return manifest, emb


def add_centroid_similarity(topic_rows: pd.DataFrame, analysis_dir: Path) -> pd.DataFrame:
    stage = find_embedding_stage(analysis_dir, "bertopic")
    out = topic_rows.copy()
    out["centroid_cosine"] = np.nan
    out["distance_rank_source"] = "none"
    if stage is None:
        return out
    try:
        manifest, emb = load_embeddings_for_segments(stage)
    except Exception as exc:
        print(f"[warning] Could not load embeddings for centroid ranking: {exc}", file=sys.stderr)
        return out
    if manifest.empty or emb.size == 0 or "segment_id" not in manifest.columns:
        return out
    assignments = out[["segment_id", "bertopic_topic"]].merge(
        manifest[["segment_id"]].reset_index().rename(columns={"index": "embedding_row"}),
        on="segment_id", how="left"
    )
    valid = assignments["embedding_row"].notna().to_numpy()
    if not valid.any():
        return out
    # L2-normalize once for cosine centroids.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = emb / norms
    similarities = np.full(len(out), np.nan, dtype="float32")
    for topic, g in assignments[valid].groupby("bertopic_topic"):
        idx = g.index.to_numpy()
        rows = g["embedding_row"].astype(int).to_numpy()
        if len(rows) == 0:
            continue
        centroid = emb_norm[rows].mean(axis=0)
        cn = np.linalg.norm(centroid)
        if cn == 0:
            continue
        centroid = centroid / cn
        similarities[idx] = emb_norm[rows] @ centroid
    out["centroid_cosine"] = similarities
    out["distance_rank_source"] = np.where(out["centroid_cosine"].notna(), "bertopic_embedding_centroid_cosine", "none")
    return out


def make_topic_review(args: argparse.Namespace, tables: Path, review: Path) -> None:
    seg_topics = read_tsv(tables / "bertopic_segment_topics.tsv")
    segments = read_tsv(tables / "segments.tsv")
    topic_summary = read_tsv(tables / "bertopic_topic_summary.tsv")
    if seg_topics.empty:
        print("[warning] No bertopic_segment_topics.tsv; topic review skipped.", file=sys.stderr)
        return
    if not segments.empty and "segment_id" in segments.columns:
        keep_cols = [c for c in ["segment_id", "unit", "doc_id", "word_token_count", "lemma_token_count", "text_preview", "title", "author", "genre", "century", "year"] if c in segments.columns]
        merged = seg_topics.drop(columns=[c for c in ["text_preview"] if c in seg_topics.columns], errors="ignore").merge(
            segments[keep_cols], on="segment_id", how="left", suffixes=("", "_segment")
        )
    else:
        merged = seg_topics.copy()
    if "text_preview" in merged.columns:
        merged["text_preview"] = merged["text_preview"].map(lambda x: readable_preview(x, args.preview_chars))
    if "word_token_count" in merged.columns:
        merged = merged[pd.to_numeric(merged["word_token_count"], errors="coerce").fillna(0) >= args.topic_min_words]
    merged = add_centroid_similarity(merged, args.analysis_dir)
    if "centroid_cosine" in merged.columns and merged["centroid_cosine"].notna().any():
        score_col = "centroid_cosine"
    elif "bertopic_probability" in merged.columns and pd.to_numeric(merged["bertopic_probability"], errors="coerce").notna().any():
        score_col = "bertopic_probability"
        merged["distance_rank_source"] = "bertopic_probability"
    else:
        score_col = None
        merged["distance_rank_source"] = "fallback_original_order"
    label_map = {}
    if not topic_summary.empty and {"topic_id", "topic_label"}.issubset(topic_summary.columns):
        label_map = topic_summary.set_index("topic_id")["topic_label"].to_dict()
    rows = []
    rng = np.random.default_rng(args.random_state)
    for topic, g in merged.groupby("bertopic_topic", sort=True):
        if score_col:
            top = g.assign(_score=pd.to_numeric(g[score_col], errors="coerce")).sort_values("_score", ascending=False).head(args.topic_top_n)
        else:
            top = g.head(args.topic_top_n).assign(_score=np.nan)
        n_random = min(args.topic_random_n, len(g))
        random_rows = g.sample(n=n_random, random_state=args.random_state) if n_random else g.head(0)
        for label, frame in [("top_centroid", top), ("random", random_rows)]:
            if frame.empty:
                continue
            tmp = frame.copy()
            tmp["review_source"] = label
            tmp["topic_label_proposed"] = label_map.get(topic, "")
            tmp["human_topic_label"] = ""
            tmp["human_topic_notes"] = ""
            tmp["keep_for_interpretation"] = ""
            rows.append(tmp)
    if rows:
        out = pd.concat(rows, ignore_index=True)
        # useful ordering
        preferred = ["bertopic_topic", "topic_label_proposed", "review_source", "centroid_cosine", "bertopic_probability", "distance_rank_source", "segment_id", "doc_id", "title", "author", "genre", "century", "year", "word_token_count", "text_preview", "human_topic_label", "keep_for_interpretation", "human_topic_notes"]
        cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
        out[cols].to_csv(review / "topic_label_review.tsv", sep="\t", index=False)
        print(f"Wrote {review / 'topic_label_review.tsv'} ({len(out):,} rows)")


def make_emotion_extremes(args: argparse.Namespace, tables: Path, review: Path) -> None:
    emo = read_tsv(tables / "segment_emotion_scores.tsv")
    segments = read_tsv(tables / "segments.tsv")
    if emo.empty or segments.empty:
        print("[warning] Emotion scores or segments missing; emotion extremes skipped.", file=sys.stderr)
        return
    keep_cols = [c for c in ["segment_id", "unit", "doc_id", "word_token_count", "lemma_token_count", "text_preview", "title", "author", "genre", "century", "year"] if c in segments.columns]
    merged = emo.merge(segments[keep_cols], on="segment_id", how="left", suffixes=("", "_segment"))
    if "unit" in merged.columns:
        merged = merged[merged["unit"].astype(str) == args.emotion_unit]
    if "word_token_count" in merged.columns:
        merged = merged[pd.to_numeric(merged["word_token_count"], errors="coerce").fillna(0) >= args.emotion_min_words]
    if "lexicon_coverage" in merged.columns:
        merged = merged[pd.to_numeric(merged["lexicon_coverage"], errors="coerce").fillna(0) >= args.emotion_min_coverage]
    if "text_preview" in merged.columns:
        merged["text_preview"] = merged["text_preview"].map(lambda x: readable_preview(x, args.preview_chars))
    rate_cols = [c for c in merged.columns if c.startswith("emotion_") and c.endswith("_per_1k")]
    rows = []
    for col in rate_cols:
        count_col = col.replace("_per_1k", "_count")
        g = merged.copy()
        if count_col in g.columns:
            g = g[pd.to_numeric(g[count_col], errors="coerce").fillna(0) >= args.emotion_min_hits]
        top = g.assign(_score=pd.to_numeric(g[col], errors="coerce")).sort_values("_score", ascending=False).head(args.emotion_top_n)
        if not top.empty:
            top["review_emotion"] = col.replace("emotion_", "").replace("_per_1k", "")
            top["human_valid_emotion"] = ""
            top["human_notes"] = ""
            rows.append(top)
    if rows:
        out = pd.concat(rows, ignore_index=True)
        preferred = ["review_emotion", "_score", "segment_id", "doc_id", "unit", "title", "author", "genre", "century", "year", "word_token_count", "lexicon_coverage", "text_preview", "human_valid_emotion", "human_notes"]
        cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
        out[cols].to_csv(review / "emotion_extremes_review.tsv", sep="\t", index=False)
        print(f"Wrote {review / 'emotion_extremes_review.tsv'} ({len(out):,} rows)")


def copy_coverage_and_oov(args: argparse.Namespace, tables: Path, review: Path) -> None:
    for name in ["metadata_coverage.tsv", "lexicon_coverage.tsv"]:
        p = tables / name
        if p.exists():
            df = read_tsv(p)
            df.to_csv(review / name.replace(".tsv", "_review.tsv"), sep="\t", index=False)
            print(f"Wrote {review / name.replace('.tsv', '_review.tsv')}")
    oov = read_tsv(tables / "top_oov_content_lemmas.tsv")
    if not oov.empty:
        if "lemma" in oov.columns:
            oov = oov[~oov["lemma"].astype(str).str.lower().isin(BUILTIN_GENERIC_OOV)]
            oov = oov[~oov["lemma"].astype(str).str.fullmatch(r"[\d\W_]+")]
            oov = oov[oov["lemma"].astype(str).str.len() >= 3]
        if "dominant_upos" in oov.columns:
            oov = oov[oov["dominant_upos"].isin(["NOUN", "PROPN", "ADJ", "VERB", "ADV"])]
        oov.head(args.oov_top_n).to_csv(review / "top_oov_content_lemmas_review.tsv", sep="\t", index=False)
        print(f"Wrote {review / 'top_oov_content_lemmas_review.tsv'} ({min(len(oov), args.oov_top_n):,} rows)")


def write_manifest(args: argparse.Namespace, review: Path) -> None:
    files = {}
    for p in sorted(review.glob("*.tsv")):
        try:
            rows = sum(1 for _ in p.open("r", encoding="utf-8", errors="replace")) - 1
        except Exception:
            rows = None
        files[p.name] = {"bytes": p.stat().st_size, "rows": rows}
    (review / "review_manifest.json").write_text(json.dumps({
        "analysis_dir": str(args.analysis_dir),
        "review_dir": str(review),
        "parameters": vars(args),
        "files": files,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--analysis-dir", type=Path, required=True)
    ap.add_argument("--review-dir", type=Path)
    ap.add_argument("--topic-top-n", type=int, default=25)
    ap.add_argument("--topic-random-n", type=int, default=10)
    ap.add_argument("--topic-min-words", type=int, default=40)
    ap.add_argument("--emotion-unit", default="window", choices=["document", "paragraph", "sentence", "window"])
    ap.add_argument("--emotion-min-words", type=int, default=100)
    ap.add_argument("--emotion-min-hits", type=float, default=2.0)
    ap.add_argument("--emotion-min-coverage", type=float, default=0.05)
    ap.add_argument("--emotion-top-n", type=int, default=50)
    ap.add_argument("--oov-top-n", type=int, default=500)
    ap.add_argument("--preview-chars", type=int, default=500)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args(argv)

    tables = args.analysis_dir / "tables"
    if not tables.exists():
        raise SystemExit(f"Missing tables directory: {tables}")
    review = args.review_dir or (args.analysis_dir / "review")
    review.mkdir(parents=True, exist_ok=True)
    make_topic_review(args, tables, review)
    make_emotion_extremes(args, tables, review)
    copy_coverage_and_oov(args, tables, review)
    write_manifest(args, review)
    print(f"Review sheets ready: {review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
