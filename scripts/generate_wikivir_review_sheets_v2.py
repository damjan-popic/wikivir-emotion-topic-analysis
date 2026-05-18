#!/usr/bin/env python3
"""Generate validation/review sheets after analysis, without plots.

This version fixes two known problems from the pilot validation:
1. For KMeans-BERTopic, example ranking uses cosine similarity to the topic
   centroid when embedding checkpoints are available; it no longer emits fake
   zero scores.
2. Text previews are readability-cleaned and emotion extremes are length-filtered.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

GENERAL_OOV_STOP = {
    "biti", "tako", "imeti", "priti", "vedeti", "videti", "moči", "kako", "ali", "hoteti", "iti", "dati",
    "reči", "storiti", "postati", "ostati", "misliti", "govoriti", "gledati", "najti", "dobiti", "slišati",
    "eden", "drug", "sam", "ves", "nov", "velik", "majhen", "dober", "lep", "slab", "prav", "res",
}
CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "INTJ"}

def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", low_memory=False, keep_default_na=False)

def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)

def readable_preview(x: object, max_chars: int = 500) -> str:
    text = str(x or "")
    text = text.replace("\t", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # Repair obvious lower->upper glue caused by missing sentence/line boundary spaces.
    text = re.sub(r"([a-zčšžćđ])([A-ZČŠŽĆĐ])", r"\1 \2", text)
    # Repair punctuation followed by uppercase without a space.
    text = re.sub(r"([.!?;:])([A-ZČŠŽĆĐ])", r"\1 \2", text)
    return text[:max_chars]

def word_count(text: object) -> int:
    return len(re.findall(r"\w+", str(text or ""), flags=re.UNICODE))

def iter_embedding_shards(stage_dir: Path) -> list[Path]:
    return sorted(stage_dir.glob("embeddings.part-*.npy"))

def load_topic_centroid_scores(analysis_dir: Path, topic_col: str = "bertopic_topic") -> pd.DataFrame:
    """Return segment_id/topic/centroid_cosine from BERTopic embedding shards.

    The checkpoint manifest ``checkpoints/bertopic/segments.tsv`` has the same
    row order as the .npy shards. We stream twice: centroid sums/counts, then
    segment cosine scores. This is safe for hundreds of thousands of windows.
    """
    ckpt = analysis_dir / "checkpoints" / "bertopic"
    manifest_path = ckpt / "segments.tsv"
    seg_topics_path = analysis_dir / "tables" / "bertopic_segment_topics.tsv"
    emb_paths = iter_embedding_shards(ckpt)
    if not manifest_path.exists() or not emb_paths or not seg_topics_path.exists():
        return pd.DataFrame()
    manifest = read_tsv(manifest_path)
    seg_topics = read_tsv(seg_topics_path)
    if manifest.empty or seg_topics.empty or "segment_id" not in manifest or "segment_id" not in seg_topics or topic_col not in seg_topics:
        return pd.DataFrame()
    topic_by_segment = dict(zip(seg_topics["segment_id"].astype(str), seg_topics[topic_col]))
    manifest_ids = manifest["segment_id"].astype(str).tolist()

    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    offset = 0
    for path in emb_paths:
        arr = np.load(path, mmap_mode="r")
        ids = manifest_ids[offset: offset + len(arr)]
        for i, sid in enumerate(ids):
            try:
                topic = int(topic_by_segment[sid])
            except Exception:
                continue
            vec = np.asarray(arr[i], dtype="float64")
            if topic not in sums:
                sums[topic] = np.zeros(vec.shape, dtype="float64")
                counts[topic] = 0
            sums[topic] += vec
            counts[topic] += 1
        offset += len(arr)
    centroids: dict[int, np.ndarray] = {}
    for topic, vec in sums.items():
        c = vec / max(1, counts[topic])
        norm = np.linalg.norm(c)
        centroids[topic] = c / norm if norm else c

    rows = []
    offset = 0
    for path in emb_paths:
        arr = np.load(path, mmap_mode="r")
        ids = manifest_ids[offset: offset + len(arr)]
        for i, sid in enumerate(ids):
            try:
                topic = int(topic_by_segment[sid])
                c = centroids[topic]
            except Exception:
                continue
            v = np.asarray(arr[i], dtype="float64")
            denom = np.linalg.norm(v)
            score = float(np.dot(v / denom, c)) if denom else 0.0
            rows.append({"segment_id": sid, topic_col: topic, "centroid_cosine": score, "review_score_source": "centroid_cosine"})
        offset += len(arr)
    return pd.DataFrame(rows)

def make_topic_review(args: argparse.Namespace, manifest: dict) -> None:
    tables = args.analysis_dir / "tables"
    review = args.output_dir
    seg_topics = read_tsv(tables / "bertopic_segment_topics.tsv")
    segments = read_tsv(tables / "segments.tsv")
    if seg_topics.empty or segments.empty:
        manifest["topic_review"] = "skipped: missing bertopic_segment_topics.tsv or segments.tsv"
        return
    merged = seg_topics.merge(segments, on="segment_id", how="left", suffixes=("", "_segment"))
    score_df = load_topic_centroid_scores(args.analysis_dir)
    if not score_df.empty:
        merged = merged.merge(score_df[["segment_id", "centroid_cosine", "review_score_source"]], on="segment_id", how="left")
        merged["_review_score"] = pd.to_numeric(merged["centroid_cosine"], errors="coerce")
        merged["review_score_source"] = merged["review_score_source"].fillna("centroid_missing")
    elif "bertopic_probability" in merged.columns and pd.to_numeric(merged["bertopic_probability"], errors="coerce").fillna(0).max() > 0:
        merged["_review_score"] = pd.to_numeric(merged["bertopic_probability"], errors="coerce")
        merged["review_score_source"] = "bertopic_probability"
    else:
        merged["_review_score"] = merged.get("lexical_tokens", merged.get("token_count", merged.get("text_preview", ""))).apply(lambda x: float(x) if str(x).isdigit() else word_count(x))
        merged["review_score_source"] = "fallback_length"
    if "text_preview" in merged.columns:
        merged["text_preview_readable"] = merged["text_preview"].map(lambda x: readable_preview(x, args.preview_chars))
    elif "text_preview_segment" in merged.columns:
        merged["text_preview_readable"] = merged["text_preview_segment"].map(lambda x: readable_preview(x, args.preview_chars))
    else:
        merged["text_preview_readable"] = ""

    rows = []
    for topic, g in merged.groupby("bertopic_topic"):
        g = g.copy()
        top = g.sort_values("_review_score", ascending=False).head(args.topic_top_n).assign(review_source="top_centroid")
        # Random examples should not duplicate top examples.
        rest = g[~g["segment_id"].isin(set(top["segment_id"]))]
        rnd = rest.sample(min(args.topic_random_n, len(rest)), random_state=args.random_state).assign(review_source="random") if len(rest) else pd.DataFrame()
        rows.extend([top, rnd])
    out = pd.concat([r for r in rows if r is not None and not r.empty], ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        first_cols = [c for c in ["bertopic_topic", "review_source", "_review_score", "review_score_source", "segment_id", "doc_id", "unit", "title", "author", "genre", "century", "year", "text_preview_readable"] if c in out.columns]
        other = [c for c in out.columns if c not in first_cols]
        out = out[first_cols + other]
    write_tsv(out, review / "topic_label_review.tsv")
    write_tsv(out[["segment_id", "bertopic_topic", "_review_score", "review_score_source"]].rename(columns={"_review_score": "review_score"}) if not out.empty else out, review / "topic_segment_review_scores.tsv")
    manifest["topic_review"] = {"rows": int(len(out)), "score_sources": out["review_score_source"].value_counts().to_dict() if not out.empty and "review_score_source" in out else {}}

def make_emotion_review(args: argparse.Namespace, manifest: dict) -> None:
    tables = args.analysis_dir / "tables"
    review = args.output_dir
    emo = read_tsv(tables / "segment_emotion_scores.tsv")
    segments = read_tsv(tables / "segments.tsv")
    if emo.empty or segments.empty:
        manifest["emotion_extremes"] = "skipped: missing segment_emotion_scores.tsv or segments.tsv"
        return
    merged = emo.merge(segments, on="segment_id", how="left", suffixes=("", "_segment"))
    if "unit" in merged.columns:
        merged = merged[merged["unit"].isin(args.emotion_units.split(","))]
    if "lexical_tokens" in merged.columns:
        lengths = pd.to_numeric(merged["lexical_tokens"], errors="coerce").fillna(0)
    elif "token_count" in merged.columns:
        lengths = pd.to_numeric(merged["token_count"], errors="coerce").fillna(0)
    else:
        preview_col = "text_preview" if "text_preview" in merged.columns else "text_preview_segment"
        lengths = merged.get(preview_col, pd.Series([""] * len(merged))).map(word_count)
    merged = merged.assign(_review_word_count=lengths)
    filtered = merged[merged["_review_word_count"] >= args.emotion_min_words].copy()
    cols = [c for c in filtered.columns if c.startswith("emotion_") and c.endswith("_per_1k")]
    rows = []
    for col in cols:
        rows.append(filtered.sort_values(col, ascending=False).head(args.emotion_top_n).assign(review_emotion=col))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        preview_col = "text_preview" if "text_preview" in out.columns else "text_preview_segment"
        out["text_preview_readable"] = out.get(preview_col, "").map(lambda x: readable_preview(x, args.preview_chars))
        first = [c for c in ["review_emotion", col, "segment_id", "doc_id", "unit", "_review_word_count", "title", "author", "genre", "century", "year", "text_preview_readable"] if c in out.columns]
        out = out[first + [c for c in out.columns if c not in first]]
    write_tsv(out, review / "emotion_extremes_review.tsv")
    manifest["emotion_extremes"] = {"rows": int(len(out)), "min_words": args.emotion_min_words, "units": args.emotion_units}

def make_oov_review(args: argparse.Namespace, manifest: dict) -> None:
    tables = args.analysis_dir / "tables"
    review = args.output_dir
    oov = read_tsv(tables / "top_oov_content_lemmas.tsv")
    if oov.empty:
        manifest["oov_review"] = "skipped: missing top_oov_content_lemmas.tsv"
        return
    df = oov.copy()
    lemma_col = "lemma" if "lemma" in df.columns else ("lexeme" if "lexeme" in df.columns else df.columns[0])
    df["_lemma_norm"] = df[lemma_col].astype(str).str.lower().str.strip()
    if "upos" in df.columns:
        df = df[df["upos"].astype(str).str.upper().isin(CONTENT_UPOS)]
    df = df[~df["_lemma_norm"].isin(GENERAL_OOV_STOP)]
    df = df[df["_lemma_norm"].str.len() >= 3]
    write_tsv(df.head(args.oov_top_n), review / "top_oov_content_lemmas_review.tsv")
    manifest["oov_review"] = {"rows": int(min(len(df), args.oov_top_n)), "filtered_from": int(len(oov))}

def copy_coverage(args: argparse.Namespace, manifest: dict) -> None:
    tables = args.analysis_dir / "tables"
    for name in ["metadata_coverage.tsv", "lexicon_coverage.tsv"]:
        df = read_tsv(tables / name)
        if not df.empty:
            write_tsv(df, args.output_dir / name.replace(".tsv", "_review.tsv"))
            manifest[name] = {"rows": int(len(df))}

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate validation sheets from Wikivir research outputs; no plots.")
    ap.add_argument("--analysis-dir", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, help="Defaults to ANALYSIS_DIR/review")
    ap.add_argument("--topic-top-n", type=int, default=25)
    ap.add_argument("--topic-random-n", type=int, default=10)
    ap.add_argument("--emotion-top-n", type=int, default=50)
    ap.add_argument("--emotion-min-words", type=int, default=80)
    ap.add_argument("--emotion-units", default="window,document")
    ap.add_argument("--oov-top-n", type=int, default=500)
    ap.add_argument("--preview-chars", type=int, default=700)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = args.analysis_dir / "review"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"analysis_dir": str(args.analysis_dir), "output_dir": str(args.output_dir)}
    make_topic_review(args, manifest)
    make_emotion_review(args, manifest)
    make_oov_review(args, manifest)
    copy_coverage(args, manifest)
    (args.output_dir / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
