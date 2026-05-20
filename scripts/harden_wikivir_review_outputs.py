#!/usr/bin/env python3
"""
Harden/verify Wikivir review outputs before interpretation or plotting.

This script does two things:
1) verifies that regenerated review sheets are wired to the clean corpus/run;
2) optionally regenerates a stricter emotion-extreme review sheet from
   tables/segment_emotion_scores.tsv using lexical-token thresholds.

It is intentionally not a plotting or interpretation script.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

BAD_KEY_RE = re.compile(r"gerne|cemtury", re.I)
GLUED_RE = re.compile(r"[a-záéíóúàèìòùäëïöüčšžćđ][A-ZČŠŽĆĐ]")
EMOTION_RATE_RE = re.compile(r"^emotion_(anger|anticipation|disgust|fear|joy|sadness|surprise|trust)_per_1k$")


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def verify_review(review_dir: Path, expected_source_xml: str) -> Dict:
    out: Dict = {"ok": True, "warnings": [], "errors": [], "files": {}}

    paths = {
        "manifest": review_dir / "review_manifest.json",
        "metadata": review_dir / "metadata_coverage_review.tsv",
        "topic": review_dir / "topic_label_review.tsv",
        "scores": review_dir / "topic_segment_review_scores.tsv",
        "emotion": review_dir / "emotion_extremes_review.tsv",
        "oov": review_dir / "top_oov_content_lemmas_review.tsv",
        "lexicon": review_dir / "lexicon_coverage_review.tsv",
    }

    for name, path in paths.items():
        out["files"][name] = {"path": str(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0}

    if paths["manifest"].exists():
        try:
            out["manifest"] = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        except Exception as e:
            out["errors"].append(f"Cannot parse review_manifest.json: {e}")
            out["ok"] = False

    meta = read_tsv(paths["metadata"])
    topic = read_tsv(paths["topic"])
    scores = read_tsv(paths["scores"])
    emotion = read_tsv(paths["emotion"])

    # Bad-key checks across columns and values.
    for name, df in [("metadata", meta), ("topic", topic), ("emotion", emotion)]:
        if df.empty:
            out["warnings"].append(f"{name} review table is empty or absent")
            continue
        bad_cols = [c for c in df.columns if BAD_KEY_RE.search(c)]
        if bad_cols:
            out["errors"].append(f"{name} contains bad metadata columns: {bad_cols}")
            out["ok"] = False
        if name == "metadata" and "metadata_key" in df.columns:
            bad_rows = df[df["metadata_key"].str.contains(BAD_KEY_RE, na=False)]
            if not bad_rows.empty:
                out["errors"].append(f"metadata coverage contains bad metadata keys: {bad_rows['metadata_key'].tolist()}")
                out["ok"] = False

    # Source XML checks.
    for name, df in [("topic", topic), ("emotion", emotion)]:
        if df.empty:
            continue
        for col in [c for c in df.columns if "source_xml" in c]:
            vals = sorted(set(df[col].dropna().astype(str)) - {""})
            out.setdefault("source_xml_values", {})[f"{name}.{col}"] = vals[:20]
            if expected_source_xml and any(v != expected_source_xml for v in vals):
                out["errors"].append(f"{name}.{col} contains non-clean source_xml values: {vals[:20]}")
                out["ok"] = False

    # Topic score checks.
    if not topic.empty:
        out["topic_review"] = {
            "rows": int(len(topic)),
            "topics": int(topic["bertopic_topic"].nunique()) if "bertopic_topic" in topic.columns else None,
        }
        if "review_score_source" in topic.columns:
            out["topic_review"]["score_sources"] = topic["review_score_source"].value_counts().to_dict()
            if set(topic["review_score_source"].unique()) != {"centroid_cosine"}:
                out["warnings"].append("Topic review score_source is not purely centroid_cosine")
        else:
            out["errors"].append("topic_label_review.tsv has no review_score_source column")
            out["ok"] = False

        score_col = "_review_score" if "_review_score" in topic.columns else "review_score" if "review_score" in topic.columns else None
        if score_col:
            s = numeric(topic[score_col])
            out["topic_review"]["score_summary"] = {
                "min": float(s.min()), "median": float(s.median()), "max": float(s.max()),
                "zeros": int((s.fillna(0) == 0).sum()), "missing": int(s.isna().sum()),
            }
            if (s.fillna(0) == 0).all():
                out["errors"].append("All topic review scores are zero")
                out["ok"] = False
        else:
            out["errors"].append("topic_label_review.tsv has no review score column")
            out["ok"] = False

        if "text_preview_readable" in topic.columns:
            glued = int(topic["text_preview_readable"].fillna("").astype(str).str.contains(GLUED_RE).sum())
            out["topic_review"]["glued_preview_hits"] = glued
            if glued:
                out["warnings"].append(f"Topic previews have {glued} lowercase-uppercase glue hits")

        if "bertopic_topic" in topic.columns:
            counts = topic.groupby("bertopic_topic").size()
            out["topic_review"]["topic_row_count_min"] = int(counts.min())
            out["topic_review"]["topic_row_count_max"] = int(counts.max())
            out["topic_review"]["topics_under_35_rows"] = int((counts < 35).sum())

    # Topic score table alignment.
    if not topic.empty and not scores.empty and {"segment_id", "bertopic_topic"}.issubset(topic.columns) and {"segment_id", "bertopic_topic"}.issubset(scores.columns):
        merged = topic[["segment_id", "bertopic_topic"]].merge(scores[["segment_id", "bertopic_topic"]], how="outer", indicator=True)
        counts = merged["_merge"].value_counts().to_dict()
        out["topic_score_alignment"] = counts
        if counts.get("left_only", 0) or counts.get("right_only", 0):
            out["errors"].append(f"topic_label_review and topic_segment_review_scores are misaligned: {counts}")
            out["ok"] = False

    # Emotion checks.
    if not emotion.empty:
        out["emotion_review"] = {"rows": int(len(emotion))}
        if "review_emotion" in emotion.columns:
            out["emotion_review"]["review_emotion_counts"] = emotion["review_emotion"].value_counts().to_dict()
        if "segment_id" in emotion.columns:
            out["emotion_review"]["unique_segment_ids"] = int(emotion["segment_id"].nunique())
            out["emotion_review"]["duplicate_rows_by_segment_id"] = int(len(emotion) - emotion["segment_id"].nunique())
        for col in ["_review_word_count", "word_token_count", "lexical_token_count", "lemma_token_count"]:
            if col in emotion.columns:
                s = numeric(emotion[col])
                out["emotion_review"][f"{col}_summary"] = {
                    "min": float(s.min()), "median": float(s.median()), "max": float(s.max()),
                    "under_80": int((s < 80).sum()), "under_50": int((s < 50).sum()),
                }
        if "text_preview_readable" in emotion.columns:
            glued = int(emotion["text_preview_readable"].fillna("").astype(str).str.contains(GLUED_RE).sum())
            out["emotion_review"]["glued_preview_hits"] = glued
            if glued:
                out["warnings"].append(f"Emotion previews have {glued} lowercase-uppercase glue hits")

    if out["errors"]:
        out["ok"] = False
    return out


def select_top_extremes_from_chunks(
    table_path: Path,
    units: List[str],
    top_n: int,
    min_lexical_tokens: int,
    min_word_tokens: int,
    chunksize: int,
) -> pd.DataFrame:
    keep: Dict[str, pd.DataFrame] = {}
    cols_known = None

    for chunk in pd.read_csv(table_path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize):
        if cols_known is None:
            cols_known = list(chunk.columns)
        if "unit" in chunk.columns and units:
            chunk = chunk[chunk["unit"].isin(units)].copy()
        if chunk.empty:
            continue
        if "lexical_token_count" in chunk.columns:
            chunk = chunk[numeric(chunk["lexical_token_count"]).fillna(0) >= min_lexical_tokens].copy()
        if "word_token_count" in chunk.columns:
            chunk = chunk[numeric(chunk["word_token_count"]).fillna(0) >= min_word_tokens].copy()
        if chunk.empty:
            continue
        emotion_cols = [c for c in chunk.columns if EMOTION_RATE_RE.match(c)]
        for col in emotion_cols:
            tmp = chunk.copy()
            tmp["__score"] = numeric(tmp[col]).fillna(-1)
            tmp = tmp[tmp["__score"] >= 0]
            if tmp.empty:
                continue
            tmp = tmp.nlargest(top_n, "__score")
            tmp.insert(0, "review_emotion", col)
            tmp.insert(1, "_review_score", tmp.pop("__score"))
            tmp.insert(2, "_review_score_source", f"{col}; lexical_token_count>={min_lexical_tokens}; word_token_count>={min_word_tokens}")
            if col in keep:
                keep[col] = pd.concat([keep[col], tmp], ignore_index=True).nlargest(top_n, "_review_score")
            else:
                keep[col] = tmp

    if not keep:
        return pd.DataFrame()
    out = pd.concat([keep[k].nlargest(top_n, "_review_score") for k in sorted(keep)], ignore_index=True)
    # Add readable preview if possible.
    if "text_preview_readable" not in out.columns and "text_preview" in out.columns:
        out.insert(min(10, len(out.columns)), "text_preview_readable", out["text_preview"].astype(str))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default="analysis/wikivir_research_v2_clean")
    ap.add_argument("--review-dir", default=None)
    ap.add_argument("--expected-source-xml", default="wikivir.cleaned.xml")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--regenerate-strict-emotion", action="store_true")
    ap.add_argument("--emotion-units", default="window,document")
    ap.add_argument("--emotion-top-n", type=int, default=50)
    ap.add_argument("--min-lexical-tokens", type=int, default=80)
    ap.add_argument("--min-word-tokens", type=int, default=80)
    ap.add_argument("--chunksize", type=int, default=200000)
    args = ap.parse_args()

    analysis_dir = Path(args.analysis_dir)
    review_dir = Path(args.review_dir) if args.review_dir else analysis_dir / "review"
    summary_path = Path(args.summary) if args.summary else review_dir / "review_hardening_summary.json"
    review_dir.mkdir(parents=True, exist_ok=True)

    result = verify_review(review_dir, args.expected_source_xml)

    if args.regenerate_strict_emotion:
        seg_scores = analysis_dir / "tables" / "segment_emotion_scores.tsv"
        if not seg_scores.exists():
            result.setdefault("strict_emotion", {})["error"] = f"Missing {seg_scores}"
            result["ok"] = False
        else:
            strict = select_top_extremes_from_chunks(
                seg_scores,
                units=[x.strip() for x in args.emotion_units.split(",") if x.strip()],
                top_n=args.emotion_top_n,
                min_lexical_tokens=args.min_lexical_tokens,
                min_word_tokens=args.min_word_tokens,
                chunksize=args.chunksize,
            )
            out_path = review_dir / f"emotion_extremes_review_lexical{args.min_lexical_tokens}.tsv"
            strict.to_csv(out_path, sep="\t", index=False)
            dedup_path = review_dir / f"emotion_extremes_review_lexical{args.min_lexical_tokens}_dedup.tsv"
            if not strict.empty and "segment_id" in strict.columns:
                dedup = strict.sort_values("_review_score", ascending=False).drop_duplicates("segment_id")
                dedup.to_csv(dedup_path, sep="\t", index=False)
            else:
                dedup = strict
                dedup.to_csv(dedup_path, sep="\t", index=False)
            result["strict_emotion"] = {
                "written": str(out_path),
                "rows": int(len(strict)),
                "dedup_written": str(dedup_path),
                "dedup_rows": int(len(dedup)),
                "min_lexical_tokens": args.min_lexical_tokens,
                "min_word_tokens": args.min_word_tokens,
            }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
