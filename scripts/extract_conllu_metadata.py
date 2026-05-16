#!/usr/bin/env python3
"""Extract embedded document metadata from a CoNLL-U corpus to TSV.

This is mostly a convenience/debugging tool. The main analysis script reads the
same embedded metadata directly, so a sidecar TSV is not required.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# The sibling module contains the canonical CoNLL-U metadata parser used by the
# analysis pipeline.
from wikivir_emotion_topic_analysis import RunArtifacts, metadata_presence_summary, parse_conllu


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract # newdoc/# title/# author/etc. metadata from CoNLL-U comments.")
    p.add_argument("--input", required=True, type=Path, help="Input CoNLL-U file.")
    p.add_argument("--output", required=True, type=Path, help="Output TSV path.")
    p.add_argument("--coverage-output", type=Path, help="Optional metadata coverage TSV path.")
    p.add_argument("--fail-if-empty", action="store_true", help="Exit with code 2 if no embedded metadata fields are found.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = RunArtifacts()
    docs = parse_conllu(args.input, artifacts)
    rows = [{"doc_id": doc.doc_id, **doc.metadata} for doc in docs]
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep="\t", index=False)

    coverage = metadata_presence_summary(docs)
    if args.coverage_output:
        args.coverage_output.parent.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(args.coverage_output, sep="\t", index=False)

    if artifacts.warnings:
        print("Warnings:", file=sys.stderr)
        for warning in artifacts.warnings:
            print(f"- {warning}", file=sys.stderr)

    metadata_cols = [c for c in df.columns if c != "doc_id"]
    if args.fail_if_empty and not metadata_cols:
        print("No embedded document metadata found.", file=sys.stderr)
        return 2
    print(f"Wrote {len(df):,} document rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
