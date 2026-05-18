#!/usr/bin/env python3
"""Preflight validation for Wikivir CLASSLA CoNLL-U before analysis.

This catches the boring-but-deadly stuff: duplicate document IDs, typo metadata
keys, low metadata coverage, malformed token rows, missing roots, and glued
preview text caused by broken spacing across sentence boundaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ALIASES = {
    "gerne": "genre", "genere": "genre", "gendre": "genre", "gernre": "genre", "zvrst": "genre", "vrsta": "genre",
    "cemtury": "century", "centry": "century", "centruy": "century", "cenutry": "century", "stoletje": "century", "century_": "century",
    "yeaar": "year", "yer": "year", "leto": "year", "date_year": "year",
    "avtor": "author", "autor": "author", "auhtor": "author", "writer": "author",
    "naslov": "title", "titel": "title", "name": "title",
    "pubication": "publication", "publicaton": "publication", "publikacija": "publication",
    "source_url": "url", "uri": "url",
}
STRUCTURAL = {"sent_id", "text", "newdoc", "newdoc_id", "newpar", "newpar_id", "par", "par_id", "paragraph_id"}
PREFIXES = ("doc_", "document_", "newdoc_")

def norm_key(key: str) -> str:
    k = unicodedata.normalize("NFKC", str(key)).strip().lower()
    k = re.sub(r"[^0-9a-zA-ZčšžćđČŠŽĆĐ]+", "_", k, flags=re.UNICODE).strip("_").lower()
    for p in PREFIXES:
        if k.startswith(p) and len(k) > len(p):
            k = k[len(p):]
            break
    return ALIASES.get(k, k)

def raw_norm_key(key: str) -> str:
    k = unicodedata.normalize("NFKC", str(key)).strip().lower()
    return re.sub(r"[^0-9a-zA-ZčšžćđČŠŽĆĐ]+", "_", k, flags=re.UNICODE).strip("_").lower()

def parse_comment(line: str) -> tuple[str, str] | None:
    if not line.startswith("#") or "=" not in line:
        return None
    body = line[1:].strip()
    key, value = body.split("=", 1)
    return key.strip(), value.strip()

def parse_thresholds(value: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not value:
        return out
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad --min-coverage item {item!r}; use key=value")
        k, v = item.split("=", 1)
        out[norm_key(k)] = float(v)
    return out

def flush_sentence(sent_rows: list[list[str]], sent_id: str | None, errors: list[dict[str, str]], stats: Counter[str]) -> None:
    if not sent_rows:
        return
    stats["sentences"] += 1
    roots = 0
    for parts in sent_rows:
        if parts[6] == "0":
            roots += 1
        if parts[3] == "_" or parts[2] == "_":
            stats["underscore_annotation_rows"] += 1
    if roots != 1:
        errors.append({"level": "sentence", "id": sent_id or "", "error": "root_count", "detail": str(roots)})

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check Wikivir CoNLL-U integrity before analysis.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--errors-output", required=True, type=Path)
    ap.add_argument("--metadata-output", required=True, type=Path)
    ap.add_argument("--coverage-output", required=True, type=Path)
    ap.add_argument("--min-docs", type=int, default=1000)
    ap.add_argument("--min-coverage", default="title=0.95,genre=0.80,century=0.70")
    ap.add_argument("--forbid-typo-keys", action="store_true", default=True)
    ap.add_argument("--allow-typo-keys", dest="forbid_typo_keys", action="store_false")
    ap.add_argument("--fail-on-error", action="store_true")
    args = ap.parse_args(argv)

    for p in [args.summary, args.errors_output, args.metadata_output, args.coverage_output]:
        p.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    docs: list[dict[str, str]] = []
    current_doc: dict[str, str] | None = None
    current_doc_id = ""
    doc_ids = []
    sent_ids = []
    current_sent_id: str | None = None
    sent_rows: list[list[str]] = []
    typo_keys: Counter[str] = Counter()
    raw_keys: Counter[str] = Counter()
    suspicious_glue_examples: list[str] = []

    def start_doc(doc_id: str):
        nonlocal current_doc, current_doc_id
        if current_doc is not None:
            docs.append(current_doc)
        current_doc_id = doc_id
        current_doc = {"doc_id": doc_id}
        doc_ids.append(doc_id)
        stats["documents"] += 1

    def add_meta(key: str, value: str):
        nonlocal current_doc
        raw = raw_norm_key(key)
        canon = norm_key(key)
        raw_keys[raw] += 1
        if raw in ALIASES:
            typo_keys[raw] += 1
        if canon and canon not in STRUCTURAL and current_doc is not None:
            if canon in current_doc and current_doc[canon] and current_doc[canon] != value:
                errors.append({"level": "document", "id": current_doc.get("doc_id", ""), "error": "metadata_conflict", "detail": f"{canon}: {current_doc[canon]!r} vs {value!r}"})
            current_doc[canon] = value

    with args.input.open("r", encoding="utf-8", errors="replace") as f:
        block_lines: list[str] = []
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                flush_sentence(sent_rows, current_sent_id, errors, stats)
                sent_rows = []
                current_sent_id = None
                block_lines = []
                continue
            if line.startswith("#"):
                kv = parse_comment(line)
                if kv:
                    key, value = kv
                    canon = norm_key(key)
                    if canon in {"newdoc", "newdoc_id"} or raw_norm_key(key) in {"newdoc", "newdoc_id"}:
                        start_doc(value)
                    elif canon == "sent_id":
                        current_sent_id = value
                        sent_ids.append(value)
                    elif canon in STRUCTURAL:
                        pass
                    else:
                        if current_doc is None:
                            start_doc("inferred-doc-000001")
                        add_meta(key, value)
                continue
            parts = line.split("\t")
            if len(parts) != 10:
                errors.append({"level": "token", "id": current_sent_id or f"line:{line_no}", "error": "bad_column_count", "detail": str(len(parts))})
                continue
            if "-" not in parts[0] and "." not in parts[0]:
                stats["tokens"] += 1
                sent_rows.append(parts)
                if len(sent_rows) >= 2:
                    prev = sent_rows[-2]
                    cur = sent_rows[-1]
                    if prev[9].find("SpaceAfter=No") >= 0 and re.search(r"[a-zčšžćđ]$", prev[1]) and re.search(r"^[A-ZČŠŽĆĐ]", cur[1]):
                        if len(suspicious_glue_examples) < 20:
                            suspicious_glue_examples.append(f"{current_sent_id or ''}: {prev[1]}{cur[1]}")

    flush_sentence(sent_rows, current_sent_id, errors, stats)
    if current_doc is not None:
        docs.append(current_doc)

    dup_doc_ids = [x for x, c in Counter(doc_ids).items() if c > 1]
    dup_sent_ids = [x for x, c in Counter(sent_ids).items() if c > 1]
    for x in dup_doc_ids[:100]:
        errors.append({"level": "document", "id": x, "error": "duplicate_doc_id", "detail": str(Counter(doc_ids)[x])})
    for x in dup_sent_ids[:100]:
        errors.append({"level": "sentence", "id": x, "error": "duplicate_sent_id", "detail": str(Counter(sent_ids)[x])})

    cols = sorted({k for d in docs for k in d})
    with args.metadata_output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader(); w.writerows(docs)

    thresholds = parse_thresholds(args.min_coverage)
    cov_rows = []
    coverage_failures = []
    n = len(docs)
    for key in sorted(set(thresholds) | {"title", "author", "genre", "century", "year", "publication"}):
        count = sum(1 for d in docs if str(d.get(key, "")).strip())
        cov = count / n if n else 0.0
        threshold = thresholds.get(key, "")
        cov_rows.append({"metadata_key": key, "documents_with_nonempty_value": count, "documents_total": n, "coverage": f"{cov:.6f}", "threshold": threshold})
        if key in thresholds and cov < thresholds[key]:
            coverage_failures.append(f"{key}: {cov:.3f} < {thresholds[key]:.3f}")
    with args.coverage_output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metadata_key", "documents_with_nonempty_value", "documents_total", "coverage", "threshold"], delimiter="\t")
        w.writeheader(); w.writerows(cov_rows)
    with args.errors_output.open("w", encoding="utf-8", newline="") as f:
        fields = ["level", "id", "error", "detail"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader(); w.writerows(errors)

    summary: dict[str, Any] = {
        "input": str(args.input),
        "documents": stats["documents"],
        "sentences": stats["sentences"],
        "tokens": stats["tokens"],
        "duplicate_doc_id_count": len(dup_doc_ids),
        "duplicate_sent_id_count": len(dup_sent_ids),
        "error_count": len(errors),
        "typo_metadata_keys": dict(typo_keys),
        "coverage_failures": coverage_failures,
        "suspicious_glue_examples": suspicious_glue_examples,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    hard_fail = False
    if stats["documents"] < args.min_docs:
        errors.append({"level": "corpus", "id": "", "error": "too_few_documents", "detail": f"{stats['documents']} < {args.min_docs}"})
        hard_fail = True
    if args.forbid_typo_keys and typo_keys:
        hard_fail = True
    if coverage_failures:
        hard_fail = True
    if args.fail_on_error and errors:
        hard_fail = True
    return 1 if hard_fail else 0

if __name__ == "__main__":
    raise SystemExit(main())
