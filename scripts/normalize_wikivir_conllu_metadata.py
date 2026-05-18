#!/usr/bin/env python3
"""Normalize and validate embedded Wikivir metadata in a CoNLL-U file.

This script is intentionally conservative: it rewrites only CoNLL-U comment
metadata keys and never touches token rows. It fixes recurring metadata typos
(e.g. gerne -> genre, cemtury -> century), consolidates metadata comments at the
start of each document, writes a clean metadata TSV, and fails fast if critical
coverage is too low.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

STRUCTURAL_KEYS = {
    "newdoc", "newdoc_id", "newpar", "newpar_id", "par", "par_id", "paragraph_id",
    "sent_id", "text", "global_columns", "global_metadata",
}
DOC_PREFIXES = ("doc_", "document_", "newdoc_", "meta_")

ALIASES = {
    "gerne": "genre",
    "genere": "genre",
    "gnere": "genre",
    "zvrst": "genre",
    "vrsta": "genre",
    "cemtury": "century",
    "centry": "century",
    "centurie": "century",
    "stoletje": "century",
    "cent": "century",
    "avtor": "author",
    "autor": "author",
    "naslov": "title",
    "leto": "year",
    "pub_year": "year",
    "publication_year": "year",
    "publikacija": "publication",
}

PRIORITY_KEYS = [
    "title", "author", "genre", "century", "year", "publication", "source", "url",
    "source_xml", "xml_doc_index", "original_doc_id",
]


def norm_key(key: str) -> str:
    k = unicodedata.normalize("NFKC", str(key)).strip().lower()
    k = re.sub(r"[^\w]+", "_", k, flags=re.UNICODE).strip("_")
    for prefix in DOC_PREFIXES:
        if k.startswith(prefix) and len(k) > len(prefix):
            k = k[len(prefix):]
            break
    return ALIASES.get(k, k)


def clean_value(value: Any) -> str:
    s = unicodedata.normalize("NFKC", str(value or ""))
    s = s.replace("\u00a0", " ").replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_comment(line: str) -> tuple[str, str] | None:
    if not line.startswith("#"):
        return None
    body = line[1:].strip()
    if "=" not in body:
        return None
    key, value = body.split("=", 1)
    return key.strip(), value.strip()


def is_structural_key(key: str) -> bool:
    return norm_key(key) in STRUCTURAL_KEYS


def parse_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    cur: list[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(line.rstrip("\n"))
    if cur:
        blocks.append(cur)
    return blocks


@dataclass
class Doc:
    doc_id: str
    blocks: list[list[str]] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    raw_key_map: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    conflicts: list[str] = field(default_factory=list)


def safe_doc_id(s: str, fallback: str) -> str:
    s = clean_value(s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w.:-]+", "_", s, flags=re.UNICODE).strip("_")
    return s or fallback


def add_meta(doc: Doc, raw_key: str, value: str) -> None:
    nk = norm_key(raw_key)
    if nk in STRUCTURAL_KEYS or not nk:
        return
    value = clean_value(value)
    if not value:
        return
    doc.raw_key_map[nk].add(raw_key.strip())
    if nk in doc.metadata and doc.metadata[nk] and doc.metadata[nk] != value:
        # Prefer canonical spelling over typo aliases; otherwise keep first and warn.
        existing_raws = {rk.lower() for rk in doc.raw_key_map.get(nk, set())}
        raw_norm = raw_key.strip().lower()
        if raw_norm == nk and existing_raws and all(r != nk for r in existing_raws):
            doc.conflicts.append(f"{nk}: replaced alias value {doc.metadata[nk]!r} with canonical value {value!r}")
            doc.metadata[nk] = value
        else:
            doc.conflicts.append(f"{nk}: kept {doc.metadata[nk]!r}, ignored conflicting {value!r}")
    else:
        doc.metadata[nk] = value


def parse_docs(blocks: list[list[str]]) -> list[Doc]:
    docs: list[Doc] = []
    current: Doc | None = None
    for block in blocks:
        newdoc_id: str | None = None
        for line in block:
            kv = parse_comment(line)
            if not kv:
                continue
            key, value = kv
            if norm_key(key) in {"newdoc", "newdoc_id"}:
                newdoc_id = value
                break
        if newdoc_id is not None or current is None:
            doc_id = safe_doc_id(newdoc_id or f"doc-{len(docs)+1:06d}", f"doc-{len(docs)+1:06d}")
            current = Doc(doc_id=doc_id)
            docs.append(current)
        current.blocks.append(block)
        for line in block:
            kv = parse_comment(line)
            if not kv:
                continue
            key, value = kv
            nk = norm_key(key)
            if nk in {"newdoc", "newdoc_id", "sent_id", "text", "newpar", "newpar_id", "par_id", "paragraph_id"}:
                continue
            add_meta(current, key, value)
    # Ensure uniqueness
    counts = Counter(d.doc_id for d in docs)
    seen = Counter()
    for i, doc in enumerate(docs, start=1):
        seen[doc.doc_id] += 1
        if counts[doc.doc_id] > 1:
            old = doc.doc_id
            doc.metadata.setdefault("original_doc_id", old)
            doc.doc_id = f"{old}__{seen[old]:04d}"
    return docs


def ordered_metadata_items(meta: dict[str, str]) -> list[tuple[str, str]]:
    keys = [k for k in PRIORITY_KEYS if k in meta]
    keys += sorted(k for k in meta if k not in set(keys))
    return [(k, meta[k]) for k in keys if str(meta[k]).strip()]


def is_metadata_comment(key: str) -> bool:
    nk = norm_key(key)
    if nk in STRUCTURAL_KEYS:
        return False
    # Treat all non-structural comments as metadata comments in Wikivir document scope.
    return True


def rewrite_doc(doc: Doc, source_name: str, replace_existing: bool = True) -> list[str]:
    out_blocks: list[str] = []
    first = True
    for block in doc.blocks:
        comments = [ln for ln in block if ln.startswith("#")]
        tokens = [ln for ln in block if not ln.startswith("#")]
        new_comments: list[str] = []
        saw_newdoc = False
        for line in comments:
            kv = parse_comment(line)
            if not kv:
                new_comments.append(line)
                continue
            key, value = kv
            nk = norm_key(key)
            if nk in {"newdoc", "newdoc_id"}:
                if first:
                    new_comments.append(f"# newdoc id = {doc.doc_id}")
                    saw_newdoc = True
                # Drop repeated newdoc comments inside the same doc.
                continue
            if is_metadata_comment(key):
                # Consolidated canonical metadata is emitted once after newdoc id.
                continue
            new_comments.append(line)
        if first:
            if not saw_newdoc:
                new_comments.insert(0, f"# newdoc id = {doc.doc_id}")
            insert_at = 1 if new_comments and new_comments[0].startswith("# newdoc id") else 0
            canonical = [f"# {k} = {v}" for k, v in ordered_metadata_items(doc.metadata)]
            alias_notes = {k: sorted(v) for k, v in doc.raw_key_map.items() if any(norm_key(x) != k for x in v)}
            if alias_notes:
                canonical.append(f"# metadata_normalized_from = {json.dumps(alias_notes, ensure_ascii=False, sort_keys=True)}")
            new_comments[insert_at:insert_at] = canonical
            first = False
        out_blocks.append("\n".join(new_comments + tokens))
    return out_blocks


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pd is not None:
        pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    else:
        import csv
        keys = sorted({k for row in rows for k in row})
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
            w.writeheader()
            for row in rows:
                w.writerow(row)


def coverage_rows(docs: list[Doc]) -> list[dict[str, Any]]:
    keys = sorted({k for d in docs for k in d.metadata})
    rows = []
    n = len(docs)
    for key in keys:
        count = sum(1 for d in docs if clean_value(d.metadata.get(key, "")))
        rows.append({"metadata_key": key, "documents_with_nonempty_value": count, "total_documents": n, "coverage": count / n if n else 0.0})
    return rows


def parse_min_coverage(values: list[str]) -> dict[str, float]:
    out = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"--min-coverage must be key=value, got: {item}")
        k, v = item.split("=", 1)
        out[norm_key(k)] = float(v)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--metadata-output", required=True, type=Path)
    ap.add_argument("--coverage-output", required=True, type=Path)
    ap.add_argument("--warnings-output", required=True, type=Path)
    ap.add_argument("--summary-output", type=Path)
    ap.add_argument("--min-coverage", action="append", default=[], help="Required coverage threshold, e.g. genre=0.85. Can be repeated.")
    ap.add_argument("--fail-on-critical", action="store_true")
    args = ap.parse_args(argv)

    text = args.input.read_text(encoding="utf-8", errors="replace")
    docs = parse_docs(parse_blocks(text))
    if not docs:
        raise SystemExit("No documents found in CoNLL-U.")

    out_blocks: list[str] = []
    for doc in docs:
        out_blocks.extend(rewrite_doc(doc, args.input.name))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(out_blocks) + "\n", encoding="utf-8")

    meta_rows = [{"doc_id": d.doc_id, **d.metadata} for d in docs]
    write_tsv(args.metadata_output, meta_rows)
    cov = coverage_rows(docs)
    write_tsv(args.coverage_output, cov)

    warnings: list[dict[str, Any]] = []
    for d in docs:
        for c in d.conflicts:
            warnings.append({"severity": "warning", "doc_id": d.doc_id, "message": c})
    alias_counter = Counter()
    for d in docs:
        for canon, raws in d.raw_key_map.items():
            for raw in raws:
                if norm_key(raw) != raw.strip().lower():
                    alias_counter[(raw, canon)] += 1
    for (raw, canon), count in alias_counter.items():
        warnings.append({"severity": "info", "doc_id": "", "message": f"normalized key {raw!r} -> {canon!r} in {count} document(s)"})

    cov_map = {r["metadata_key"]: float(r["coverage"]) for r in cov}
    failed = []
    for key, threshold in parse_min_coverage(args.min_coverage).items():
        actual = cov_map.get(key, 0.0)
        if actual < threshold:
            failed.append((key, actual, threshold))
            warnings.append({"severity": "critical", "doc_id": "", "message": f"coverage for {key} is {actual:.3f}, below required {threshold:.3f}"})
    write_tsv(args.warnings_output, warnings)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps({
            "input": str(args.input), "output": str(args.output), "documents": len(docs),
            "coverage": cov_map, "failed_thresholds": failed,
            "warning_count": len(warnings),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote clean CoNLL-U: {args.output}")
    print(f"Wrote metadata TSV: {args.metadata_output}")
    print(f"Wrote coverage TSV: {args.coverage_output}")
    if failed and args.fail_on_critical:
        print("Critical metadata coverage thresholds failed:", file=sys.stderr)
        for key, actual, threshold in failed:
            print(f"  {key}: {actual:.3f} < {threshold:.3f}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
