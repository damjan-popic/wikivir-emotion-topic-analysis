#!/usr/bin/env python3
"""Normalize Wikivir XML document metadata before CLASSLA annotation.

This is a preflight/repair script, not an analysis script. It fixes observed
attribute-name typos (e.g. gerne -> genre, cemtury -> century), normalizes simple
century/year values, guarantees unique document IDs, and writes a metadata audit.

It expects the student's amended XML to be well-formed. If parsing fails, stop
and fix the XML rather than silently recovering.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

ALIASES = {
    "gerne": "genre", "genere": "genre", "gendre": "genre", "gernre": "genre", "zvrst": "genre", "vrsta": "genre",
    "cemtury": "century", "centry": "century", "centruy": "century", "cenutry": "century", "stoletje": "century", "century_": "century",
    "yeaar": "year", "yer": "year", "leto": "year", "date_year": "year",
    "avtor": "author", "autor": "author", "auhtor": "author", "writer": "author",
    "naslov": "title", "titel": "title", "name": "title",
    "pubication": "publication", "publicaton": "publication", "publikacija": "publication",
    "source_url": "url", "uri": "url",
}
CORE_KEYS = ["doc_id", "title", "author", "genre", "century", "year", "publication", "url", "source"]

ROMAN = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
    "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17, "xviii": 18,
    "xix": 19, "xx": 20, "xxi": 21, "xxii": 22,
}

def norm_key(key: str) -> str:
    if key == XML_ID:
        return "xml_id"
    k = unicodedata.normalize("NFKC", str(key)).strip().lower()
    k = re.sub(r"[^0-9a-zA-ZčšžćđČŠŽĆĐ]+", "_", k, flags=re.UNICODE).strip("_")
    k = k.lower()
    return ALIASES.get(k, k)

def clean_value(value: Any) -> str:
    v = unicodedata.normalize("NFKC", str(value or ""))
    v = re.sub(r"\s+", " ", v).strip()
    return v

def normalize_year(value: str) -> str:
    v = clean_value(value)
    if not v:
        return ""
    m = re.search(r"(?<!\d)(1[5-9]\d{2}|20\d{2})(?!\d)", v)
    return m.group(1) if m else v

def normalize_century(value: str) -> str:
    v = clean_value(value)
    if not v:
        return ""
    low = v.lower().strip().rstrip(".")
    low = re.sub(r"\s*(stol\.?|stoletje|century)\s*$", "", low).strip().rstrip(".")
    if low in ROMAN:
        return str(ROMAN[low])
    m = re.search(r"(?<!\d)(\d{1,2})(?!\d)", low)
    return str(int(m.group(1))) if m else v

def derive_century_from_year(year: str) -> str:
    y = normalize_year(year)
    if re.fullmatch(r"\d{4}", y):
        return str((int(y) - 1) // 100 + 1)
    return ""

def safe_doc_id(value: str, index: int) -> str:
    raw = clean_value(value) or f"wikivir-{index:06d}"
    s = re.sub(r"\s+", "_", raw)
    s = re.sub(r"[^\w.:-]+", "_", s, flags=re.UNICODE).strip("_")
    return s or f"wikivir-{index:06d}"

def parse_thresholds(value: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not value:
        return out
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad --min-coverage item: {item!r}; use key=value")
        k, v = item.split("=", 1)
        out[norm_key(k)] = float(v)
    return out

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Normalize Wikivir XML metadata before annotation.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--metadata-output", required=True, type=Path)
    ap.add_argument("--fixes-output", required=True, type=Path)
    ap.add_argument("--coverage-output", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--doc-tag", default="doc")
    ap.add_argument("--id-attribute", default="id")
    ap.add_argument("--required", default="title,genre,century", help="Comma-separated canonical metadata keys to monitor.")
    ap.add_argument("--min-coverage", default="title=0.95,genre=0.80,century=0.70", help="Comma-separated key=ratio thresholds.")
    ap.add_argument("--fail-on-typo-attrs", action="store_true", help="Fail if any typo aliases were corrected. Useful for final locked corpora.")
    ap.add_argument("--fail-on-threshold", action="store_true", help="Exit nonzero if coverage thresholds fail.")
    ap.add_argument("--write-back-updated-attrs", action="store_true", default=True)
    args = ap.parse_args(argv)

    for p in [args.output, args.metadata_output, args.fixes_output, args.coverage_output, args.summary]:
        p.parent.mkdir(parents=True, exist_ok=True)

    try:
        tree = ET.parse(args.input)
    except ET.ParseError as exc:
        print(f"XML is not well-formed: {exc}", file=sys.stderr)
        return 2
    root = tree.getroot()
    docs = list(root.iter(args.doc_tag))
    if not docs:
        print(f"No <{args.doc_tag}> elements found.", file=sys.stderr)
        return 2

    used_ids: Counter[str] = Counter()
    metadata_rows: list[dict[str, str]] = []
    fixes: list[dict[str, str]] = []
    unknown_keys: Counter[str] = Counter()
    alias_count = 0

    for idx, elem in enumerate(docs, start=1):
        original_attrs = dict(elem.attrib)
        canonical: dict[str, str] = {}
        original_id = original_attrs.get(args.id_attribute) or original_attrs.get(XML_ID) or original_attrs.get("doc_id") or ""

        for raw_key, raw_value in original_attrs.items():
            ck = norm_key(raw_key)
            rv = clean_value(raw_value)
            if ck != raw_key and raw_key != XML_ID:
                alias_count += 1
                fixes.append({"doc_index": str(idx), "action": "rename_key", "old_key": str(raw_key), "new_key": ck, "old_value": rv, "new_value": rv})
            if ck in {"id", "xml_id", "doc_id"}:
                # handled below as doc_id
                continue
            if ck in canonical and canonical[ck] and rv and canonical[ck] != rv:
                fixes.append({"doc_index": str(idx), "action": "conflict_keep_first", "old_key": str(raw_key), "new_key": ck, "old_value": rv, "new_value": canonical[ck]})
                continue
            canonical.setdefault(ck, rv)

        # Normalize values.
        for key, func in [("year", normalize_year), ("century", normalize_century)]:
            if key in canonical:
                old = canonical[key]
                new = func(old)
                if new != old:
                    fixes.append({"doc_index": str(idx), "action": "normalize_value", "old_key": key, "new_key": key, "old_value": old, "new_value": new})
                canonical[key] = new
        if not canonical.get("century") and canonical.get("year"):
            derived = derive_century_from_year(canonical["year"])
            if derived:
                canonical["century"] = derived
                fixes.append({"doc_index": str(idx), "action": "derive_century_from_year", "old_key": "year", "new_key": "century", "old_value": canonical["year"], "new_value": derived})

        base_id = safe_doc_id(original_id or canonical.get("title", ""), idx)
        used_ids[base_id] += 1
        doc_id = base_id if used_ids[base_id] == 1 else f"{base_id}__{used_ids[base_id]:04d}"
        if doc_id != original_id:
            fixes.append({"doc_index": str(idx), "action": "set_doc_id", "old_key": args.id_attribute, "new_key": "doc_id", "old_value": clean_value(original_id), "new_value": doc_id})
        canonical["doc_id"] = doc_id

        # Rewrite attributes canonically: keep text content, canonicalize doc attrs.
        elem.attrib.clear()
        elem.set(args.id_attribute, doc_id)
        for key in sorted(k for k in canonical if k != "doc_id"):
            if canonical[key] != "":
                elem.set(key, canonical[key])

        metadata_rows.append({k: canonical.get(k, "") for k in CORE_KEYS} | {k: v for k, v in canonical.items() if k not in CORE_KEYS})
        for k in canonical:
            if k not in set(CORE_KEYS) | {"period", "date", "type", "subgenre", "collection", "source_id"}:
                unknown_keys[k] += 1

    # Write XML and TSVs.
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    all_cols = list(dict.fromkeys(CORE_KEYS + sorted({k for row in metadata_rows for k in row if k not in CORE_KEYS})))
    with args.metadata_output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_cols, delimiter="\t", extrasaction="ignore")
        w.writeheader(); w.writerows(metadata_rows)
    with args.fixes_output.open("w", encoding="utf-8", newline="") as f:
        fields = ["doc_index", "action", "old_key", "new_key", "old_value", "new_value"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader(); w.writerows(fixes)

    required = [norm_key(x) for x in args.required.split(",") if x.strip()]
    thresholds = parse_thresholds(args.min_coverage)
    coverage_rows = []
    failures = []
    n = len(metadata_rows)
    for key in sorted(set(required) | set(thresholds) | {"title", "author", "genre", "century", "year", "publication"}):
        cnt = sum(1 for row in metadata_rows if clean_value(row.get(key, "")))
        cov = cnt / n if n else 0.0
        coverage_rows.append({"metadata_key": key, "documents_with_nonempty_value": cnt, "documents_total": n, "coverage": f"{cov:.6f}", "threshold": thresholds.get(key, "")})
        if key in thresholds and cov < thresholds[key]:
            failures.append(f"{key}: {cov:.3f} < {thresholds[key]:.3f}")
    with args.coverage_output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metadata_key", "documents_with_nonempty_value", "documents_total", "coverage", "threshold"], delimiter="\t")
        w.writeheader(); w.writerows(coverage_rows)

    summary = {
        "input": str(args.input), "output": str(args.output), "documents": n,
        "fixes": len(fixes), "alias_fixes": alias_count,
        "duplicate_doc_ids_fixed": sum(c - 1 for c in used_ids.values() if c > 1),
        "unknown_metadata_keys": dict(unknown_keys.most_common()),
        "coverage_failures": failures,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_typo_attrs and alias_count:
        return 3
    if args.fail_on_threshold and failures:
        return 4
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
