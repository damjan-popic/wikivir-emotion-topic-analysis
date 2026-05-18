#!/usr/bin/env python3
"""Apply reviewed metadata key/value fixes to Wikivir-style XML <doc> tags.

This script deliberately modifies only <doc ...> start-tag attributes.
It leaves document text untouched. It is intended for the manual-QA workflow:

  1. inspect metadata_value_inventory.tsv / metadata_key_inventory.tsv
  2. edit key/value maps manually
  3. apply only reviewed fixes
  4. re-inventory before CLASSLA annotation

Conflict policy:
- abort: fail if a renamed key collides with an existing different value
- prefer-existing: keep the canonical key's existing value, log the incoming value
- prefer-incoming: overwrite canonical key with renamed typo key's value
- join: join both values with ' | '
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

DOC_TAG_RE_TEMPLATE = r"<(?P<tag>{tag})\b(?P<attrs>(?:[^<>\"']+|\"[^\"]*\"|'[^']*')*)>"


def read_tsv(path: Path) -> List[dict]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def parse_attrs(attrs_text: str, doc_index: int) -> Dict[str, str]:
    # The regex captures the leading whitespace before attributes; wrap in fake XML.
    wrapped = f"<x{attrs_text}/>"
    try:
        elem = ET.fromstring(wrapped)
    except ET.ParseError as e:
        raise RuntimeError(f"Cannot parse <doc> attributes at XML doc {doc_index}: {e}") from e
    return dict(elem.attrib)


def attrs_to_xml(attrs: Dict[str, str]) -> str:
    return " ".join(f'{k}="{html.escape(str(v), quote=True)}"' for k, v in attrs.items())


def load_key_map(path: Path) -> Dict[str, Tuple[str, str]]:
    rows = read_tsv(path)
    out = {}
    for r in rows:
        orig = (r.get("original_key") or "").strip()
        canon = (r.get("canonical_key") or orig).strip()
        action = (r.get("action") or "keep").strip().lower()
        if not orig:
            continue
        if action not in {"keep", "rename", "drop"}:
            raise SystemExit(f"Invalid action for key {orig!r}: {action!r}")
        # Helpful: if canonical differs but action left as keep, treat as rename.
        if action == "keep" and canon != orig:
            action = "rename"
        out[orig] = (canon, action)
    return out


def load_value_map(path: Path) -> Dict[Tuple[str, str], Tuple[str, str]]:
    rows = read_tsv(path)
    out = {}
    for r in rows:
        key = (r.get("key") or "").strip()
        old = (r.get("old_value") or "").strip()
        new = (r.get("new_value") or "").strip()
        action = (r.get("action") or "replace").strip().lower()
        if not key or not old:
            continue
        if action not in {"replace", "keep"}:
            raise SystemExit(f"Invalid value action for {key}={old!r}: {action!r}")
        out[(key, old)] = (new, action)
    return out


def inventory_attrs(attrs_by_doc: List[Dict[str, str]]) -> Tuple[List[dict], List[dict], List[dict]]:
    key_counts = Counter()
    empty_counts = Counter()
    values = defaultdict(Counter)
    all_keys = set()
    doc_rows = []
    for i, attrs in enumerate(attrs_by_doc, 1):
        row = {"xml_doc_index": str(i)}
        for k, v in attrs.items():
            all_keys.add(k)
            key_counts[k] += 1
            if not str(v).strip():
                empty_counts[k] += 1
            values[k][v] += 1
            row[k] = v
        doc_rows.append(row)
    key_rows = []
    for k in sorted(all_keys):
        key_rows.append({
            "key": k,
            "count": key_counts[k],
            "empty_count": empty_counts[k],
            "unique_values": len(values[k]),
            "top_values": " | ".join(f"{v} ({c})" for v, c in values[k].most_common(12)),
        })
    value_rows = []
    for k in sorted(all_keys):
        for v, c in values[k].most_common():
            value_rows.append({"key": k, "value": v, "count": c})
    fields = ["xml_doc_index"] + sorted(all_keys)
    return key_rows, value_rows, doc_rows, fields


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--doc-tag", default="doc")
    ap.add_argument("--key-map", required=True)
    ap.add_argument("--value-map")
    ap.add_argument("--on-conflict", choices=["abort", "prefer-existing", "prefer-incoming", "join"], default="abort")
    ap.add_argument("--fixes", required=True)
    ap.add_argument("--conflicts", required=True)
    ap.add_argument("--inventory-out-dir", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    text = in_path.read_text(encoding="utf-8", errors="replace")

    key_map = load_key_map(Path(args.key_map))
    value_map = load_value_map(Path(args.value_map)) if args.value_map else {}

    fixes: List[dict] = []
    conflicts: List[dict] = []
    new_attrs_for_inventory: List[Dict[str, str]] = []
    doc_index = 0

    tag_re = re.compile(DOC_TAG_RE_TEMPLATE.format(tag=re.escape(args.doc_tag)), flags=re.DOTALL)

    def repl(m: re.Match) -> str:
        nonlocal doc_index
        doc_index += 1
        attrs = parse_attrs(m.group("attrs"), doc_index)
        title = attrs.get("title", "")

        new_attrs: Dict[str, str] = {}
        dropped_attrs: List[Tuple[str, str]] = []

        # Key normalization first.
        for k, v in attrs.items():
            canon, action = key_map.get(k, (k, "keep"))
            if action == "drop":
                dropped_attrs.append((k, v))
                fixes.append({
                    "xml_doc_index": doc_index, "title": title,
                    "fix_type": "key", "original_key": k, "canonical_key": "",
                    "original_value": v, "new_value": "", "status": "dropped",
                })
                continue
            target = canon if action == "rename" else k
            if target in new_attrs and new_attrs[target] != v:
                conflicts.append({
                    "xml_doc_index": doc_index, "title": title,
                    "original_key": k, "canonical_key": target,
                    "existing_value": new_attrs[target], "incoming_value": v,
                    "policy": args.on_conflict,
                })
                if args.on_conflict == "abort":
                    raise RuntimeError(
                        f"Conflict at doc {doc_index}: {k}->{target}, existing={new_attrs[target]!r}, incoming={v!r}"
                    )
                if args.on_conflict == "prefer-existing":
                    fixes.append({
                        "xml_doc_index": doc_index, "title": title,
                        "fix_type": "key", "original_key": k, "canonical_key": target,
                        "original_value": v, "new_value": new_attrs[target], "status": "renamed_conflict_prefer_existing",
                    })
                    continue
                if args.on_conflict == "prefer-incoming":
                    new_attrs[target] = v
                elif args.on_conflict == "join":
                    vals = []
                    for x in [new_attrs[target], v]:
                        if x and x not in vals:
                            vals.append(x)
                    new_attrs[target] = " | ".join(vals)
            else:
                new_attrs[target] = v
            if action == "rename":
                fixes.append({
                    "xml_doc_index": doc_index, "title": title,
                    "fix_type": "key", "original_key": k, "canonical_key": target,
                    "original_value": v, "new_value": v, "status": "renamed",
                })

        # Value normalization second, after keys have canonical names.
        for k in list(new_attrs.keys()):
            old = str(new_attrs[k]).strip()
            if (k, old) in value_map:
                new, action = value_map[(k, old)]
                if action == "replace" and new != old:
                    new_attrs[k] = new
                    fixes.append({
                        "xml_doc_index": doc_index, "title": title,
                        "fix_type": "value", "original_key": k, "canonical_key": k,
                        "original_value": old, "new_value": new, "status": "value_replaced",
                    })

        new_attrs_for_inventory.append(dict(new_attrs))
        attrs_xml = attrs_to_xml(new_attrs)
        if attrs_xml:
            return f"<{args.doc_tag} {attrs_xml}>"
        return f"<{args.doc_tag}>"

    try:
        new_text = tag_re.sub(repl, text)
    except Exception as e:
        write_tsv(Path(args.conflicts), conflicts, ["xml_doc_index", "title", "original_key", "canonical_key", "existing_value", "incoming_value", "policy"])
        raise

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(new_text, encoding="utf-8")

    inv_dir = Path(args.inventory_out_dir)
    inv_dir.mkdir(parents=True, exist_ok=True)
    key_rows, value_rows, doc_rows, doc_fields = inventory_attrs(new_attrs_for_inventory)
    write_tsv(inv_dir / "metadata_key_inventory.tsv", key_rows, ["key", "count", "empty_count", "unique_values", "top_values"])
    write_tsv(inv_dir / "metadata_value_inventory.tsv", value_rows, ["key", "value", "count"])
    write_tsv(inv_dir / "metadata_docs.tsv", doc_rows, doc_fields)

    write_tsv(Path(args.fixes), fixes, ["xml_doc_index", "title", "fix_type", "original_key", "canonical_key", "original_value", "new_value", "status"])
    write_tsv(Path(args.conflicts), conflicts, ["xml_doc_index", "title", "original_key", "canonical_key", "existing_value", "incoming_value", "policy"])

    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "doc_tag": args.doc_tag,
        "documents_processed": doc_index,
        "fixes": len(fixes),
        "conflicts": len(conflicts),
        "metadata_keys_after": [r["key"] for r in key_rows],
        "inventory_out_dir": str(inv_dir),
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
