#!/usr/bin/env python3
"""Manual metadata inventory and selective normalization for Wikivir-style XML/CoNLL-U.

Use this when you do not want a hardcoded typo list. The workflow is:

1. inventory XML metadata keys/values
2. edit the generated key-map TSV manually
3. apply only the chosen key renames/drops to the XML
4. inspect CoNLL-U metadata after reannotation

The XML apply step rewrites only <doc ...> start tags and leaves document text/content
otherwise untouched. It reads the XML as text, so it can handle large files on machines
with enough RAM. For a 60M-token Wikivir corpus this is usually acceptable on a 64GB box,
but run it outside a huge Python analysis process.
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
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DEFAULT_EXPECTED_KEYS = [
    "id", "title", "author", "genre", "century", "year", "publication", "source", "url",
    "subtitle", "translator", "editor", "collection", "language", "license", "date",
]

DOC_TAG_RE_TEMPLATE = r"<(?P<tag>{tag})\b(?P<attrs>(?:[^<>\"']+|\"[^\"]*\"|'[^']*')*)>"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def read_tsv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def sample_values(counter: Counter, n: int) -> str:
    vals = []
    for value, count in counter.most_common(n):
        v = str(value).replace("\t", " ").replace("\n", " ").strip()
        if len(v) > 80:
            v = v[:77] + "..."
        vals.append(f"{v} ({count})")
    return " | ".join(vals)


def inventory_xml(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    expected = [x.strip() for x in args.expected_keys.split(",") if x.strip()]
    expected_set = set(expected)

    key_counts: Counter = Counter()
    key_empty: Counter = Counter()
    value_counts: Dict[str, Counter] = defaultdict(Counter)
    docs: List[dict] = []
    all_keys = set()
    doc_count = 0

    for event, elem in ET.iterparse(str(in_path), events=("start",)):
        if local_name(elem.tag) != args.doc_tag:
            continue
        doc_count += 1
        row = {"xml_doc_index": str(doc_count)}
        for k, v in elem.attrib.items():
            k = str(k).strip()
            v = str(v).strip()
            all_keys.add(k)
            row[k] = v
            key_counts[k] += 1
            if not v:
                key_empty[k] += 1
            value_counts[k][v] += 1
        docs.append(row)
        elem.clear()

    all_keys_sorted = sorted(all_keys)
    doc_fields = ["xml_doc_index"] + all_keys_sorted
    write_tsv(out_dir / "metadata_docs.tsv", docs, doc_fields)

    key_rows = []
    for key in all_keys_sorted:
        key_rows.append({
            "key": key,
            "count": key_counts[key],
            "empty_count": key_empty[key],
            "unique_values": len(value_counts[key]),
            "sample_values": sample_values(value_counts[key], args.sample_values),
            "expected": "yes" if key in expected_set else "no",
            "close_expected_keys": ", ".join(get_close_matches(key, expected, n=4, cutoff=0.60)),
        })
    write_tsv(
        out_dir / "metadata_key_inventory.tsv",
        key_rows,
        ["key", "count", "empty_count", "unique_values", "sample_values", "expected", "close_expected_keys"],
    )

    value_rows = []
    for key in all_keys_sorted:
        for value, count in value_counts[key].most_common():
            value_rows.append({"key": key, "value": value, "count": count})
    write_tsv(out_dir / "metadata_value_inventory.tsv", value_rows, ["key", "value", "count"])

    # Manual map template. By default everything is kept. User edits only the wrong rows.
    map_rows = []
    for row in key_rows:
        key = row["key"]
        close = row["close_expected_keys"].split(", ")[0] if row["close_expected_keys"] else ""
        suggested = key if row["expected"] == "yes" else close
        map_rows.append({
            "original_key": key,
            "canonical_key": key,
            "action": "keep",
            "suggested_canonical_key": suggested,
            "count": row["count"],
            "sample_values": row["sample_values"],
            "notes": "edit canonical_key/action only if this key is wrong",
        })
    write_tsv(
        out_dir / "metadata_key_map_TEMPLATE.tsv",
        map_rows,
        ["original_key", "canonical_key", "action", "suggested_canonical_key", "count", "sample_values", "notes"],
    )

    summary = {
        "input": str(in_path),
        "doc_tag": args.doc_tag,
        "documents": doc_count,
        "metadata_keys": len(all_keys_sorted),
        "unknown_keys": [r["key"] for r in key_rows if r["expected"] == "no"],
        "outputs": {
            "metadata_docs": str(out_dir / "metadata_docs.tsv"),
            "metadata_key_inventory": str(out_dir / "metadata_key_inventory.tsv"),
            "metadata_value_inventory": str(out_dir / "metadata_value_inventory.tsv"),
            "metadata_key_map_template": str(out_dir / "metadata_key_map_TEMPLATE.tsv"),
        },
    }
    (out_dir / "metadata_inventory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Documents: {doc_count}")
    print(f"Metadata keys: {len(all_keys_sorted)}")
    print(f"Wrote: {out_dir / 'metadata_key_inventory.tsv'}")
    print(f"Wrote: {out_dir / 'metadata_value_inventory.tsv'}")
    print(f"Wrote: {out_dir / 'metadata_docs.tsv'}")
    print(f"Edit:  {out_dir / 'metadata_key_map_TEMPLATE.tsv'}")
    return 0


def attrs_to_xml(attrs: Dict[str, str]) -> str:
    parts = []
    for k, v in attrs.items():
        escaped = html.escape(str(v), quote=True)
        parts.append(f'{k}="{escaped}"')
    return " ".join(parts)


@dataclass
class MapEntry:
    original_key: str
    canonical_key: str
    action: str


def load_key_map(path: Path) -> Dict[str, MapEntry]:
    rows = read_tsv(path)
    mapping = {}
    required = {"original_key", "canonical_key", "action"}
    if rows and not required.issubset(rows[0].keys()):
        raise SystemExit(f"Key map must contain columns: {', '.join(sorted(required))}")
    for row in rows:
        orig = row.get("original_key", "").strip()
        canon = row.get("canonical_key", "").strip() or orig
        action = row.get("action", "keep").strip().lower() or "keep"
        if not orig:
            continue
        if action not in {"keep", "rename", "drop"}:
            raise SystemExit(f"Invalid action for {orig}: {action}. Use keep, rename, or drop.")
        if action == "keep" and canon != orig:
            # Helpful leniency: if user changed canonical_key but forgot action.
            action = "rename"
        mapping[orig] = MapEntry(orig, canon, action)
    return mapping


def parse_doc_attrs(attrs_text: str, doc_index: int) -> Dict[str, str]:
    # Parse attributes by wrapping them in a fake XML element.
    wrapped = f"<x{attrs_text}/>"
    try:
        elem = ET.fromstring(wrapped)
    except ET.ParseError as e:
        raise RuntimeError(f"Could not parse <doc> attributes at doc {doc_index}: {e}") from e
    return dict(elem.attrib)


def apply_map(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_path = Path(args.output)
    map_path = Path(args.key_map)
    ensure_dir(out_path.parent)
    ensure_dir(Path(args.fixes).parent)
    ensure_dir(Path(args.conflicts).parent)

    mapping = load_key_map(map_path)
    text = in_path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(DOC_TAG_RE_TEMPLATE.format(tag=re.escape(args.doc_tag)), flags=re.DOTALL)

    fixes: List[dict] = []
    conflicts: List[dict] = []
    doc_index = 0
    changed_docs = 0

    def repl(match: re.Match) -> str:
        nonlocal doc_index, changed_docs
        doc_index += 1
        tag = match.group("tag")
        attrs_text = match.group("attrs")
        attrs = parse_doc_attrs(attrs_text, doc_index)
        new_attrs: Dict[str, str] = {}
        changed = False

        # Prefer already-canonical attributes when conflicts happen.
        original_keys = set(attrs.keys())

        for key, value in attrs.items():
            entry = mapping.get(key, MapEntry(key, key, "keep"))
            if entry.action == "drop":
                fixes.append({
                    "xml_doc_index": doc_index,
                    "title": attrs.get("title", ""),
                    "original_key": key,
                    "canonical_key": "",
                    "action": "drop",
                    "value": value,
                    "status": "dropped",
                })
                changed = True
                continue

            target = entry.canonical_key if entry.action == "rename" else key
            if target != key:
                changed = True

            if target in new_attrs and new_attrs[target] != value:
                # If there was an original correct key and this is the typo, prefer existing.
                conflict = {
                    "xml_doc_index": doc_index,
                    "title": attrs.get("title", ""),
                    "original_key": key,
                    "canonical_key": target,
                    "existing_value": new_attrs[target],
                    "incoming_value": value,
                    "policy": args.on_conflict,
                }
                if args.on_conflict == "abort":
                    conflicts.append(conflict)
                    continue
                elif args.on_conflict == "prefer-existing":
                    conflicts.append({**conflict, "status": "kept_existing"})
                    continue
                elif args.on_conflict == "prefer-incoming":
                    conflicts.append({**conflict, "status": "replaced_existing"})
                    new_attrs[target] = value
                elif args.on_conflict == "combine":
                    conflicts.append({**conflict, "status": "combined"})
                    vals = [x for x in [new_attrs[target], value] if x]
                    new_attrs[target] = " | ".join(dict.fromkeys(vals))
                changed = True
            else:
                new_attrs[target] = value

            if target != key:
                fixes.append({
                    "xml_doc_index": doc_index,
                    "title": attrs.get("title", ""),
                    "original_key": key,
                    "canonical_key": target,
                    "action": "rename",
                    "value": value,
                    "status": "renamed",
                })

        if changed:
            changed_docs += 1
        return f"<{tag} {attrs_to_xml(new_attrs)}>"

    new_text = pattern.sub(repl, text)

    if args.on_conflict == "abort" and conflicts:
        write_tsv(Path(args.conflicts), conflicts, [
            "xml_doc_index", "title", "original_key", "canonical_key", "existing_value", "incoming_value", "policy"
        ])
        print(f"Conflicts found: {len(conflicts)}. Wrote {args.conflicts}. Output not written.", file=sys.stderr)
        return 2

    out_path.write_text(new_text, encoding="utf-8")

    fix_fields = ["xml_doc_index", "title", "original_key", "canonical_key", "action", "value", "status"]
    write_tsv(Path(args.fixes), fixes, fix_fields)
    conflict_fields = ["xml_doc_index", "title", "original_key", "canonical_key", "existing_value", "incoming_value", "policy", "status"]
    write_tsv(Path(args.conflicts), conflicts, conflict_fields)

    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "key_map": str(map_path),
        "doc_tag": args.doc_tag,
        "documents_seen": doc_index,
        "documents_changed": changed_docs,
        "fixes": len(fixes),
        "conflicts": len(conflicts),
        "on_conflict": args.on_conflict,
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def conllu_inventory(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    key_counts: Counter = Counter()
    key_empty: Counter = Counter()
    value_counts: Dict[str, Counter] = defaultdict(Counter)
    doc_rows = []
    current = {}
    doc_index = 0

    def flush_doc():
        nonlocal current
        if current:
            doc_rows.append(current)
            current = {}

    with in_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# newdoc id = "):
                flush_doc()
                doc_index += 1
                current = {"xml_doc_index": str(doc_index), "newdoc id": line.split(" = ", 1)[1].strip()}
            if not line.startswith("# ") or " = " not in line:
                continue
            key, value = line[2:].split(" = ", 1)
            key = key.strip()
            value = value.strip()
            key_counts[key] += 1
            if not value:
                key_empty[key] += 1
            value_counts[key][value] += 1
            if key not in {"sent_id", "text", "newpar id"} and current is not None:
                current[key] = value
    flush_doc()

    all_keys = sorted(key_counts)
    rows = []
    for key in all_keys:
        rows.append({
            "key": key,
            "count": key_counts[key],
            "empty_count": key_empty[key],
            "unique_values": len(value_counts[key]),
            "sample_values": sample_values(value_counts[key], args.sample_values),
        })
    write_tsv(out_dir / "conllu_comment_key_inventory.tsv", rows,
              ["key", "count", "empty_count", "unique_values", "sample_values"])

    doc_fields = sorted({k for row in doc_rows for k in row.keys()})
    doc_fields = ["xml_doc_index", "newdoc id"] + [k for k in doc_fields if k not in {"xml_doc_index", "newdoc id"}]
    write_tsv(out_dir / "conllu_document_metadata.tsv", doc_rows, doc_fields)

    summary = {
        "input": str(in_path),
        "documents": len(doc_rows),
        "comment_keys": len(all_keys),
        "outputs": {
            "conllu_comment_key_inventory": str(out_dir / "conllu_comment_key_inventory.tsv"),
            "conllu_document_metadata": str(out_dir / "conllu_document_metadata.tsv"),
        },
    }
    (out_dir / "conllu_metadata_inventory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Manual Wikivir metadata QC: inventory, selective XML key normalization, CoNLL-U metadata inventory.")
    sub = p.add_subparsers(dest="cmd", required=True)

    inv = sub.add_parser("inventory-xml", help="Inventory all <doc> metadata keys and values in XML.")
    inv.add_argument("--input", required=True)
    inv.add_argument("--out-dir", required=True)
    inv.add_argument("--doc-tag", default="doc")
    inv.add_argument("--expected-keys", default=",".join(DEFAULT_EXPECTED_KEYS))
    inv.add_argument("--sample-values", type=int, default=12)
    inv.set_defaults(func=inventory_xml)

    app = sub.add_parser("apply-map", help="Apply a manually edited metadata key map to XML <doc> attributes.")
    app.add_argument("--input", required=True)
    app.add_argument("--key-map", required=True)
    app.add_argument("--output", required=True)
    app.add_argument("--doc-tag", default="doc")
    app.add_argument("--on-conflict", choices=["abort", "prefer-existing", "prefer-incoming", "combine"], default="abort")
    app.add_argument("--fixes", default="reports/qc/metadata_manual/metadata_fixes.tsv")
    app.add_argument("--conflicts", default="reports/qc/metadata_manual/metadata_conflicts.tsv")
    app.add_argument("--summary", default="reports/qc/metadata_manual/apply_map_summary.json")
    app.set_defaults(func=apply_map)

    ci = sub.add_parser("inventory-conllu", help="Inventory embedded metadata/comment keys in CoNLL-U.")
    ci.add_argument("--input", required=True)
    ci.add_argument("--out-dir", required=True)
    ci.add_argument("--sample-values", type=int, default=12)
    ci.set_defaults(func=conllu_inventory)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
