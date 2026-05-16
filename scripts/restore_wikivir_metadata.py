#!/usr/bin/env python3
"""Restore Wikivir document metadata into an annotated CoNLL-U file.

Use this when annotation accidentally stripped the original XML/CoNLL-U document
metadata comments. It injects metadata comments such as::

    # newdoc id = doc-001
    # title = Slovenja
    # author = Jovan Vesel Koseski
    # century = 19
    # genre = priložnostne pesmi

The script never changes token rows. It only edits/creates document-level comment
lines at the beginning of document blocks.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

PREFERRED_KEYS = [
    "title", "author", "century", "year", "date", "genre", "period", "publication",
    "source", "url", "language", "translator", "editor", "collection",
]
STRUCTURAL_KEYS = {
    "sent_id", "text", "newdoc", "newdoc_id", "newpar", "newpar_id", "par", "par_id",
    "paragraph_id", "global_columns", "global_metadata",
}


@dataclass
class AnnotatedDoc:
    doc_id: str
    first_block_index: int
    metadata: dict[str, str] = field(default_factory=dict)


def read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1250", "latin2", "latin1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_col(col: str) -> str:
    col = unicodedata.normalize("NFKC", str(col)).strip().lower()
    col = re.sub(r"[^\w]+", "_", col, flags=re.UNICODE).strip("_")
    return col


def clean_meta_key(key: str) -> str:
    key_norm = normalize_col(key)
    if key_norm in STRUCTURAL_KEYS:
        return ""
    for prefix in ("doc_", "document_", "newdoc_", "meta_"):
        if key_norm.startswith(prefix) and len(key_norm) > len(prefix):
            stripped = key_norm[len(prefix):]
            if stripped not in STRUCTURAL_KEYS:
                return stripped
    return key_norm


def parse_kv_comment(line: str) -> tuple[str, str] | None:
    if not line.startswith("#") or "=" not in line:
        return None
    body = line[1:].strip()
    key, value = body.split("=", 1)
    return key.strip(), value.strip()


def newdoc_id_from_comment(line: str) -> str | None:
    kv = parse_kv_comment(line)
    if not kv:
        return None
    key, value = kv
    if normalize_col(key) in {"newdoc", "newdoc_id"}:
        return value
    return None


def sanitize_value(value: object) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_meta_value(value: str) -> dict[str, str]:
    value = value.strip()
    if not value:
        return {}
    if value.startswith("{") and value.endswith("}"):
        try:
            obj = json.loads(value)
            if isinstance(obj, dict):
                return {clean_meta_key(k): sanitize_value(v) for k, v in obj.items() if clean_meta_key(k)}
        except Exception:
            pass
    parts = re.split(r"[|;]", value)
    out: dict[str, str] = {}
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            k = clean_meta_key(k)
            if k:
                out[k] = sanitize_value(v)
    return out


def split_conllu_blocks(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    stripped = text.strip("\n")
    if not stripped:
        return []
    return re.split(r"\n\s*\n", stripped)


def parse_annotated_docs(blocks: list[str], default_doc_id: str = "doc") -> list[AnnotatedDoc]:
    docs: list[AnnotatedDoc] = []
    current: AnnotatedDoc | None = None
    for i, block in enumerate(blocks):
        comments = [ln for ln in block.splitlines() if ln.startswith("#")]
        newdoc_id = None
        block_meta: dict[str, str] = {}
        sent_id = None
        for line in comments:
            kv = parse_kv_comment(line)
            if not kv:
                continue
            key, value = kv
            key_norm = normalize_col(key)
            if key_norm in {"newdoc", "newdoc_id"}:
                newdoc_id = value
            elif key_norm == "sent_id":
                sent_id = value
            elif key_norm in {"doc_meta", "metadata", "meta", "newdoc_meta", "document_meta"}:
                block_meta.update(parse_meta_value(value))
            else:
                meta_key = clean_meta_key(key)
                if meta_key:
                    block_meta[meta_key] = sanitize_value(value)
        if newdoc_id is not None:
            current = AnnotatedDoc(doc_id=newdoc_id, first_block_index=i, metadata=block_meta)
            docs.append(current)
        elif current is None:
            inferred = sent_id.split(".", 1)[0] if sent_id and "." in sent_id else default_doc_id
            current = AnnotatedDoc(doc_id=inferred, first_block_index=i, metadata=block_meta)
            docs.append(current)
        elif block_meta:
            current.metadata.update(block_meta)
    return docs


def strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def is_legal_xml_char(ch: str) -> bool:
    """Return True for characters allowed by XML 1.0.

    Wikivir dumps sometimes contain literal control bytes. Strict XML parsers abort
    on those even though the <doc ...> start tags we need are perfectly usable.
    """
    if ch in "\t\n\r":
        return True
    code = ord(ch)
    return code >= 0x20 and code not in {0xFFFE, 0xFFFF}


def strip_illegal_xml_chars(text: str) -> str:
    return "".join(ch if is_legal_xml_char(ch) else " " for ch in text)


ATTR_RE = re.compile(
    r"""([A-Za-z_:\-][\w:.\-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""",
    re.UNICODE | re.DOTALL,
)


def xml_attrs_to_metadata(attrs: Mapping[str, str], ordinal: int) -> dict[str, str]:
    """Normalize XML attributes to the document metadata schema."""
    raw_id = (
        attrs.get("id")
        or attrs.get("xml:id")
        or attrs.get("{http://www.w3.org/XML/1998/namespace}id")
        or attrs.get("doc_id")
        or attrs.get("document_id")
        or attrs.get("wikivir_id")
        or attrs.get("wid")
    )
    doc_id = sanitize_value(raw_id) if raw_id else f"doc-{ordinal:06d}"
    meta: dict[str, str] = {"doc_id": doc_id}
    for key, value in attrs.items():
        # id/xml:id identifies the document; do not duplicate it as # id = ...
        if key in {"id", "xml:id", "{http://www.w3.org/XML/1998/namespace}id", "doc_id", "document_id"}:
            continue
        meta_key = clean_meta_key(key)
        if not meta_key or meta_key in {"id", "xml_id"}:
            continue
        clean_value = sanitize_value(html.unescape(str(value)))
        if clean_value:
            meta[meta_key] = clean_value
    return meta


def load_xml_metadata_strict(path: Path, doc_tag: str = "doc") -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    wanted = doc_tag.split(":", 1)[-1]
    for event, elem in ET.iterparse(path, events=("start",)):
        if strip_ns(elem.tag) != wanted:
            continue
        out.append(xml_attrs_to_metadata(dict(elem.attrib), len(out) + 1))
        elem.clear()
    return out


def parse_lenient_start_tag_attrs(tag_text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in ATTR_RE.finditer(tag_text):
        key = match.group(1)
        value = next((g for g in match.groups()[1:] if g is not None), "")
        attrs[key] = html.unescape(strip_illegal_xml_chars(value))
    return attrs


def load_xml_metadata_lenient(path: Path, doc_tag: str = "doc") -> list[dict[str, str]]:
    """Extract <doc ...> attributes without requiring well-formed XML.

    This is intentionally conservative: it does not parse the document tree or
    text content. It scans only opening start tags such as <doc title="...">,
    which is enough to recover Wikivir document metadata after annotation.
    """
    text = strip_illegal_xml_chars(read_text(path))
    wanted = re.escape(doc_tag.split(":", 1)[-1])
    # Accept <doc ...> and namespaced <tei:doc ...>.  Do not match </doc>.
    tag_re = re.compile(r"<\s*(?:[A-Za-z_][\w.\-]*:)?" + wanted + r"\b[^>]*>", re.IGNORECASE | re.DOTALL)
    out: list[dict[str, str]] = []
    for match in tag_re.finditer(text):
        tag_text = match.group(0)
        attrs = parse_lenient_start_tag_attrs(tag_text)
        out.append(xml_attrs_to_metadata(attrs, len(out) + 1))
    return out


def load_xml_metadata(path: Path, doc_tag: str = "doc", parse_mode: str = "auto") -> list[dict[str, str]]:
    """Load document metadata from XML.

    parse_mode:
      - strict: require well-formed XML and fail on XML parser errors
      - recover: always use the tolerant start-tag scanner
      - auto: try strict parsing, then fall back to tolerant scanning
    """
    if parse_mode not in {"auto", "strict", "recover"}:
        raise ValueError(f"Unknown XML parse mode: {parse_mode}")

    if parse_mode == "recover":
        rows = load_xml_metadata_lenient(path, doc_tag)
        print(f"Recovered {len(rows):,} <{doc_tag}> metadata rows with the tolerant XML scanner.", file=sys.stderr)
        return rows

    try:
        rows = load_xml_metadata_strict(path, doc_tag)
        print(f"Read {len(rows):,} <{doc_tag}> metadata rows with the strict XML parser.", file=sys.stderr)
        return rows
    except ET.ParseError as exc:
        if parse_mode == "strict":
            raise ValueError(
                f"XML source is not well-formed at line {getattr(exc, 'position', ('?', '?'))[0]}, "
                f"column {getattr(exc, 'position', ('?', '?'))[1]}: {exc}. "
                "Run again with --xml-parse-mode recover or omit the flag for auto recovery."
            ) from exc
        print(
            f"Warning: strict XML parsing failed for {path}: {exc}. "
            "Falling back to tolerant <doc ...> metadata scanning.",
            file=sys.stderr,
        )
        rows = load_xml_metadata_lenient(path, doc_tag)
        print(f"Recovered {len(rows):,} <{doc_tag}> metadata rows with the tolerant XML scanner.", file=sys.stderr)
        return rows


def load_conllu_metadata(path: Path) -> list[dict[str, str]]:
    blocks = split_conllu_blocks(read_text(path))
    docs = parse_annotated_docs(blocks, path.stem)
    rows = []
    for doc in docs:
        rows.append({"doc_id": doc.doc_id, **doc.metadata})
    return rows


def read_table_auto(path: Path) -> list[dict[str, str]]:
    sample = read_text(path)[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        sep = dialect.delimiter
    except Exception:
        sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        return [{clean_meta_key(k): sanitize_value(v) for k, v in row.items() if clean_meta_key(k) and sanitize_value(v)} for row in reader]


def load_source_metadata(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.source_xml:
        return load_xml_metadata(args.source_xml, args.xml_doc_tag, args.xml_parse_mode)
    if args.source_conllu:
        return load_conllu_metadata(args.source_conllu)
    if args.source_table:
        return read_table_auto(args.source_table)
    raise ValueError("Pass one of --source-xml, --source-conllu, or --source-table.")


def count_duplicates(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}


def safe_doc_id(value: object, fallback: str) -> str:
    s = sanitize_value(value)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w.:-]+", "_", s, flags=re.UNICODE).strip("_")
    return s or fallback


def make_unique(value: str, used: set[str], ordinal: int) -> str:
    base = safe_doc_id(value, f"doc-{ordinal:06d}")
    if base not in used:
        used.add(base)
        return base
    i = 2
    while True:
        candidate = f"{base}__{i:04d}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def choose_alignment(annotated_docs: list[AnnotatedDoc], source_rows: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    """Return source metadata rows aligned 1:1 with annotated documents.

    Important: this must return a list, not a dict keyed by annotated doc_id.
    Wikivir annotation sometimes produces repeated/generic # newdoc ids; a dict
    would silently overwrite earlier metadata rows and inject the same metadata
    into many documents.
    """
    ann_ids = [d.doc_id for d in annotated_docs]
    source_by_id: dict[str, dict[str, str]] = {}
    duplicate_source_ids = count_duplicates(row.get("doc_id", "") for row in source_rows if row.get("doc_id"))
    if duplicate_source_ids:
        print(
            f"Warning: source metadata has {len(duplicate_source_ids)} duplicate doc_id value(s); "
            "by-id alignment will keep the first occurrence for each duplicated ID.",
            file=sys.stderr,
        )
    for row in source_rows:
        did = row.get("doc_id", "")
        if did and did not in source_by_id:
            source_by_id[did] = row
    id_matches = sum(1 for did in ann_ids if did in source_by_id)
    ann_unique = len(set(ann_ids)) == len(ann_ids)

    if mode == "by-id" or (mode == "auto" and ann_unique and id_matches > 0 and id_matches >= max(1, int(0.8 * len(annotated_docs)))):
        if id_matches == 0:
            raise ValueError("--match-mode by-id selected, but no source doc_id values match annotated # newdoc ids.")
        return [dict(source_by_id.get(doc.doc_id, {})) for doc in annotated_docs]

    if mode == "by-order" or mode == "auto":
        if len(source_rows) != len(annotated_docs):
            raise ValueError(
                f"Cannot safely align by order: annotated CoNLL-U has {len(annotated_docs)} documents, "
                f"source metadata has {len(source_rows)} rows. Add matching IDs or pass a corrected source."
            )
        return [dict(row) for row in source_rows]

    raise ValueError(f"Unknown match mode: {mode}")


def should_use_source_ids(annotated_docs: list[AnnotatedDoc], source_rows: list[dict[str, str]], mode: str) -> bool:
    if mode == "source":
        return True
    if mode in {"keep", "order"}:
        return False
    ann_ids = [doc.doc_id for doc in annotated_docs]
    if len(set(ann_ids)) < len(ann_ids):
        return True
    generic = {"doc", "document", "corpus", "wikivir", "wikivir-classla", "wikivir_classla"}
    if len(set(ann_ids)) <= max(1, len(ann_ids) // 100) or any(str(x).lower() in generic for x in ann_ids):
        return True
    # Good annotated IDs are usually the most stable keys for downstream work.
    # Use XML/source IDs explicitly with --doc-id-mode source.
    return False


def choose_output_doc_ids(
    annotated_docs: list[AnnotatedDoc],
    aligned_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
    mode: str = "auto",
    prefix: str = "wikivir-doc",
) -> list[str]:
    """Choose stable unique # newdoc ids for the restored output."""
    used: set[str] = set()
    out: list[str] = []
    use_source = should_use_source_ids(annotated_docs, source_rows, mode)
    for i, (doc, row) in enumerate(zip(annotated_docs, aligned_rows), start=1):
        if mode == "order":
            candidate = f"{prefix}-{i:06d}"
        elif use_source:
            candidate = row.get("doc_id") or f"{prefix}-{i:06d}"
        else:
            candidate = doc.doc_id or row.get("doc_id") or f"{prefix}-{i:06d}"
        out.append(make_unique(candidate, used, i))
    return out


def is_metadata_comment(line: str) -> bool:
    kv = parse_kv_comment(line)
    if not kv:
        return False
    key, _ = kv
    key_norm = normalize_col(key)
    if key_norm in STRUCTURAL_KEYS:
        return False
    if key_norm in {"doc_meta", "metadata", "meta", "newdoc_meta", "document_meta"}:
        return True
    return bool(clean_meta_key(key))


def comment_lines_for_metadata(meta: Mapping[str, str]) -> list[str]:
    ordered = [k for k in PREFERRED_KEYS if k in meta and sanitize_value(meta[k])]
    ordered += sorted(k for k in meta if k not in ordered and k != "doc_id" and sanitize_value(meta[k]))
    return [f"# {key} = {sanitize_value(meta[key])}" for key in ordered]


def inject_metadata_into_block(
    block: str,
    doc_id: str,
    metadata: Mapping[str, str],
    replace_existing: bool,
    ensure_newdoc: bool,
    rewrite_newdoc_id: bool = True,
) -> str:
    lines = block.splitlines()
    if replace_existing:
        lines = [ln for ln in lines if not is_metadata_comment(ln)]

    has_newdoc = False
    for i, line in enumerate(lines):
        if newdoc_id_from_comment(line) is not None:
            has_newdoc = True
            if rewrite_newdoc_id:
                lines[i] = f"# newdoc id = {doc_id}"
            break
    if ensure_newdoc and not has_newdoc:
        lines.insert(0, f"# newdoc id = {doc_id}")

    insert_at = 0
    for i, line in enumerate(lines):
        if newdoc_id_from_comment(line) is not None:
            insert_at = i + 1
            break
    meta_lines = comment_lines_for_metadata(metadata)
    return "\n".join(lines[:insert_at] + meta_lines + lines[insert_at:])


def write_metadata_table(path: Path, annotated_docs: list[AnnotatedDoc], output_doc_ids: list[str], aligned_rows: list[dict[str, str]]) -> None:
    all_keys = set()
    rows = []
    for doc, output_id, meta in zip(annotated_docs, output_doc_ids, aligned_rows):
        source_id = meta.get("doc_id", "")
        row = {"doc_id": output_id, "annotated_doc_id": doc.doc_id, "source_doc_id": source_id, **{k: v for k, v in meta.items() if k != "doc_id"}}
        rows.append(row)
        all_keys.update(row)
    fieldnames = ["doc_id", "annotated_doc_id", "source_doc_id"] + [k for k in PREFERRED_KEYS if k in all_keys and k not in {"doc_id", "annotated_doc_id", "source_doc_id"}] + sorted(k for k in all_keys if k not in set(PREFERRED_KEYS) | {"doc_id", "annotated_doc_id", "source_doc_id"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Restore XML/original-CoNLL-U metadata comments into an annotated Wikivir CoNLL-U file.")
    p.add_argument("--annotated", required=True, type=Path, help="Annotated CoNLL-U whose token rows should be kept.")
    p.add_argument("--output", required=True, type=Path, help="Output CoNLL-U with restored metadata comments.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source-xml", type=Path, help="Original Wikivir XML with <doc ...> attributes.")
    src.add_argument("--source-conllu", type=Path, help="Original CoNLL-U that still contains metadata comments.")
    src.add_argument("--source-table", type=Path, help="Metadata TSV/CSV, used only as a recovery source.")
    p.add_argument("--xml-doc-tag", default="doc", help="XML element name containing document metadata attributes.")
    p.add_argument("--xml-parse-mode", choices=["auto", "strict", "recover"], default="auto", help="How to parse --source-xml. auto tries strict XML first and falls back to a tolerant <doc ...> scanner; recover always uses the scanner.")
    p.add_argument("--match-mode", choices=["auto", "by-id", "by-order"], default="auto", help="How to align source metadata to annotated documents.")
    p.add_argument("--replace-existing", action="store_true", help="Remove existing metadata comments before inserting recovered ones.")
    p.add_argument("--doc-id-mode", choices=["auto", "keep", "source", "order"], default="auto", help="How to choose output # newdoc ids. auto uses source/XML IDs when annotated IDs are duplicated or generic; order writes prefix-000001 IDs.")
    p.add_argument("--id-prefix", default="wikivir-doc", help="Prefix for generated output document IDs when --doc-id-mode order or a source ID is missing.")
    p.add_argument("--no-rewrite-newdoc-id", action="store_true", help="Keep existing # newdoc id comments exactly as they are. Not recommended when IDs are duplicated.")
    p.add_argument("--no-ensure-newdoc", action="store_true", help="Do not add # newdoc id when missing.")
    p.add_argument("--metadata-output", type=Path, help="Optional TSV of the metadata that was injected.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    annotated_text = read_text(args.annotated)
    blocks = split_conllu_blocks(annotated_text)
    if not blocks:
        raise ValueError(f"No CoNLL-U blocks found in {args.annotated}")
    annotated_docs = parse_annotated_docs(blocks, args.annotated.stem)
    source_rows = load_source_metadata(args)
    if not source_rows:
        raise ValueError("No source metadata rows found.")

    aligned_rows = choose_alignment(annotated_docs, source_rows, args.match_mode)
    output_doc_ids = choose_output_doc_ids(annotated_docs, aligned_rows, source_rows, args.doc_id_mode, args.id_prefix)

    annotated_duplicates = count_duplicates(doc.doc_id for doc in annotated_docs)
    if annotated_duplicates:
        print(
            f"Warning: annotated CoNLL-U contains {len(annotated_duplicates)} duplicated # newdoc id value(s). "
            "The restored output will use unique document IDs unless --no-rewrite-newdoc-id was passed.",
            file=sys.stderr,
        )
    if len(set(output_doc_ids)) != len(output_doc_ids):
        raise ValueError("Internal error: output document IDs are not unique after normalization.")

    first_block_to_doc_index = {doc.first_block_index: idx for idx, doc in enumerate(annotated_docs)}
    restored_blocks: list[str] = []
    for i, block in enumerate(blocks):
        doc_index = first_block_to_doc_index.get(i)
        if doc_index is None:
            restored_blocks.append(block)
            continue
        meta = {k: v for k, v in aligned_rows[doc_index].items() if k != "doc_id"}
        # Keep traceability when we had to replace duplicated/generic annotated IDs.
        if annotated_docs[doc_index].doc_id != output_doc_ids[doc_index]:
            meta.setdefault("annotated_doc_id", annotated_docs[doc_index].doc_id)
        source_id = aligned_rows[doc_index].get("doc_id", "")
        if source_id and source_id != output_doc_ids[doc_index]:
            meta.setdefault("source_doc_id", source_id)
        restored_blocks.append(
            inject_metadata_into_block(
                block,
                doc_id=output_doc_ids[doc_index],
                metadata=meta,
                replace_existing=args.replace_existing,
                ensure_newdoc=not args.no_ensure_newdoc,
                rewrite_newdoc_id=not args.no_rewrite_newdoc_id,
            )
        )

    write_text(args.output, "\n\n".join(restored_blocks) + "\n")
    if args.metadata_output:
        write_metadata_table(args.metadata_output, annotated_docs, output_doc_ids, aligned_rows)

    nonempty_docs = sum(1 for row in aligned_rows if any(str(v).strip() for k, v in row.items() if k != "doc_id"))
    print(f"Restored metadata for {nonempty_docs:,}/{len(annotated_docs):,} documents into {args.output}", file=sys.stderr)
    print(f"Output has {len(set(output_doc_ids)):,} unique # newdoc ids for {len(output_doc_ids):,} documents.", file=sys.stderr)
    if nonempty_docs < len(annotated_docs):
        print("Warning: some documents received no metadata. Check ID/order alignment.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
