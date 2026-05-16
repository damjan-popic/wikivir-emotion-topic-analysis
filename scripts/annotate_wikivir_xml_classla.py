#!/usr/bin/env python3
"""Annotate a well-formed Wikivir XML corpus with the full CLASSLA stack.

This script is meant to be the *clean* way to produce the annotated Wikivir
CoNLL-U file used by the emotion/topic analysis scripts.  It reads metadata from
XML ``<doc ...>`` attributes and writes that metadata directly into the CoNLL-U
comments, e.g.::

    # newdoc id = wikivir-000001
    # title = Slovenja
    # author = Jovan Vesel Koseski
    # century = 19
    # genre = priložnostne pesmi

No post-hoc metadata restoration should be needed if this script is used.

Typical usage::

    python scripts/annotate_wikivir_xml_classla.py \
      --input data/raw/wikivir.xml \
      --output data/annotated/wikivir-classla.conllu \
      --metadata-output data/metadata/wikivir-metadata.tsv \
      --summary reports/wikivir-classla-summary.json \
      --errors reports/wikivir-classla-errors.tsv \
      --validation-report reports/wikivir-classla-validation.tsv \
      --models-dir ~/classla_resources_shared \
      --download-models \
      --overwrite

The implementation is intentionally defensive: it streams XML documents, keeps
model resources shareable, splits long documents into manageable blocks, writes
validation/audit files, and refuses to overwrite existing output unless asked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

TOKEN_COLUMNS: Tuple[str, ...] = (
    "ID",
    "FORM",
    "LEMMA",
    "UPOS",
    "XPOS",
    "FEATS",
    "HEAD",
    "DEPREL",
    "DEPS",
    "MISC",
)

DEFAULT_PROCESSORS = "tokenize,ner,pos,lemma,depparse"
XML_NS = "{http://www.w3.org/XML/1998/namespace}"
WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
BAD_COMMENT_KEY_RE = re.compile(r"[^0-9A-Za-z_\-.]+")
BAD_ID_RE = re.compile(r"[^0-9A-Za-z_.:-]+")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
SENTENCE_COMMENT_RE = re.compile(r"^#\s*([^=]+?)\s*=\s*(.*)$")
RECOVER_DOC_TAG_RE = re.compile(r"<doc\b(?P<attrs>[^>]*)>", flags=re.IGNORECASE | re.DOTALL)
RECOVER_ATTR_RE = re.compile(
    r"(?P<key>[A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    flags=re.DOTALL,
)


@dataclass
class XmlDocument:
    index: int
    doc_id: str
    text: str
    metadata: Dict[str, str]
    raw_attrs: Dict[str, str]
    source: str


@dataclass
class ConlluSentence:
    comments: List[str] = field(default_factory=list)
    rows: List[str] = field(default_factory=list)

    def comment_value(self, key: str) -> Optional[str]:
        prefix = f"# {key} = "
        for line in self.comments:
            if line.startswith(prefix):
                return line[len(prefix):].strip()
        return None

    @property
    def text(self) -> str:
        return self.comment_value("text") or reconstruct_text_from_rows(self.rows)

    @property
    def word_token_count(self) -> int:
        count = 0
        for row in self.rows:
            if not row or row.startswith("#"):
                continue
            token_id = row.split("\t", 1)[0]
            if token_id.isdigit():
                count += 1
        return count

    def to_conllu(self) -> str:
        return "\n".join(self.comments + self.rows)


@dataclass
class RunStats:
    docs_seen: int = 0
    docs_selected: int = 0
    docs_with_text: int = 0
    docs_written: int = 0
    docs_failed: int = 0
    blocks_attempted: int = 0
    blocks_written: int = 0
    blocks_failed: int = 0
    sentences_written: int = 0
    tokens_written: int = 0
    chars_total: int = 0
    seconds: float = 0.0


class ClasslaImportError(SystemExit):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def clean_text_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_body_text(text: str) -> str:
    text = CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Keep paragraph breaks, but remove trailing whitespace on individual lines.
    lines = [line.strip() for line in text.split("\n")]
    # Trim only at document boundaries; internal blank lines matter for paragraph IDs.
    return "\n".join(lines).strip()


def strip_namespace(value: str) -> str:
    if value.startswith(XML_NS):
        return "xml_" + value[len(XML_NS):]
    if "}" in value:
        return value.rsplit("}", 1)[-1]
    return value


def sanitize_comment_key(key: str) -> str:
    key = strip_namespace(str(key)).strip()
    key = key.replace(":", "_").replace(" ", "_")
    key = BAD_COMMENT_KEY_RE.sub("_", key)
    key = re.sub(r"_+", "_", key).strip("_").lower()
    return key or "meta"


def sanitize_doc_id(value: str, fallback: str = "doc") -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).strip()
    value = WHITESPACE_RE.sub("-", value)
    value = BAD_ID_RE.sub("-", value)
    value = value.strip("-_.:")
    return value or fallback


def safe_comment_line(key: str, value: object) -> Optional[str]:
    clean_key = sanitize_comment_key(key)
    clean_value = clean_text_value(value)
    if clean_value == "":
        return None
    return f"# {clean_key} = {clean_value}"


def parse_list_arg(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def unique_id(base: str, seen: Dict[str, int]) -> str:
    base = sanitize_doc_id(base)
    seen[base] += 1
    if seen[base] == 1:
        return base
    return f"{base}-{seen[base]:03d}"


def choose_base_doc_id(
    *,
    attrs: Dict[str, str],
    index: int,
    id_attr: Optional[str],
    id_template: str,
    prefix: str,
) -> str:
    # Explicit CLI attribute wins.
    if id_attr:
        key = sanitize_comment_key(id_attr)
        if attrs.get(key):
            return attrs[key]

    # Common XML identity attributes next.
    for candidate in ("xml_id", "id", "doc_id", "document_id", "wikivir_id", "slug"):
        if attrs.get(candidate):
            return attrs[candidate]

    # Otherwise use a stable order-based ID. This is safer than title-based IDs,
    # because title/author can duplicate and can contain awkward punctuation.
    try:
        return id_template.format(index=index, prefix=prefix, **attrs)
    except Exception:
        return f"{prefix}-{index:06d}"


def element_text(elem: ET.Element) -> str:
    return normalize_body_text("".join(elem.itertext()))


def iter_xml_documents_strict(
    path: Path,
    *,
    doc_tag: str,
    id_attr: Optional[str],
    id_template: str,
    id_prefix: str,
) -> Iterator[XmlDocument]:
    seen: Dict[str, int] = defaultdict(int)
    index = 0
    for _event, elem in ET.iterparse(path, events=("end",)):
        if strip_namespace(elem.tag) != doc_tag:
            continue
        index += 1
        attrs: Dict[str, str] = {}
        raw_attrs: Dict[str, str] = {}
        for raw_key, raw_value in elem.attrib.items():
            key = sanitize_comment_key(raw_key)
            value = clean_text_value(raw_value)
            attrs[key] = value
            raw_attrs[strip_namespace(raw_key)] = value

        base = choose_base_doc_id(
            attrs=attrs,
            index=index,
            id_attr=id_attr,
            id_template=id_template,
            prefix=id_prefix,
        )
        doc_id = unique_id(base, seen)
        metadata = dict(attrs)
        metadata.setdefault("xml_doc_index", str(index))
        metadata.setdefault("source_xml", path.name)
        if doc_id != sanitize_doc_id(base):
            metadata.setdefault("doc_id_disambiguated_from", sanitize_doc_id(base))

        yield XmlDocument(
            index=index,
            doc_id=doc_id,
            text=element_text(elem),
            metadata=metadata,
            raw_attrs=raw_attrs,
            source=str(path),
        )
        elem.clear()


def unescape_xmlish(value: str) -> str:
    # For recover mode only. Keep lightweight; strict mode should be used for real XML.
    import html

    return clean_text_value(html.unescape(value))


def iter_xml_documents_recover(
    path: Path,
    *,
    doc_tag: str,
    id_attr: Optional[str],
    id_template: str,
    id_prefix: str,
) -> Iterator[XmlDocument]:
    if doc_tag != "doc":
        raise ValueError("recover mode currently supports <doc ...> tags only; use strict mode for custom tags")
    text = read_text(path)
    seen: Dict[str, int] = defaultdict(int)
    matches = list(RECOVER_DOC_TAG_RE.finditer(text))
    for index, match in enumerate(matches, start=1):
        attrs: Dict[str, str] = {}
        raw_attrs: Dict[str, str] = {}
        attr_text = match.group("attrs")
        for attr_match in RECOVER_ATTR_RE.finditer(attr_text):
            raw_key = attr_match.group("key")
            key = sanitize_comment_key(raw_key)
            value = unescape_xmlish(attr_match.group("value"))
            attrs[key] = value
            raw_attrs[raw_key] = value

        body_start = match.end()
        next_start = matches[index].start() if index < len(matches) else len(text)
        body = text[body_start:next_start]
        body = re.sub(r"</doc>.*$", "", body, flags=re.DOTALL | re.IGNORECASE)
        body = re.sub(r"<[^>]+>", " ", body)
        body = unescape_xmlish(body)
        body = normalize_body_text(body)

        base = choose_base_doc_id(
            attrs=attrs,
            index=index,
            id_attr=id_attr,
            id_template=id_template,
            prefix=id_prefix,
        )
        doc_id = unique_id(base, seen)
        metadata = dict(attrs)
        metadata.setdefault("xml_doc_index", str(index))
        metadata.setdefault("source_xml", path.name)
        metadata.setdefault("xml_parse_mode", "recover")
        yield XmlDocument(index=index, doc_id=doc_id, text=body, metadata=metadata, raw_attrs=raw_attrs, source=str(path))


def iter_xml_documents(args: argparse.Namespace) -> Iterator[XmlDocument]:
    kwargs = {
        "doc_tag": args.xml_doc_tag,
        "id_attr": args.xml_id_attr,
        "id_template": args.doc_id_template,
        "id_prefix": args.doc_id_prefix,
    }
    if args.xml_parse_mode == "recover":
        yield from iter_xml_documents_recover(args.input, **kwargs)
    else:
        yield from iter_xml_documents_strict(args.input, **kwargs)


def split_into_blocks(text: str, mode: str) -> List[str]:
    text = normalize_body_text(text)
    if not text:
        return []
    if mode == "none":
        return [text]
    if mode == "blanklines":
        return [part.strip() for part in PARAGRAPH_SPLIT_RE.split(text) if part.strip()]
    if mode == "lines":
        return [line.strip() for line in text.splitlines() if line.strip()]
    raise ValueError(f"Unsupported block mode: {mode}")


def split_overlong_block(block: str, max_chars: int) -> List[str]:
    block = block.strip()
    if not block:
        return []
    if max_chars <= 0 or len(block) <= max_chars:
        return [block]

    # Prefer line boundaries if there are several lines.
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) > 1:
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for line in lines:
            addition = len(line) + (1 if current else 0)
            if current and current_len + addition > max_chars:
                chunks.append("\n".join(current))
                current = [line]
                current_len = len(line)
            else:
                current.append(line)
                current_len += addition
        if current:
            chunks.append("\n".join(current))
        return chunks

    # Fall back to whitespace chunks.
    words = block.split()
    chunks = []
    current_words: List[str] = []
    current_len = 0
    for word in words:
        addition = len(word) + (1 if current_words else 0)
        if current_words and current_len + addition > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_len = len(word)
        else:
            current_words.append(word)
            current_len += addition
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def make_blocks(text: str, mode: str, max_chars: int) -> List[str]:
    blocks: List[str] = []
    for block in split_into_blocks(text, mode):
        blocks.extend(split_overlong_block(block, max_chars=max_chars))
    return [block for block in blocks if block.strip()]


def parse_conllu_block(conllu_text: str) -> List[ConlluSentence]:
    sentences: List[ConlluSentence] = []
    for raw_block in re.split(r"\n\s*\n", conllu_text.strip()):
        if not raw_block.strip():
            continue
        comments: List[str] = []
        rows: List[str] = []
        for line in raw_block.splitlines():
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                comments.append(line)
            else:
                parts = line.split("\t")
                if len(parts) != 10:
                    # Keep the line, but downstream validation will flag it.
                    rows.append(line)
                else:
                    rows.append(line)
        if comments or rows:
            sentences.append(ConlluSentence(comments=comments, rows=rows))
    return sentences


def reconstruct_text_from_rows(rows: Sequence[str]) -> str:
    pieces: List[str] = []
    for row in rows:
        parts = row.split("\t")
        if len(parts) != 10:
            continue
        token_id, form, *_middle, misc = parts
        # Skip multiword range / empty nodes when reconstructing from word lines.
        if "-" in token_id or "." in token_id:
            continue
        pieces.append(form)
        if "SpaceAfter=No" not in misc.split("|"):
            pieces.append(" ")
    return "".join(pieces).strip()


def metadata_comments(document: XmlDocument, metadata_keys: Optional[Sequence[str]] = None) -> List[str]:
    comments = [f"# newdoc id = {document.doc_id}"]
    comments.append(f"# source_xml = {clean_text_value(Path(document.source).name)}")
    comments.append(f"# xml_doc_index = {document.index}")

    keys = list(metadata_keys) if metadata_keys else sorted(document.metadata)
    already = {"source_xml", "xml_doc_index"}
    for key in keys:
        clean_key = sanitize_comment_key(key)
        if clean_key in already:
            continue
        if clean_key not in document.metadata:
            continue
        line = safe_comment_line(clean_key, document.metadata.get(clean_key, ""))
        if line:
            comments.append(line)
            already.add(clean_key)

    # Add any remaining metadata not explicitly listed.
    for key in sorted(document.metadata):
        clean_key = sanitize_comment_key(key)
        if clean_key in already:
            continue
        line = safe_comment_line(clean_key, document.metadata.get(key, ""))
        if line:
            comments.append(line)
            already.add(clean_key)
    return comments


def sentence_comments(
    *,
    document: XmlDocument,
    sentence: ConlluSentence,
    sentence_number: int,
    block_number: int,
    first_sentence_in_doc: bool,
    first_sentence_in_block: bool,
    block_mode: str,
    metadata_keys: Optional[Sequence[str]],
) -> List[str]:
    comments: List[str] = []
    if first_sentence_in_doc:
        comments.extend(metadata_comments(document, metadata_keys=metadata_keys))
    if block_mode != "none" and first_sentence_in_block:
        comments.append(f"# newpar id = {document.doc_id}.p{block_number:06d}")
    comments.append(f"# sent_id = {document.doc_id}.s{sentence_number:06d}")
    comments.append(f"# text = {clean_text_value(sentence.text)}")
    return comments


def write_tsv(path: Path, rows: Sequence[Dict[str, Any]], header: Sequence[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            safe_row = {key: clean_text_value(row.get(key, "")) for key in header}
            writer.writerow(safe_row)


def ensure_classla():
    try:
        import classla  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ClasslaImportError(
            "The 'classla' package is not installed. Install annotation dependencies with: "
            "pip install -r requirements-annotation.txt"
        ) from exc
    return classla


def signature_names(func: Callable[..., Any]) -> set[str]:
    try:
        return set(inspect.signature(func).parameters)
    except Exception:
        return set()


def add_models_dir_argument(func: Callable[..., Any], kwargs: Dict[str, Any], models_dir: Optional[Path]) -> Dict[str, Any]:
    out = dict(kwargs)
    if models_dir is None:
        return out
    normalized = str(Path(models_dir).expanduser().resolve())
    Path(normalized).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CLASSLA_RESOURCES_DIR", normalized)
    os.environ.setdefault("STANZA_RESOURCES_DIR", normalized)
    names = signature_names(func)
    for candidate in ("dir", "model_dir", "resources_dir"):
        if candidate in names:
            out[candidate] = normalized
            return out
    out.setdefault("dir", normalized)
    return out


def get_classla_version(classla: Any) -> str:
    for attr in ("__version__", "VERSION"):
        if hasattr(classla, attr):
            try:
                return str(getattr(classla, attr))
            except Exception:
                pass
    return "unknown"


def download_models(*, lang: str, classla_type: str, models_dir: Optional[Path]) -> None:
    classla = ensure_classla()
    kwargs: Dict[str, Any] = {"lang": lang}
    if classla_type and classla_type != "standard":
        kwargs["type"] = classla_type
    kwargs = add_models_dir_argument(classla.download, kwargs, models_dir)
    eprint(f"Downloading/checking CLASSLA models: lang={lang}, type={classla_type}, models_dir={models_dir or 'default'}")
    classla.download(**kwargs)


def build_pipeline(args: argparse.Namespace):
    classla = ensure_classla()
    if args.download_models:
        download_models(lang=args.lang, classla_type=args.classla_type, models_dir=args.models_dir)
    kwargs: Dict[str, Any] = {"processors": args.processors}
    if args.classla_type and args.classla_type != "standard":
        kwargs["type"] = args.classla_type
    if args.processing_mode == "pretokenized":
        kwargs["tokenize_pretokenized"] = True
    kwargs = add_models_dir_argument(classla.Pipeline, kwargs, args.models_dir)
    eprint(f"Initializing CLASSLA pipeline: lang={args.lang}, type={args.classla_type}, processors={args.processors}")
    return classla.Pipeline(args.lang, **kwargs)


def validate_output_file(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    doc_ids: Counter[str] = Counter()
    current_doc = ""
    sent_count = 0
    token_count = 0
    bad_rows = 0
    root_issues = 0
    sent_ids: Counter[str] = Counter()

    current_sent_rows: List[str] = []
    current_sent_id = ""

    def finish_sentence() -> None:
        nonlocal root_issues, token_count, bad_rows, current_sent_rows
        if not current_sent_rows:
            return
        roots = 0
        for line in current_sent_rows:
            parts = line.split("\t")
            if len(parts) != 10:
                bad_rows += 1
                continue
            token_id = parts[0]
            if token_id.isdigit():
                token_count += 1
                if parts[6] == "0":
                    roots += 1
        if roots != 1:
            root_issues += 1
        current_sent_rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                finish_sentence()
                current_sent_id = ""
                continue
            if line.startswith("# newdoc id = "):
                current_doc = line.split("=", 1)[1].strip()
                doc_ids[current_doc] += 1
            elif line.startswith("# sent_id = "):
                current_sent_id = line.split("=", 1)[1].strip()
                sent_ids[current_sent_id] += 1
                sent_count += 1
            elif not line.startswith("#"):
                current_sent_rows.append(line)
        finish_sentence()

    rows.append({"check": "documents", "value": sum(doc_ids.values()), "status": "ok" if doc_ids else "fail"})
    duplicate_docs = sum(1 for count in doc_ids.values() if count > 1)
    rows.append({"check": "duplicate_doc_ids", "value": duplicate_docs, "status": "ok" if duplicate_docs == 0 else "fail"})
    duplicate_sents = sum(1 for count in sent_ids.values() if count > 1)
    rows.append({"check": "duplicate_sent_ids", "value": duplicate_sents, "status": "ok" if duplicate_sents == 0 else "fail"})
    rows.append({"check": "sentences", "value": sent_count, "status": "ok" if sent_count else "fail"})
    rows.append({"check": "word_tokens", "value": token_count, "status": "ok" if token_count else "fail"})
    rows.append({"check": "bad_token_rows", "value": bad_rows, "status": "ok" if bad_rows == 0 else "fail"})
    rows.append({"check": "sentences_without_one_root", "value": root_issues, "status": "ok" if root_issues == 0 else "warn"})
    return rows


def collect_xml_metadata_preview(args: argparse.Namespace, limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Counter[str], RunStats]:
    rows: List[Dict[str, Any]] = []
    key_counter: Counter[str] = Counter()
    stats = RunStats()
    for doc in iter_xml_documents(args):
        stats.docs_seen += 1
        if args.start_doc and doc.index < args.start_doc:
            continue
        if limit is not None and len(rows) >= limit:
            break
        stats.docs_selected += 1
        if doc.text.strip():
            stats.docs_with_text += 1
        stats.chars_total += len(doc.text)
        row: Dict[str, Any] = {
            "doc_id": doc.doc_id,
            "xml_doc_index": doc.index,
            "text_chars": len(doc.text),
            "source_xml": Path(doc.source).name,
        }
        row.update(doc.metadata)
        rows.append(row)
        key_counter.update(row.keys())
    return rows, key_counter, stats


def run_annotation(args: argparse.Namespace) -> int:
    start = time.time()
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.metadata_output = args.metadata_output.expanduser().resolve() if args.metadata_output else None
    args.summary = args.summary.expanduser().resolve() if args.summary else args.output.with_suffix(".summary.json")
    args.errors = args.errors.expanduser().resolve() if args.errors else args.output.with_suffix(".errors.tsv")
    args.validation_report = args.validation_report.expanduser().resolve() if args.validation_report else args.output.with_suffix(".validation.tsv")
    args.manifest = args.manifest.expanduser().resolve() if args.manifest else args.output.with_suffix(".manifest.json")
    args.models_dir = args.models_dir.expanduser().resolve() if args.models_dir else None

    if not args.input.exists():
        raise SystemExit(f"Input XML not found: {args.input}")
    if args.output.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"Output already exists: {args.output}. Pass --overwrite to replace it.")

    required_metadata = parse_list_arg(args.require_metadata)
    preferred_comment_keys = parse_list_arg(args.metadata_comment_order)

    if args.dry_run:
        preview_rows, key_counter, stats = collect_xml_metadata_preview(args, limit=(args.limit_docs or None))
        print(json.dumps({
            "mode": "dry_run",
            "input": str(args.input),
            "documents_seen": stats.docs_seen,
            "documents_selected_previewed": len(preview_rows),
            "documents_with_text": stats.docs_with_text,
            "metadata_keys": sorted(key_counter),
            "sample_documents": preview_rows[:5],
        }, ensure_ascii=False, indent=2))
        return 0

    pipeline = build_pipeline(args)
    classla = ensure_classla()

    stats = RunStats()
    metadata_rows: List[Dict[str, Any]] = []
    metadata_keys: set[str] = {"doc_id", "xml_doc_index", "source_xml", "text_chars", "sentences", "tokens", "blocks"}
    error_rows: List[Dict[str, Any]] = []
    missing_metadata_counts: Counter[str] = Counter()

    ensure_parent(args.output)
    ensure_parent(args.errors)
    ensure_parent(args.summary)
    if args.metadata_output:
        ensure_parent(args.metadata_output)
    if args.validation_report:
        ensure_parent(args.validation_report)

    eprint(f"Reading XML: {args.input}")
    eprint(f"Writing CoNLL-U: {args.output}")

    with args.output.open("w", encoding="utf-8", newline="\n") as out_handle:
        for doc in iter_xml_documents(args):
            stats.docs_seen += 1
            if args.start_doc and doc.index < args.start_doc:
                continue
            if args.limit_docs and stats.docs_selected >= args.limit_docs:
                break
            stats.docs_selected += 1

            if stats.docs_selected == 1 or stats.docs_selected % args.progress_every == 0:
                eprint(f"[{stats.docs_selected}] Annotating {doc.doc_id} ({len(doc.text):,} chars)")

            row: Dict[str, Any] = {
                "doc_id": doc.doc_id,
                "xml_doc_index": doc.index,
                "source_xml": Path(doc.source).name,
                "text_chars": len(doc.text),
            }
            row.update(doc.metadata)
            for key in required_metadata:
                clean_key = sanitize_comment_key(key)
                if not row.get(clean_key):
                    missing_metadata_counts[clean_key] += 1
            metadata_keys.update(row.keys())
            stats.chars_total += len(doc.text)

            if not doc.text.strip():
                error_rows.append({
                    "doc_id": doc.doc_id,
                    "xml_doc_index": doc.index,
                    "block_number": "",
                    "error_type": "empty_document",
                    "error": "Document text is empty after XML text extraction.",
                })
                metadata_rows.append(row)
                continue

            stats.docs_with_text += 1
            blocks = make_blocks(doc.text, mode=args.block_mode, max_chars=args.max_chars)
            doc_sentences = 0
            doc_tokens = 0
            wrote_any_sentence = False

            for block_number, block in enumerate(blocks, start=1):
                stats.blocks_attempted += 1
                try:
                    predicted_doc = pipeline(block)
                    conllu_text = predicted_doc.to_conll()
                    predicted_sentences = parse_conllu_block(conllu_text)
                except Exception as exc:  # pragma: no cover - requires model/runtime
                    stats.blocks_failed += 1
                    error_rows.append({
                        "doc_id": doc.doc_id,
                        "xml_doc_index": doc.index,
                        "block_number": block_number,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    if args.fail_on_error:
                        raise
                    continue

                if not predicted_sentences:
                    stats.blocks_failed += 1
                    error_rows.append({
                        "doc_id": doc.doc_id,
                        "xml_doc_index": doc.index,
                        "block_number": block_number,
                        "error_type": "empty_classla_output",
                        "error": "CLASSLA returned no CoNLL-U sentences for this block.",
                    })
                    continue

                stats.blocks_written += 1
                for sent_index_in_block, sent in enumerate(predicted_sentences, start=1):
                    doc_sentences += 1
                    sent.comments = sentence_comments(
                        document=doc,
                        sentence=sent,
                        sentence_number=doc_sentences,
                        block_number=block_number,
                        first_sentence_in_doc=not wrote_any_sentence,
                        first_sentence_in_block=sent_index_in_block == 1,
                        block_mode=args.block_mode,
                        metadata_keys=preferred_comment_keys or None,
                    )
                    out_handle.write(sent.to_conllu())
                    out_handle.write("\n\n")
                    wrote_any_sentence = True
                    stats.sentences_written += 1
                    tokens = sent.word_token_count
                    doc_tokens += tokens
                    stats.tokens_written += tokens

            row["sentences"] = doc_sentences
            row["tokens"] = doc_tokens
            row["blocks"] = len(blocks)
            metadata_rows.append(row)

            if wrote_any_sentence:
                stats.docs_written += 1
            else:
                stats.docs_failed += 1
                error_rows.append({
                    "doc_id": doc.doc_id,
                    "xml_doc_index": doc.index,
                    "block_number": "",
                    "error_type": "no_sentences_written",
                    "error": "No sentences were written for this document.",
                })

    stats.seconds = round(time.time() - start, 3)

    # Metadata TSV.
    if args.metadata_output:
        header = ["doc_id", "xml_doc_index", "source_xml", "text_chars", "sentences", "tokens", "blocks"]
        for key in sorted(metadata_keys):
            if key not in header:
                header.append(key)
        write_tsv(args.metadata_output, metadata_rows, header=header)

    # Error TSV.
    write_tsv(
        args.errors,
        error_rows,
        header=["doc_id", "xml_doc_index", "block_number", "error_type", "error"],
    )

    # Validation TSV.
    validation_rows = validate_output_file(args.output)
    for key, count in sorted(missing_metadata_counts.items()):
        validation_rows.append({
            "check": f"missing_required_metadata:{key}",
            "value": count,
            "status": "ok" if count == 0 else "warn",
        })
    write_tsv(args.validation_report, validation_rows, header=["check", "value", "status"])

    validation_failures = [row for row in validation_rows if row.get("status") == "fail"]
    if validation_failures and args.require_clean_output:
        raise SystemExit(f"Validation failed: {validation_failures}. See {args.validation_report}")
    if error_rows and args.require_clean_output:
        raise SystemExit(f"Annotation produced {len(error_rows)} errors. See {args.errors}")

    output_hash = file_sha256(args.output) if args.output.exists() else None
    summary = {
        "created_utc": now_iso(),
        "input": str(args.input),
        "input_sha256": file_sha256(args.input),
        "output": str(args.output),
        "output_sha256": output_hash,
        "metadata_output": str(args.metadata_output) if args.metadata_output else None,
        "errors": str(args.errors),
        "validation_report": str(args.validation_report),
        "lang": args.lang,
        "classla_type": args.classla_type,
        "processors": args.processors,
        "models_dir": str(args.models_dir) if args.models_dir else None,
        "classla_version": get_classla_version(classla),
        "python": platform.python_version(),
        "xml_parse_mode": args.xml_parse_mode,
        "xml_doc_tag": args.xml_doc_tag,
        "xml_id_attr": args.xml_id_attr,
        "doc_id_prefix": args.doc_id_prefix,
        "block_mode": args.block_mode,
        "max_chars": args.max_chars,
        "required_metadata": required_metadata,
        "stats": asdict(stats),
        "error_count": len(error_rows),
        "validation": validation_rows,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Written annotated CoNLL-U: {args.output}")
    if args.metadata_output:
        print(f"Written metadata TSV: {args.metadata_output}")
    print(f"Written errors TSV: {args.errors}")
    print(f"Written validation TSV: {args.validation_report}")
    print(f"Written summary JSON: {args.summary}")
    print(
        f"Annotated {stats.docs_written}/{stats.docs_selected} selected documents, "
        f"{stats.sentences_written:,} sentences, {stats.tokens_written:,} word tokens."
    )
    if error_rows:
        eprint(f"Warning: {len(error_rows)} annotation errors were recorded. Inspect {args.errors}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate well-formed Wikivir XML with CLASSLA and preserve XML metadata in CoNLL-U comments."
    )
    parser.add_argument("--input", type=Path, required=True, help="Well-formed Wikivir XML file.")
    parser.add_argument("--output", type=Path, required=True, help="Output CoNLL-U file.")
    parser.add_argument("--metadata-output", type=Path, default=None, help="Optional document metadata TSV audit file.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional JSON summary path.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest JSON path. Defaults to output.manifest.json.")
    parser.add_argument("--errors", type=Path, default=None, help="Optional block/document error TSV path.")
    parser.add_argument("--validation-report", type=Path, default=None, help="Optional validation TSV path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Parse XML/metadata and print a JSON preview; do not run CLASSLA.")

    parser.add_argument("--lang", default="sl", help="CLASSLA language code. Default: sl.")
    parser.add_argument("--classla-type", default="standard", help="CLASSLA model type, e.g. standard, nonstandard, spoken.")
    parser.add_argument(
        "--processors",
        default=DEFAULT_PROCESSORS,
        help=f"CLASSLA processors. Default: {DEFAULT_PROCESSORS}.",
    )
    parser.add_argument("--models-dir", type=Path, default=None, help="Shared CLASSLA model/resource directory.")
    parser.add_argument("--download-models", action="store_true", help="Download/check models before annotation.")
    parser.add_argument(
        "--processing-mode",
        choices=["raw", "pretokenized"],
        default="raw",
        help="Use raw CLASSLA tokenization or pretokenized mode. Default: raw.",
    )

    parser.add_argument("--xml-doc-tag", default="doc", help="XML element used for documents. Default: doc.")
    parser.add_argument("--xml-id-attr", default=None, help="Optional XML attribute to use as the base document ID.")
    parser.add_argument(
        "--xml-parse-mode",
        choices=["strict", "recover"],
        default="strict",
        help="strict uses ElementTree and requires well-formed XML; recover scans <doc> tags only.",
    )
    parser.add_argument("--doc-id-prefix", default="wikivir", help="Prefix for generated order-based doc IDs. Default: wikivir.")
    parser.add_argument(
        "--doc-id-template",
        default="{prefix}-{index:06d}",
        help="Python format string for generated document IDs when XML has no id attribute.",
    )
    parser.add_argument(
        "--metadata-comment-order",
        default="title,author,century,year,genre,publication,date,source,url,xml_id,id",
        help="Comma-separated metadata keys to put first in CoNLL-U comments.",
    )
    parser.add_argument(
        "--require-metadata",
        default="title,author",
        help="Comma-separated metadata keys that should be present; missing values are reported. Default: title,author.",
    )

    parser.add_argument(
        "--block-mode",
        choices=["none", "blanklines", "lines"],
        default="blanklines",
        help="Split documents into CLASSLA annotation blocks. Default: blanklines.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=20000,
        help="Split overlong blocks beyond this many characters. 0 disables. Default: 20000.",
    )
    parser.add_argument("--start-doc", type=int, default=0, help="Start at XML document index N, useful for debugging/resuming.")
    parser.add_argument("--limit-docs", type=int, default=0, help="Annotate at most N selected documents, useful for smoke tests.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N selected docs. Default: 100.")
    parser.add_argument("--fail-on-error", action="store_true", help="Abort on first CLASSLA/block error.")
    parser.add_argument(
        "--require-clean-output",
        action="store_true",
        help="Exit non-zero if any annotation errors or validation failures are recorded.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_annotation(args)


if __name__ == "__main__":
    raise SystemExit(main())
