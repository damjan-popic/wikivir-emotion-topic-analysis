#!/usr/bin/env python3
"""Emotion and topic analysis for CLASSLA-annotated Wikivir-style CoNLL-U corpora.

The script is designed for already annotated corpora: tokenization, lemmatization,
UPOS/XPOS, dependency parses and NER are expected to be present in the input.

It combines three families of evidence:
  1. SloEmoLex lexicon matching over lemmas/forms, including Plutchik emotions,
     positive/negative sentiment, VAD, and emotion intensity columns when present.
  2. Unsupervised LDA topic modelling over CLASSLA lemmas.
  3. Optional transformer-based topic classification and embedding clustering.
  4. Optional BERTopic modelling with transformer embeddings and lemma-based c-TF-IDF.

The code deliberately fails softly for optional heavy dependencies. Core lexicon +
LDA analysis needs pandas/numpy/scikit-learn. Transformer outputs need transformers
sentence-transformers, and/or BERTopic.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script requires numpy. Install requirements.txt first.") from exc

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script requires pandas. Install requirements.txt first.") from exc


EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]
SENTIMENTS = ["positive", "negative"]
VAD = ["valence", "arousal", "dominance"]

EMOTION_ALIASES = {
    "anger": {"anger", "jeza", "angry"},
    "anticipation": {"anticipation", "anticip", "pričakovanje", "pricakovanje"},
    "disgust": {"disgust", "gnus", "odpor"},
    "fear": {"fear", "strah"},
    "joy": {"joy", "veselje", "radost", "happiness", "happy"},
    "sadness": {"sadness", "zalost", "žalost", "sad"},
    "surprise": {"surprise", "presenecenje", "presenečenje"},
    "trust": {"trust", "zaupanje"},
}
SENTIMENT_ALIASES = {
    "positive": {"positive", "positivity", "pozitivno", "pozitiven", "pos"},
    "negative": {"negative", "negativity", "negativno", "negativen", "neg"},
}
VAD_ALIASES = {
    "valence": {"valence", "valenca", "vad_valence", "nrc_vad_valence"},
    "arousal": {"arousal", "vzburjenje", "vzburjenost", "aktivacija", "vad_arousal", "nrc_vad_arousal"},
    "dominance": {"dominance", "dominanca", "vad_dominance", "nrc_vad_dominance"},
}
WORD_COLUMN_CANDIDATES = [
    "slovene",
    "slovenian",
    "sl",
    "word_sl",
    "sl_word",
    "slo",
    "lemma",
    "lemmas",
    "word",
    "term",
    "entry",
    "lexeme",
    "token",
    "form",
]

DEFAULT_CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "INTJ"}
DEFAULT_LEXICON_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "INTJ"}

# Small built-in Slovene stoplist. For serious runs, pass --stopwords with a domain-specific file.
DEFAULT_STOPWORDS = {
    "a", "ali", "bi", "bil", "bila", "bilo", "bili", "bile", "biti", "bo", "bom", "boš", "bomo", "boste", "bodo",
    "brez", "čez", "če", "četudi", "da", "de", "do", "ga", "jih", "jim", "in", "iz", "jaz", "je", "ji", "jo",
    "kaj", "kakor", "kako", "kamor", "kar", "kateri", "ki", "ko", "kot", "me", "med", "mene", "mi", "midva",
    "moj", "moja", "moje", "mu", "na", "nad", "naj", "najin", "nas", "naš", "naša", "naše", "ne", "nek", "nekaj",
    "ni", "nič", "njega", "njegov", "njej", "njen", "nje", "njih", "njihov", "njim", "njo", "no", "ob", "od", "ona",
    "oni", "ono", "on", "pa", "po", "pod", "pred", "pri", "proti", "sem", "se", "si", "smo", "so", "sta", "ste",
    "ta", "te", "tebe", "tebi", "tega", "tem", "temu", "ter", "ti", "tisti", "to", "toda", "tu", "tudi", "v", "vaš",
    "ve", "vedno", "vendar", "vi", "vse", "vsi", "vsak", "z", "za", "zaradi", "že",
}


# CoNLL-U comments are the primary metadata carrier for Wikivir in this project.
# The parser treats arbitrary document-level comments as metadata unless they are
# clearly sentence/structure comments. This keeps XML attributes such as title,
# author, century, genre, publication, year, source, etc. inside the annotated
# corpus instead of requiring a sidecar TSV.
STRUCTURAL_COMMENT_KEYS = {
    "sent_id", "text", "newdoc", "newdoc_id", "newpar", "newpar_id",
    "par", "par_id", "paragraph_id", "global_columns", "global_metadata",
}
DOC_METADATA_PREFIXES = ("doc_", "document_", "newdoc_")

GENERATED_AT = time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Token:
    id: str
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: str
    deprel: str
    deps: str
    misc: str
    sent_id: str
    doc_id: str
    par_id: str | None = None
    sentence_index: int = 0
    token_index: int = 0

    @property
    def ner(self) -> str:
        return parse_misc(self.misc).get("NER", "O")

    @property
    def no_space_after(self) -> bool:
        return parse_misc(self.misc).get("SpaceAfter") == "No"

    @property
    def is_word(self) -> bool:
        if self.upos in {"PUNCT", "SYM", "X"}:
            return False
        return bool(re.search(r"\w", self.form, flags=re.UNICODE))


@dataclass
class Sentence:
    sent_id: str
    doc_id: str
    tokens: list[Token]
    comments: list[str] = field(default_factory=list)
    text: str = ""
    par_id: str | None = None
    index: int = 0


@dataclass
class Document:
    doc_id: str
    sentences: list[Sentence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> list[Token]:
        return [tok for sent in self.sentences for tok in sent.tokens]

    @property
    def text(self) -> str:
        return "\n".join(sent.text for sent in self.sentences if sent.text)


@dataclass
class Segment:
    segment_id: str
    unit: str
    doc_id: str
    tokens: list[Token]
    text: str
    lemma_text: str
    sent_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return len(self.tokens)


@dataclass
class RunArtifacts:
    tables: dict[str, Path] = field(default_factory=dict)
    plots: dict[str, Path] = field(default_factory=dict)
    models: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[warning] {message}", file=sys.stderr)


def normalize_col(col: str) -> str:
    col = unicodedata.normalize("NFKC", str(col)).strip().lower()
    col = re.sub(r"[^\w]+", "_", col, flags=re.UNICODE).strip("_")
    return col


def strip_diacritics(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def normalize_lexeme(text: Any, mode: str = "simple", strip_accents: bool = False) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = unicodedata.normalize("NFKC", str(text)).strip().lower()
    s = s.replace("\u00a0", " ")
    s = s.strip(" \t\r\n\"'.,;:!?()[]{}<>»«„“”’`´…")
    if mode in {"simple", "historical", "ascii"}:
        s = re.sub(r"\s+", " ", s)
        s = s.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    if mode == "historical":
        replacements = {
            "ſ": "s",
            "ẛ": "s",
            "ꝛ": "r",
            "ȷ": "j",
            "ſh": "š",
            "zh": "č",
        }
        for old, new in replacements.items():
            s = s.replace(old, new)
        # Very cautious historical clean-up only. We do not rewrite old Slovene morphology.
        s = re.sub(r"([a-zčšž])\1{2,}", r"\1\1", s)
    if mode == "ascii" or strip_accents:
        s = strip_diacritics(s)
    return s


def parse_misc(misc: str) -> dict[str, str]:
    if not misc or misc == "_":
        return {}
    out: dict[str, str] = {}
    for part in misc.split("|"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
        elif part:
            out[part] = ""
    return out


def read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1250", "latin2", "latin1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "base": output_dir,
        "tables": output_dir / "tables",
        "plots": output_dir / "plots",
        "models": output_dir / "models",
        "logs": output_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def memory_rss_mb() -> float | None:
    """Best-effort resident-memory reading for terminal progress."""
    try:
        import psutil  # type: ignore
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 ** 2)
    except Exception:
        pass
    try:
        import resource
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB, macOS reports bytes. Heuristic keeps both sane.
        if rss > 10_000_000:
            return rss / (1024 ** 2)
        return rss / 1024
    except Exception:
        return None


def memory_string() -> str:
    mb = memory_rss_mb()
    return f", rss={mb:,.0f}MB" if mb is not None else ""


class ProgressLogger:
    """Small stderr progress reporter with elapsed time, ETA and memory."""

    def __init__(self, every: int = 1000, enabled: bool = True) -> None:
        self.every = max(1, int(every or 1))
        self.enabled = enabled
        self._starts: dict[str, float] = {}
        self._last: dict[str, float] = {}

    def stage_start(self, name: str, detail: str = "") -> None:
        if not self.enabled:
            return
        self._starts[name] = time.time()
        self._last[name] = 0.0
        suffix = f" — {detail}" if detail else ""
        print(f"[{now_stamp()}] ▶ {name}{suffix}{memory_string()}", file=sys.stderr, flush=True)

    def stage_done(self, name: str, detail: str = "") -> None:
        if not self.enabled:
            return
        elapsed = time.time() - self._starts.get(name, time.time())
        suffix = f" — {detail}" if detail else ""
        print(f"[{now_stamp()}] ✓ {name} done in {format_duration(elapsed)}{suffix}{memory_string()}", file=sys.stderr, flush=True)

    def stage_skip(self, name: str, detail: str = "existing checkpoint/output") -> None:
        if not self.enabled:
            return
        print(f"[{now_stamp()}] ↷ {name} skipped/resumed — {detail}{memory_string()}", file=sys.stderr, flush=True)

    def progress(self, name: str, current: int, total: int | None = None, force: bool = False, detail: str = "") -> None:
        if not self.enabled:
            return
        if not force and current % self.every != 0:
            return
        start = self._starts.get(name, time.time())
        elapsed = time.time() - start
        if total and total > 0:
            pct = min(100.0, current / total * 100.0)
            rate = current / elapsed if elapsed > 0 else 0.0
            eta = (total - current) / rate if rate > 0 else 0.0
            msg = f"[{now_stamp()}]   {name}: {current:,}/{total:,} ({pct:5.1f}%), elapsed={format_duration(elapsed)}, eta={format_duration(eta)}"
        else:
            msg = f"[{now_stamp()}]   {name}: {current:,}, elapsed={format_duration(elapsed)}"
        if detail:
            msg += f", {detail}"
        msg += memory_string()
        print(msg, file=sys.stderr, flush=True)


def get_progress(args: argparse.Namespace | None = None) -> ProgressLogger:
    every = getattr(args, "progress_every", 1000) if args is not None else 1000
    quiet = bool(getattr(args, "quiet_progress", False)) if args is not None else False
    return ProgressLogger(every=every, enabled=not quiet)


def checkpoint_dirs(args: argparse.Namespace, dirs: Mapping[str, Path]) -> dict[str, Path]:
    base = Path(getattr(args, "checkpoint_dir", "") or (dirs["base"] / "checkpoints"))
    stage_dirs = {
        "base": base,
        "lexicon": base / "lexicon",
        "transformer_classifier": base / "transformer_classifier",
        "embeddings": base / "embeddings",
        "bertopic": base / "bertopic",
        "lda": base / "lda",
    }
    for d in stage_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return stage_dirs


def should_resume(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "resume", False)) and not bool(getattr(args, "force", False))


def read_table_if_exists(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        # Let pandas infer numeric columns because downstream summaries/plots expect them.
        return pd.read_csv(path, sep="\t", keep_default_na=False)
    return pd.DataFrame()


def outputs_exist(paths: Sequence[Path]) -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_tsv_stream(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str] | None = None, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_iter = iter(rows)
    try:
        first = next(rows_iter)
    except StopIteration:
        if not append:
            path.write_text("", encoding="utf-8")
        return 0
    if fieldnames is None:
        fieldnames = list(first.keys())
    write_header = not append or not path.exists() or path.stat().st_size == 0
    with path.open("a" if append else "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), delimiter="\t", extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(dict(first))
        count = 1
        for row in rows_iter:
            writer.writerow(dict(row))
            count += 1
    return count


def concat_tsv_files(parts: Sequence[Path], output: Path) -> int:
    """Concatenate TSV parts without loading them all into memory."""
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    header: str | None = None
    with output.open("w", encoding="utf-8", newline="") as out:
        for part in parts:
            if not part.exists() or part.stat().st_size == 0:
                continue
            with part.open("r", encoding="utf-8") as f:
                first = f.readline()
                if not first:
                    continue
                if header is None:
                    header = first
                    out.write(first)
                # If headers differ, keep the first schema and still stream rows.
                for line in f:
                    out.write(line)
                    written += 1
    return written


def chunked(seq: Sequence[Any], size: int) -> Iterator[tuple[int, Sequence[Any]]]:
    size = max(1, int(size or 1))
    for start in range(0, len(seq), size):
        yield start, seq[start:start + size]


def shard_name(prefix: str, index: int) -> str:
    return f"{prefix}.part-{index:06d}"


def existing_part_indices(stage_dir: Path, prefix: str, suffix: str) -> set[int]:
    out: set[int] = set()
    pattern = re.compile(rf"^{re.escape(prefix)}\.part-(\d+)\.{re.escape(suffix.lstrip('.'))}$")
    for pth in stage_dir.glob(f"{prefix}.part-*.{suffix.lstrip('.')}"):
        m = pattern.match(pth.name)
        if m:
            out.add(int(m.group(1)))
    return out


def dataframe_headroom_note(rows: int, cols: int) -> str:
    return f"rows={rows:,}, cols={cols:,}"


def apply_low_memory_defaults(args: argparse.Namespace, artifacts: RunArtifacts) -> None:
    if not getattr(args, "low_memory", False):
        return
    changed: list[str] = []
    if args.transformer_batch_size > 4:
        args.transformer_batch_size = 4
        changed.append("--transformer-batch-size=4")
    if args.lda_n_jobs != 1:
        args.lda_n_jobs = 1
        changed.append("--lda-n-jobs=1")
    if args.lda_max_features > 30000:
        args.lda_max_features = 30000
        changed.append("--lda-max-features=30000")
    if args.bertopic_max_features > 30000:
        args.bertopic_max_features = 30000
        changed.append("--bertopic-max-features=30000")
    if args.bertopic_max_segments <= 0:
        args.bertopic_max_segments = 25000
        changed.append("--bertopic-max-segments=25000")
    if args.embedding_max_segments <= 0:
        args.embedding_max_segments = 50000
        changed.append("--embedding-max-segments=50000")
    if args.embedding_full_load_max_mb > 512:
        args.embedding_full_load_max_mb = 512
        changed.append("--embedding-full-load-max-mb=512")
    if args.embedding_pca_max_segments > 25000:
        args.embedding_pca_max_segments = 25000
        changed.append("--embedding-pca-max-segments=25000")
    if changed:
        artifacts.add_warning("Low-memory profile adjusted: " + ", ".join(changed))

def read_table_auto(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    # Sniff. SloEmoLex is TSV, but this also handles slightly renamed files.
    sample = read_text(path)[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        sep = dialect.delimiter
    except Exception:
        sep = "\t"
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, engine="python")


def parse_key_value_comment(line: str) -> tuple[str, str] | None:
    if not line.startswith("#"):
        return None
    body = line[1:].strip()
    if "=" not in body:
        return None
    key, value = body.split("=", 1)
    return key.strip(), value.strip()


def parse_doc_meta_value(value: str) -> dict[str, str]:
    """Parse metadata encoded as key=value|key=value or JSON-like strings."""
    value = value.strip()
    if not value:
        return {}
    if value.startswith("{") and value.endswith("}"):
        try:
            obj = json.loads(value)
            if isinstance(obj, dict):
                return {clean_metadata_key(str(k)): str(v) for k, v in obj.items() if clean_metadata_key(str(k))}
        except Exception:
            pass
    out: dict[str, str] = {}
    for sep in ("|", ";"):
        if sep in value:
            parts = value.split(sep)
            break
    else:
        parts = [value]
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            k_clean = clean_metadata_key(k)
            if k_clean:
                out[k_clean] = v.strip()
    return out


def clean_metadata_key(key: str) -> str:
    """Normalize a CoNLL-U comment key into a document metadata key.

    Examples accepted by this parser:
    - ``# title = ...`` -> ``title``
    - ``# newdoc title = ...`` -> ``title``
    - ``# doc_author = ...`` -> ``author``
    - ``# document.century = ...`` -> ``century``
    """
    key_norm = normalize_col(key)
    if key_norm in STRUCTURAL_COMMENT_KEYS:
        return ""
    for prefix in DOC_METADATA_PREFIXES:
        if key_norm.startswith(prefix) and len(key_norm) > len(prefix):
            stripped = key_norm[len(prefix):]
            if stripped not in STRUCTURAL_COMMENT_KEYS:
                return stripped
    if key_norm.startswith("meta_") and len(key_norm) > 5:
        return key_norm[5:]
    return key_norm


def apply_doc_comment_to_state(
    key: str,
    value: str,
    block_meta: MutableMapping[str, str],
) -> tuple[str | None, str | None, str | None, str | None]:
    """Classify a comment and update block-level metadata.

    Returns ``(newdoc_id, par_id, sent_id, sent_text)`` where each item is either
    a detected value or ``None``. Document metadata comments are inserted into
    ``block_meta`` in-place.
    """
    key_norm = normalize_col(key)
    if key_norm in {"newdoc_id", "newdoc"}:
        return value, None, None, None
    if key_norm in {"newpar_id", "newpar", "par_id", "paragraph_id"}:
        return None, value, None, None
    if key_norm == "sent_id":
        return None, None, value, None
    if key_norm == "text":
        return None, None, None, value
    if key_norm in {"doc_meta", "metadata", "meta", "newdoc_meta", "document_meta"}:
        block_meta.update(parse_doc_meta_value(value))
        return None, None, None, None

    meta_key = clean_metadata_key(key)
    if meta_key:
        block_meta[meta_key] = value
    return None, None, None, None


def metadata_presence_summary(docs: Sequence[Document]) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    nonempty: Counter[str] = Counter()
    for doc in docs:
        for key, value in doc.metadata.items():
            counts[key] += 1
            if str(value).strip():
                nonempty[key] += 1
    rows = []
    n = len(docs)
    for key in sorted(counts):
        rows.append({
            "metadata_key": key,
            "documents_with_key": counts[key],
            "documents_with_nonempty_value": nonempty[key],
            "coverage": nonempty[key] / n if n else 0.0,
        })
    return pd.DataFrame(rows)


def safe_doc_id(value: Any, fallback: str = "doc") -> str:
    """Return a CoNLL-U-comment-safe document ID."""
    s = unicodedata.normalize("NFKC", str(value or "")).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w.:-]+", "_", s, flags=re.UNICODE).strip("_")
    return s or fallback


def make_unique_doc_ids(docs: list[Document], artifacts: RunArtifacts | None = None) -> None:
    """Ensure every Document.doc_id is unique and update child sentence/token IDs.

    Badly restored CoNLL-U can contain repeated ``# newdoc id`` values. Pandas
    joins, segment IDs, transformer summaries, and document-level aggregation all
    require a unique document key. We keep the original value in
    ``doc.metadata['original_doc_id']`` and append a stable occurrence suffix.
    """
    if not docs:
        return
    totals = Counter(safe_doc_id(doc.doc_id, f"doc-{i:06d}") for i, doc in enumerate(docs, start=1))
    if all(count == 1 for count in totals.values()):
        # Still normalize truly empty/unsafe IDs, but do not touch already unique normal IDs.
        for i, doc in enumerate(docs, start=1):
            normalized = safe_doc_id(doc.doc_id, f"doc-{i:06d}")
            if normalized != doc.doc_id:
                old = doc.doc_id
                doc.doc_id = normalized
                doc.metadata.setdefault("original_doc_id", old)
                for sent in doc.sentences:
                    sent.doc_id = normalized
                    for tok in sent.tokens:
                        tok.doc_id = normalized
        return

    seen: Counter[str] = Counter()
    used: set[str] = set()
    changed = 0
    for i, doc in enumerate(docs, start=1):
        original = safe_doc_id(doc.doc_id, f"doc-{i:06d}")
        seen[original] += 1
        if totals[original] == 1 and original not in used:
            new_id = original
        else:
            doc.metadata.setdefault("original_doc_id", doc.doc_id)
            base = original or f"doc-{i:06d}"
            new_id = f"{base}__{seen[original]:04d}"
            while new_id in used:
                seen[original] += 1
                new_id = f"{base}__{seen[original]:04d}"
        used.add(new_id)
        if new_id != doc.doc_id:
            changed += 1
            doc.doc_id = new_id
            for sent in doc.sentences:
                sent.doc_id = new_id
                for tok in sent.tokens:
                    tok.doc_id = new_id

    duplicate_groups = sum(1 for count in totals.values() if count > 1)
    duplicate_docs = sum(count for count in totals.values() if count > 1)
    msg = (
        f"Found non-unique CoNLL-U document IDs: {duplicate_docs} documents in "
        f"{duplicate_groups} duplicate ID group(s). IDs were made unique with __0001-style "
        "suffixes; the original value is stored in original_doc_id."
    )
    if artifacts:
        artifacts.add_warning(msg)
    else:
        print(f"[warning] {msg}", file=sys.stderr)


def dataframe_to_unique_doc_mapping(df: pd.DataFrame, artifacts: RunArtifacts, label: str = "document metadata") -> dict[str, dict[str, Any]]:
    """Convert a doc_id-keyed DataFrame to a dict without crashing on duplicates."""
    if df.empty or "doc_id" not in df.columns:
        return {}
    work = df.copy()
    dup_mask = work["doc_id"].duplicated(keep=False)
    if bool(dup_mask.any()):
        dup_count = int(dup_mask.sum())
        group_count = int(work.loc[dup_mask, "doc_id"].nunique())
        artifacts.add_warning(
            f"{label} has {dup_count} rows in {group_count} duplicate doc_id group(s). "
            "Keeping the first row for metadata lookups; full rows are still written to tables."
        )
        work = work.drop_duplicates(subset=["doc_id"], keep="first")
    return work.set_index("doc_id", drop=False).to_dict(orient="index")


def has_useful_embedded_metadata(docs: Sequence[Document]) -> bool:
    useful = {"title", "author", "century", "year", "genre", "period", "publication", "source", "url", "date"}
    return any(any(k in useful and str(v).strip() for k, v in doc.metadata.items()) for doc in docs)

def reconstruct_text(tokens: Sequence[Token]) -> str:
    pieces: list[str] = []
    for tok in tokens:
        pieces.append(tok.form)
        if not tok.no_space_after:
            pieces.append(" ")
    return "".join(pieces).strip()


def parse_conllu(path: Path, artifacts: RunArtifacts | None = None) -> list[Document]:
    text = read_text(path)
    blocks = re.split(r"\n\s*\n", text.strip()) if text.strip() else []
    docs: list[Document] = []
    current_doc: Document | None = None
    default_doc_id = path.stem
    current_par_id: str | None = None
    sentence_index = 0
    global_token_index = 0

    def start_doc(doc_id: str, metadata: dict[str, Any] | None = None) -> Document:
        d = Document(doc_id=doc_id, metadata=metadata or {})
        docs.append(d)
        return d

    for block_i, block in enumerate(blocks, start=1):
        lines = [ln.rstrip("\n") for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        comments = [ln for ln in lines if ln.startswith("#")]
        token_lines = [ln for ln in lines if not ln.startswith("#")]

        newdoc_id: str | None = None
        block_meta: dict[str, str] = {}
        sent_id: str | None = None
        sent_text: str | None = None
        block_par_id = current_par_id

        for comment in comments:
            kv = parse_key_value_comment(comment)
            if not kv:
                continue
            key, value = kv
            detected_newdoc, detected_par, detected_sent, detected_text = apply_doc_comment_to_state(key, value, block_meta)
            if detected_newdoc is not None:
                newdoc_id = detected_newdoc
            if detected_par is not None:
                block_par_id = detected_par
                current_par_id = detected_par
            if detected_sent is not None:
                sent_id = detected_sent
            if detected_text is not None:
                sent_text = detected_text

        if newdoc_id:
            current_doc = start_doc(newdoc_id, block_meta)
            sentence_index = 0
            current_par_id = None
            if block_par_id:
                current_par_id = block_par_id
        elif current_doc is None:
            # If no explicit newdoc comments exist, infer a stable document ID from sent_id prefix if possible.
            inferred = default_doc_id
            if sent_id and "." in sent_id:
                inferred = sent_id.split(".", 1)[0]
            current_doc = start_doc(inferred, {})
        elif sent_id and "." in sent_id:
            # Fallback for CoNLL-U files that have no # newdoc lines but encode document IDs
            # in sent_id prefixes: doc123.1, doc123.2, doc124.1 ...
            inferred = sent_id.split(".", 1)[0]
            if inferred and inferred != current_doc.doc_id and current_doc.doc_id != default_doc_id:
                current_doc = start_doc(inferred, {})
                sentence_index = 0

        if block_meta and current_doc is not None:
            # Metadata comments sometimes appear after # newdoc or on first sentence blocks.
            current_doc.metadata.update({k: v for k, v in block_meta.items() if v != ""})

        if not token_lines:
            continue
        if sent_id is None:
            sent_id = f"{current_doc.doc_id}-sent-{len(current_doc.sentences) + 1}"
            if artifacts:
                artifacts.add_warning(f"Missing sent_id in block {block_i}; generated {sent_id}.")

        tokens: list[Token] = []
        for line in token_lines:
            parts = line.split("\t")
            if len(parts) != 10:
                if artifacts:
                    artifacts.add_warning(f"Skipping malformed CoNLL-U row in block {block_i}: expected 10 columns, got {len(parts)}.")
                continue
            tid = parts[0]
            if "-" in tid or "." in tid:
                # Multi-word tokens and empty nodes are not real lexical tokens.
                continue
            global_token_index += 1
            token = Token(
                id=tid,
                form=parts[1],
                lemma=parts[2],
                upos=parts[3],
                xpos=parts[4],
                feats=parts[5],
                head=parts[6],
                deprel=parts[7],
                deps=parts[8],
                misc=parts[9],
                sent_id=sent_id,
                doc_id=current_doc.doc_id,
                par_id=block_par_id,
                sentence_index=sentence_index,
                token_index=global_token_index,
            )
            tokens.append(token)
        if not tokens:
            continue
        sentence_text = sent_text if sent_text is not None else reconstruct_text(tokens)
        sent = Sentence(
            sent_id=sent_id,
            doc_id=current_doc.doc_id,
            tokens=tokens,
            comments=comments,
            text=sentence_text,
            par_id=block_par_id,
            index=sentence_index,
        )
        current_doc.sentences.append(sent)
        sentence_index += 1

    make_unique_doc_ids(docs, artifacts)
    return docs


def load_metadata(path: Path | None, doc_id_column: str | None = None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = read_table_auto(path)
    if df.empty:
        return df
    original_cols = list(df.columns)
    norm_to_original = {normalize_col(c): c for c in original_cols}
    if doc_id_column:
        if doc_id_column not in df.columns:
            norm = normalize_col(doc_id_column)
            if norm in norm_to_original:
                doc_id_column = norm_to_original[norm]
            else:
                raise ValueError(f"Metadata doc ID column not found: {doc_id_column}")
    else:
        for candidate in ("doc_id", "document_id", "id", "newdoc_id", "wikivir_id", "source_id", "title"):
            if candidate in norm_to_original:
                doc_id_column = norm_to_original[candidate]
                break
    if not doc_id_column:
        raise ValueError(
            "Could not infer metadata doc ID column. Pass --metadata-doc-id-column. "
            f"Available columns: {', '.join(original_cols)}"
        )
    if doc_id_column != "doc_id":
        df = df.rename(columns={doc_id_column: "doc_id"})
    df["doc_id"] = df["doc_id"].astype(str)
    return df


def merge_metadata(docs: list[Document], metadata_df: pd.DataFrame, artifacts: RunArtifacts) -> pd.DataFrame:
    rows = []
    for doc in docs:
        row = {"doc_id": doc.doc_id, **doc.metadata}
        rows.append(row)
    doc_df = pd.DataFrame(rows)
    if doc_df.empty:
        return doc_df
    if metadata_df.empty:
        return doc_df
    merged = doc_df.merge(metadata_df, on="doc_id", how="left", suffixes=("", "_metadata"))
    matched = int(merged[metadata_df.columns.difference(["doc_id"])].notna().any(axis=1).sum()) if len(metadata_df.columns) > 1 else 0
    if matched == 0:
        artifacts.add_warning("Metadata file loaded but no documents matched by doc_id. Check --metadata-doc-id-column and CoNLL-U # newdoc ids.")
    else:
        ratio = matched / max(1, len(doc_df))
        if ratio < 0.8:
            artifacts.add_warning(f"Only {matched}/{len(doc_df)} documents matched metadata ({ratio:.1%}).")
    return merged


def parse_upos_set(value: str | None, default: set[str]) -> set[str]:
    if not value:
        return set(default)
    return {x.strip().upper() for x in value.split(",") if x.strip()}


def load_stopwords(path: Path | None, extra: str | None = None) -> set[str]:
    stop = set(DEFAULT_STOPWORDS)
    if path:
        for line in read_text(path).splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                stop.add(normalize_lexeme(line))
    if extra:
        stop.update(normalize_lexeme(x) for x in extra.split(",") if x.strip())
    return {s for s in stop if s}


def token_for_topics(tok: Token, content_upos: set[str], stopwords: set[str], normalization: str, strip_accents: bool) -> str | None:
    if tok.upos not in content_upos:
        return None
    lemma = tok.lemma if tok.lemma and tok.lemma != "_" else tok.form
    lemma = normalize_lexeme(lemma, normalization, strip_accents)
    if not lemma or len(lemma) < 2 or lemma in stopwords:
        return None
    if re.fullmatch(r"[\d\W_]+", lemma, flags=re.UNICODE):
        return None
    return lemma


def build_lemma_text(tokens: Sequence[Token], content_upos: set[str], stopwords: set[str], normalization: str, strip_accents: bool) -> str:
    lemmas = []
    for tok in tokens:
        lemma = token_for_topics(tok, content_upos, stopwords, normalization, strip_accents)
        if lemma:
            lemmas.append(lemma)
    return " ".join(lemmas)


def build_segments(
    docs: list[Document],
    levels: set[str],
    content_upos: set[str],
    stopwords: set[str],
    normalization: str,
    strip_accents: bool,
    window_size: int = 250,
    window_step: int = 125,
    metadata_by_doc: Mapping[str, dict[str, Any]] | None = None,
) -> list[Segment]:
    segments: list[Segment] = []
    metadata_by_doc = metadata_by_doc or {}

    def make_segment(unit: str, doc: Document, suffix: str, tokens: list[Token], sent_ids: list[str]) -> Segment:
        text = reconstruct_text(tokens) if tokens else ""
        return Segment(
            segment_id=f"{doc.doc_id}::{unit}::{suffix}",
            unit=unit,
            doc_id=doc.doc_id,
            tokens=tokens,
            text=text,
            lemma_text=build_lemma_text(tokens, content_upos, stopwords, normalization, strip_accents),
            sent_ids=sent_ids,
            metadata=dict(metadata_by_doc.get(doc.doc_id, doc.metadata)),
        )

    for doc in docs:
        if "document" in levels:
            tokens = doc.tokens
            segments.append(make_segment("document", doc, "1", tokens, [s.sent_id for s in doc.sentences]))

        if "sentence" in levels:
            for i, sent in enumerate(doc.sentences, start=1):
                segments.append(make_segment("sentence", doc, str(i), sent.tokens, [sent.sent_id]))

        if "paragraph" in levels:
            grouped: dict[str, list[Sentence]] = defaultdict(list)
            any_par = False
            for sent in doc.sentences:
                key = sent.par_id or "paragraph-1"
                if sent.par_id:
                    any_par = True
                grouped[key].append(sent)
            if not any_par:
                # No paragraph information survived in the CoNLL-U; fall back to one paragraph per document.
                grouped = {"paragraph-1": doc.sentences}
            for key, sents in grouped.items():
                tokens = [tok for sent in sents for tok in sent.tokens]
                segments.append(make_segment("paragraph", doc, key, tokens, [s.sent_id for s in sents]))

        if "window" in levels:
            word_tokens = [tok for tok in doc.tokens if tok.is_word]
            if not word_tokens:
                continue
            if len(word_tokens) <= window_size:
                segments.append(make_segment("window", doc, "1", word_tokens, sorted({tok.sent_id for tok in word_tokens})))
            else:
                start = 0
                win_i = 1
                while start < len(word_tokens):
                    win_tokens = word_tokens[start : start + window_size]
                    if len(win_tokens) < max(25, window_size // 5):
                        break
                    segments.append(make_segment("window", doc, str(win_i), win_tokens, sorted({tok.sent_id for tok in win_tokens})))
                    win_i += 1
                    start += window_step
    return segments


def segments_to_frame(segments: Sequence[Segment]) -> pd.DataFrame:
    rows = []
    for seg in segments:
        lexical_tokens = [tok for tok in seg.tokens if tok.is_word]
        rows.append(
            {
                "segment_id": seg.segment_id,
                "unit": seg.unit,
                "doc_id": seg.doc_id,
                "sent_ids": "|".join(seg.sent_ids[:20]),
                "n_sent_ids": len(seg.sent_ids),
                "token_count": len(seg.tokens),
                "word_token_count": len(lexical_tokens),
                "lemma_token_count": len(seg.lemma_text.split()) if seg.lemma_text else 0,
                "text_preview": seg.text[:300].replace("\t", " ").replace("\n", " "),
                **seg.metadata,
            }
        )
    return pd.DataFrame(rows)


def detect_word_columns(df: pd.DataFrame, explicit: Sequence[str] | None = None) -> list[str]:
    if explicit:
        cols: list[str] = []
        norm_to_original = {normalize_col(c): c for c in df.columns}
        for c in explicit:
            if c in df.columns:
                cols.append(c)
            elif normalize_col(c) in norm_to_original:
                cols.append(norm_to_original[normalize_col(c)])
            else:
                raise ValueError(f"Requested lexicon match column not found: {c}")
        return cols

    norm_to_original = {normalize_col(c): c for c in df.columns}
    cols = []
    for candidate in WORD_COLUMN_CANDIDATES:
        if candidate in norm_to_original:
            cols.append(norm_to_original[candidate])
    if cols:
        # Prefer first Slovene-looking column. Additional lemma/word columns can still be used if passed explicitly.
        return [cols[0]]

    # Fallback: first non-numeric-looking column that is not an obvious metadata/citation column.
    bad = {"id", "index", "source", "origin", "english", "en", "translation", "notes", "comment"}
    for col in df.columns:
        cn = normalize_col(col)
        if cn in bad:
            continue
        sample = df[col].astype(str).head(50)
        numeric_fraction = pd.to_numeric(sample, errors="coerce").notna().mean()
        if numeric_fraction < 0.2:
            return [col]
    raise ValueError("Could not detect lexicon word/lemma column. Pass --sloemolex-word-columns.")


def numeric_series(values: pd.Series) -> pd.Series:
    # Handle decimal commas and textual booleans.
    s = values.astype(str).str.strip().str.replace(",", ".", regex=False)
    replacements = {
        "true": "1", "yes": "1", "y": "1", "da": "1", "t": "1",
        "false": "0", "no": "0", "n": "0", "ne": "0", "f": "0", "": "0", "_": "0",
    }
    s = s.str.lower().map(lambda x: replacements.get(x, x))
    return pd.to_numeric(s, errors="coerce")


def column_contains_alias(col_norm: str, aliases: set[str]) -> bool:
    parts = set(col_norm.split("_")) | {col_norm}
    ascii_col = strip_diacritics(col_norm)
    parts |= set(ascii_col.split("_")) | {ascii_col}
    return any(strip_diacritics(alias.lower()) in parts or strip_diacritics(alias.lower()) == ascii_col for alias in aliases)


@dataclass
class SloEmoLex:
    lookup: dict[str, dict[str, float]]
    word_columns: list[str]
    emotion_columns: dict[str, list[str]]
    intensity_columns: dict[str, list[str]]
    sentiment_columns: dict[str, list[str]]
    vad_columns: dict[str, list[str]]
    row_count: int
    key_count: int

    @classmethod
    def load(
        cls,
        path: Path,
        word_columns: Sequence[str] | None = None,
        normalization: str = "simple",
        strip_accents: bool = False,
        artifacts: RunArtifacts | None = None,
    ) -> "SloEmoLex":
        df = read_table_auto(path)
        if df.empty:
            raise ValueError(f"SloEmoLex file is empty: {path}")
        original_columns = list(df.columns)
        word_cols = detect_word_columns(df, word_columns)
        norm_cols = {col: normalize_col(col) for col in original_columns}

        emotion_cols: dict[str, list[str]] = {emo: [] for emo in EMOTIONS}
        intensity_cols: dict[str, list[str]] = {emo: [] for emo in EMOTIONS}
        sentiment_cols: dict[str, list[str]] = {s: [] for s in SENTIMENTS}
        vad_cols: dict[str, list[str]] = {v: [] for v in VAD}

        for col, cn in norm_cols.items():
            if col in word_cols:
                continue
            cn_ascii = strip_diacritics(cn)
            for emo, aliases in EMOTION_ALIASES.items():
                if column_contains_alias(cn, aliases):
                    if any(x in cn_ascii for x in ("intensity", "intens", "affect", "score")) and not cn_ascii in aliases:
                        intensity_cols[emo].append(col)
                    else:
                        emotion_cols[emo].append(col)
            for sent, aliases in SENTIMENT_ALIASES.items():
                if column_contains_alias(cn, aliases):
                    # Avoid treating valence as positive sentiment.
                    if "valence" not in cn and "valenca" not in cn:
                        sentiment_cols[sent].append(col)
            for vad, aliases in VAD_ALIASES.items():
                if column_contains_alias(cn, aliases):
                    vad_cols[vad].append(col)

        # Long-format fallback: word, emotion, score. This is not the expected SloEmoLex shape,
        # but it helps with derivatives.
        norm_to_original = {normalize_col(c): c for c in df.columns}
        if not any(emotion_cols.values()) and {"emotion", "score"}.issubset(norm_to_original):
            word_col = word_cols[0]
            emotion_col = norm_to_original["emotion"]
            score_col = norm_to_original["score"]
            pivot_rows = []
            for _, row in df.iterrows():
                word = row[word_col]
                emo = normalize_lexeme(row[emotion_col], strip_accents=True)
                score = pd.to_numeric(str(row[score_col]).replace(",", "."), errors="coerce")
                if emo in EMOTIONS:
                    pivot_rows.append({word_col: word, emo: score})
            if pivot_rows:
                df = pd.DataFrame(pivot_rows).groupby(word_col, as_index=False).max(numeric_only=True)
                word_cols = [word_col]
                emotion_cols = {emo: [emo] if emo in df.columns else [] for emo in EMOTIONS}

        if artifacts:
            missing = [emo for emo, cols in emotion_cols.items() if not cols]
            if missing:
                artifacts.add_warning(
                    "No explicit SloEmoLex association columns detected for: " + ", ".join(missing) +
                    ". This may be fine if the file only contains intensity/VAD columns, but check lexicon_coverage.tsv."
                )

        keyed_rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            keys: set[str] = set()
            for col in word_cols:
                raw = row.get(col, "")
                for candidate in re.split(r"[|;/]", str(raw)):
                    key = normalize_lexeme(candidate, normalization, strip_accents)
                    if key:
                        keys.add(key)
            if not keys:
                continue
            values: dict[str, float] = {}
            for emo, cols in emotion_cols.items():
                nums = [numeric_series(pd.Series([row[c]])).iloc[0] for c in cols]
                nums = [float(x) for x in nums if not pd.isna(x)]
                values[f"emotion_{emo}"] = max(nums) if nums else 0.0
            for emo, cols in intensity_cols.items():
                nums = [numeric_series(pd.Series([row[c]])).iloc[0] for c in cols]
                nums = [float(x) for x in nums if not pd.isna(x)]
                values[f"intensity_{emo}"] = max(nums) if nums else np.nan
            for sent, cols in sentiment_cols.items():
                nums = [numeric_series(pd.Series([row[c]])).iloc[0] for c in cols]
                nums = [float(x) for x in nums if not pd.isna(x)]
                values[f"sentiment_{sent}"] = max(nums) if nums else 0.0
            for vad, cols in vad_cols.items():
                nums = [numeric_series(pd.Series([row[c]])).iloc[0] for c in cols]
                nums = [float(x) for x in nums if not pd.isna(x)]
                values[f"vad_{vad}"] = float(np.mean(nums)) if nums else np.nan
            for key in keys:
                keyed_rows.append({"key": key, **values})

        if not keyed_rows:
            raise ValueError(f"No usable lexicon entries found in {path}")
        keyed = pd.DataFrame(keyed_rows)
        agg_spec = {}
        for c in keyed.columns:
            if c == "key":
                continue
            if c.startswith("vad_"):
                agg_spec[c] = "mean"
            else:
                agg_spec[c] = "max"
        keyed = keyed.groupby("key", as_index=False).agg(agg_spec)
        lookup = {row["key"]: {k: float(v) if not pd.isna(v) else np.nan for k, v in row.items() if k != "key"} for _, row in keyed.iterrows()}
        return cls(
            lookup=lookup,
            word_columns=word_cols,
            emotion_columns=emotion_cols,
            intensity_columns=intensity_cols,
            sentiment_columns=sentiment_cols,
            vad_columns=vad_cols,
            row_count=len(df),
            key_count=len(lookup),
        )


def detect_fuzzy_matcher(artifacts: RunArtifacts) -> Callable[[str, Sequence[str], int], tuple[str | None, float]] | None:
    try:
        from rapidfuzz import process, fuzz  # type: ignore
    except Exception:
        artifacts.add_warning("--fuzzy-lexicon was requested, but rapidfuzz is not installed. Fuzzy matching disabled.")
        return None

    def match(query: str, choices: Sequence[str], threshold: int) -> tuple[str | None, float]:
        result = process.extractOne(query, choices, scorer=fuzz.WRatio, score_cutoff=threshold)
        if not result:
            return None, 0.0
        return str(result[0]), float(result[1])

    return match


def score_segment_with_lexicon(
    seg: Segment,
    lexicon: SloEmoLex,
    lexicon_upos: set[str],
    match_on: Sequence[str],
    normalization: str,
    strip_accents: bool,
    fuzzy_matcher: Callable[[str, Sequence[str], int], tuple[str | None, float]] | None = None,
    fuzzy_threshold: int = 92,
    fuzzy_cache: MutableMapping[str, tuple[str | None, float]] | None = None,
    export_matches: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lexical_tokens = [tok for tok in seg.tokens if tok.is_word and tok.upos in lexicon_upos]
    row: dict[str, Any] = {
        "segment_id": seg.segment_id,
        "unit": seg.unit,
        "doc_id": seg.doc_id,
        "token_count": len(seg.tokens),
        "lexical_token_count": len(lexical_tokens),
    }
    for emo in EMOTIONS:
        row[f"emotion_{emo}_count"] = 0.0
        row[f"emotion_{emo}_per_1k"] = 0.0
        row[f"emotion_{emo}_prop"] = 0.0
        row[f"intensity_{emo}_mean"] = np.nan
    for sent in SENTIMENTS:
        row[f"sentiment_{sent}_count"] = 0.0
        row[f"sentiment_{sent}_per_1k"] = 0.0
    for vad in VAD:
        row[f"vad_{vad}_mean"] = np.nan

    hit_count = 0
    unique_hits: set[str] = set()
    match_rows: list[dict[str, Any]] = []
    intensity_values: dict[str, list[float]] = {emo: [] for emo in EMOTIONS}
    vad_values: dict[str, list[float]] = {v: [] for v in VAD}
    lexicon_keys = list(lexicon.lookup.keys())
    fuzzy_cache = fuzzy_cache if fuzzy_cache is not None else {}

    for tok in lexical_tokens:
        candidates: list[tuple[str, str]] = []
        for source in match_on:
            if source == "lemma":
                raw = tok.lemma if tok.lemma and tok.lemma != "_" else ""
            elif source == "form":
                raw = tok.form
            elif source == "lemma_or_form":
                raw = tok.lemma if tok.lemma and tok.lemma != "_" else tok.form
            else:
                continue
            key = normalize_lexeme(raw, normalization, strip_accents)
            if key:
                candidates.append((source, key))
        matched_key: str | None = None
        matched_source = ""
        fuzzy_score = np.nan
        for source, key in candidates:
            if key in lexicon.lookup:
                matched_key = key
                matched_source = source
                break
        if matched_key is None and fuzzy_matcher:
            for source, key in candidates:
                if len(key) < 5:
                    continue
                if key in fuzzy_cache:
                    fkey, fscore = fuzzy_cache[key]
                else:
                    fkey, fscore = fuzzy_matcher(key, lexicon_keys, fuzzy_threshold)
                    fuzzy_cache[key] = (fkey, fscore)
                if fkey:
                    matched_key = fkey
                    matched_source = f"{source}:fuzzy:{key}"
                    fuzzy_score = fscore
                    break
        if matched_key is None:
            continue
        entry = lexicon.lookup[matched_key]
        hit_count += 1
        unique_hits.add(matched_key)
        for emo in EMOTIONS:
            val = entry.get(f"emotion_{emo}", 0.0)
            if not pd.isna(val):
                row[f"emotion_{emo}_count"] += float(val)
            intensity = entry.get(f"intensity_{emo}", np.nan)
            if not pd.isna(intensity):
                intensity_values[emo].append(float(intensity))
        for sent in SENTIMENTS:
            val = entry.get(f"sentiment_{sent}", 0.0)
            if not pd.isna(val):
                row[f"sentiment_{sent}_count"] += float(val)
        for vad in VAD:
            val = entry.get(f"vad_{vad}", np.nan)
            if not pd.isna(val):
                vad_values[vad].append(float(val))
        if export_matches:
            mr = {
                "segment_id": seg.segment_id,
                "unit": seg.unit,
                "doc_id": seg.doc_id,
                "sent_id": tok.sent_id,
                "token_id": tok.id,
                "form": tok.form,
                "lemma": tok.lemma,
                "upos": tok.upos,
                "matched_key": matched_key,
                "match_source": matched_source,
                "fuzzy_score": fuzzy_score,
            }
            for emo in EMOTIONS:
                mr[f"emotion_{emo}"] = entry.get(f"emotion_{emo}", 0.0)
                mr[f"intensity_{emo}"] = entry.get(f"intensity_{emo}", np.nan)
            for sent in SENTIMENTS:
                mr[f"sentiment_{sent}"] = entry.get(f"sentiment_{sent}", 0.0)
            for vad in VAD:
                mr[f"vad_{vad}"] = entry.get(f"vad_{vad}", np.nan)
            match_rows.append(mr)

    denom = max(1, len(lexical_tokens))
    emotion_total = sum(row[f"emotion_{emo}_count"] for emo in EMOTIONS)
    for emo in EMOTIONS:
        row[f"emotion_{emo}_per_1k"] = row[f"emotion_{emo}_count"] / denom * 1000.0
        row[f"emotion_{emo}_prop"] = row[f"emotion_{emo}_count"] / emotion_total if emotion_total > 0 else 0.0
        if intensity_values[emo]:
            row[f"intensity_{emo}_mean"] = float(np.mean(intensity_values[emo]))
    for sent in SENTIMENTS:
        row[f"sentiment_{sent}_per_1k"] = row[f"sentiment_{sent}_count"] / denom * 1000.0
    pos = row["sentiment_positive_count"]
    neg = row["sentiment_negative_count"]
    row["sentiment_balance"] = (pos - neg) / (pos + neg) if (pos + neg) else np.nan
    for vad in VAD:
        if vad_values[vad]:
            row[f"vad_{vad}_mean"] = float(np.mean(vad_values[vad]))
    row["lexicon_hit_count"] = hit_count
    row["lexicon_unique_hit_count"] = len(unique_hits)
    row["lexicon_coverage"] = hit_count / denom if denom else 0.0
    if emotion_total > 0:
        row["dominant_emotion"] = max(EMOTIONS, key=lambda emo: row[f"emotion_{emo}_count"])
        row["dominant_emotion_score"] = row[f"emotion_{row['dominant_emotion']}_prop"]
    else:
        row["dominant_emotion"] = ""
        row["dominant_emotion_score"] = np.nan
    return row, match_rows


def run_lexicon_analysis(
    segments: Sequence[Segment],
    lexicon: SloEmoLex,
    args: argparse.Namespace,
    dirs: Mapping[str, Path],
    artifacts: RunArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score segments with SloEmoLex, checkpointed by segment shards.

    The old implementation accumulated every token-level hit in memory. This
    version writes per-shard TSV files and concatenates them at the end, so a
    crash can resume from the next unfinished shard and token-match exports do
    not balloon RAM.
    """
    scores_out = dirs["tables"] / "segment_emotion_scores.tsv"
    matches_out = dirs["tables"] / "token_emotion_matches.tsv"
    if should_resume(args) and scores_out.exists() and scores_out.stat().st_size > 0:
        get_progress(args).stage_skip("SloEmoLex emotion scoring", f"loaded {scores_out}")
        artifacts.tables["segment_emotion_scores"] = scores_out
        if matches_out.exists() and matches_out.stat().st_size > 0:
            artifacts.tables["token_emotion_matches"] = matches_out
        return read_table_if_exists(scores_out), read_table_if_exists(matches_out) if matches_out.exists() else pd.DataFrame()

    progress = get_progress(args)
    progress.stage_start("SloEmoLex emotion scoring", f"{len(segments):,} segments, shard={args.lexicon_shard_size:,}")
    ckpt = checkpoint_dirs(args, dirs)["lexicon"]
    lexicon_upos = parse_upos_set(args.lexicon_upos, DEFAULT_LEXICON_UPOS)
    match_on = [m.strip() for m in args.match_on.split(",") if m.strip()]
    fuzzy_matcher = None
    if args.fuzzy_lexicon:
        fuzzy_matcher = detect_fuzzy_matcher(artifacts)
    fuzzy_cache: dict[str, tuple[str | None, float]] = {}
    score_parts: list[Path] = []
    match_parts: list[Path] = []
    shard_size = max(1, int(args.lexicon_shard_size))
    total_shards = math.ceil(len(segments) / shard_size) if segments else 0

    for shard_idx, start in enumerate(range(0, len(segments), shard_size), start=1):
        shard = segments[start:start + shard_size]
        score_part = ckpt / f"segment_emotion_scores.part-{shard_idx:06d}.tsv"
        match_part = ckpt / f"token_emotion_matches.part-{shard_idx:06d}.tsv"
        done_marker = ckpt / f"lexicon.part-{shard_idx:06d}.done.json"
        if should_resume(args) and done_marker.exists() and score_part.exists() and score_part.stat().st_size > 0:
            score_parts.append(score_part)
            if match_part.exists() and match_part.stat().st_size > 0:
                match_parts.append(match_part)
            progress.progress("SloEmoLex emotion scoring", min(start + len(shard), len(segments)), len(segments), force=True, detail=f"resumed shard {shard_idx}/{total_shards}")
            continue

        rows: list[dict[str, Any]] = []
        match_rows: list[dict[str, Any]] = []
        for local_i, seg in enumerate(shard, start=1):
            row, matches = score_segment_with_lexicon(
                seg,
                lexicon,
                lexicon_upos=lexicon_upos,
                match_on=match_on,
                normalization=args.normalization,
                strip_accents=args.strip_diacritics,
                fuzzy_matcher=fuzzy_matcher,
                fuzzy_threshold=args.fuzzy_threshold,
                fuzzy_cache=fuzzy_cache,
                export_matches=args.export_token_matches,
            )
            row.update(seg.metadata)
            rows.append(row)
            if args.export_token_matches:
                match_rows.extend(matches)
            global_i = start + local_i
            progress.progress("SloEmoLex emotion scoring", global_i, len(segments))

        pd.DataFrame(rows).to_csv(score_part, sep="\t", index=False)
        score_parts.append(score_part)
        if args.export_token_matches and match_rows:
            pd.DataFrame(match_rows).to_csv(match_part, sep="\t", index=False)
            match_parts.append(match_part)
        write_json(done_marker, {"stage": "lexicon", "shard": shard_idx, "start": start, "count": len(shard), "finished_at": now_stamp()})
        progress.progress("SloEmoLex emotion scoring", min(start + len(shard), len(segments)), len(segments), force=True, detail=f"wrote shard {shard_idx}/{total_shards}")
        del rows, match_rows
        gc.collect()

    concat_tsv_files(score_parts, scores_out)
    artifacts.tables["segment_emotion_scores"] = scores_out
    matches = pd.DataFrame()
    if args.export_token_matches and match_parts:
        concat_tsv_files(match_parts, matches_out)
        artifacts.tables["token_emotion_matches"] = matches_out
        if args.load_token_matches:
            matches = read_table_if_exists(matches_out)
    scores = read_table_if_exists(scores_out)
    progress.stage_done("SloEmoLex emotion scoring", dataframe_headroom_note(len(scores), len(scores.columns)))
    return scores, matches

def aggregate_document_scores(segment_scores: pd.DataFrame, dirs: Mapping[str, Path], artifacts: RunArtifacts) -> pd.DataFrame:
    if segment_scores.empty:
        return pd.DataFrame()
    doc_rows = segment_scores[segment_scores["unit"] == "document"].copy()
    if doc_rows.empty:
        # Weighted aggregation fallback.
        numeric_cols = [c for c in segment_scores.columns if c.endswith("_count") or c.endswith("_per_1k") or c in {"lexicon_hit_count", "lexical_token_count", "token_count"}]
        doc_rows = segment_scores.groupby("doc_id", as_index=False)[numeric_cols].sum(numeric_only=True)
    out = dirs["tables"] / "document_emotion_scores.tsv"
    doc_rows.to_csv(out, sep="\t", index=False)
    artifacts.tables["document_emotion_scores"] = out
    return doc_rows


def weighted_group_summary(df: pd.DataFrame, group_col: str, weight_col: str = "lexical_token_count") -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    metric_cols = []
    for c in df.columns:
        if c.startswith("emotion_") and (c.endswith("_per_1k") or c.endswith("_prop") or c.endswith("_count")):
            metric_cols.append(c)
        elif c.startswith("sentiment_") and (c.endswith("_per_1k") or c.endswith("_count")):
            metric_cols.append(c)
        elif c.startswith("vad_") and c.endswith("_mean"):
            metric_cols.append(c)
        elif c in {"sentiment_balance", "lexicon_coverage", "lexicon_hit_count", "lexical_token_count", "token_count"}:
            metric_cols.append(c)
    rows = []
    for value, g in df.dropna(subset=[group_col]).groupby(group_col, dropna=False):
        weights = pd.to_numeric(g.get(weight_col, pd.Series(np.ones(len(g)))), errors="coerce").fillna(0).to_numpy(dtype=float)
        if not weights.any():
            weights = np.ones(len(g), dtype=float)
        row = {group_col: value, "n_documents": len(g), "weight_sum": float(weights.sum())}
        for c in metric_cols:
            vals = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            mask = ~np.isnan(vals)
            if not mask.any():
                row[c] = np.nan
            elif c.endswith("_count") or c in {"lexicon_hit_count", "lexical_token_count", "token_count"}:
                row[c] = float(np.nansum(vals))
            else:
                w = weights[mask]
                if not w.any():
                    row[c] = float(np.nanmean(vals[mask]))
                else:
                    row[c] = float(np.average(vals[mask], weights=w))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_documents", ascending=False)


def write_grouped_emotion_summaries(doc_scores: pd.DataFrame, dirs: Mapping[str, Path], artifacts: RunArtifacts, max_unique: int = 80) -> list[Path]:
    if doc_scores.empty:
        return []
    excluded = {"doc_id", "segment_id", "unit", "text_preview", "sent_ids"}
    common = ["author", "title", "century", "year", "genre", "period", "publication", "source"]
    candidates = []
    for c in common + list(doc_scores.columns):
        if c in excluded or c not in doc_scores.columns or c in candidates:
            continue
        nonempty_values = doc_scores[c].astype(str).map(lambda x: np.nan if x == "" else x)
        nunique = nonempty_values.dropna().nunique()
        if 1 < nunique <= max_unique:
            candidates.append(c)
    paths = []
    for c in candidates:
        summary = weighted_group_summary(doc_scores, c)
        if summary.empty:
            continue
        safe = re.sub(r"[^\w-]+", "_", c)
        path = dirs["tables"] / f"emotion_by_{safe}.tsv"
        summary.to_csv(path, sep="\t", index=False)
        artifacts.tables[f"emotion_by_{safe}"] = path
        paths.append(path)
    return paths


def extract_named_entities(docs: list[Document], dirs: Mapping[str, Path], artifacts: RunArtifacts) -> pd.DataFrame:
    rows = []
    for doc in docs:
        for sent in doc.sentences:
            current: dict[str, Any] | None = None
            for tok in sent.tokens + [Token("", "", "", "", "", "", "", "", "", "", sent.sent_id, doc.doc_id)]:
                ner = tok.ner if tok.id else "O"
                if ner.startswith("B-"):
                    if current:
                        rows.append(current)
                    label = ner[2:]
                    current = {
                        "doc_id": doc.doc_id,
                        "sent_id": sent.sent_id,
                        "entity_type": label,
                        "entity_text": tok.form,
                        "entity_lemma": tok.lemma if tok.lemma != "_" else tok.form,
                        "start_token_index": tok.token_index,
                        "end_token_index": tok.token_index,
                    }
                elif ner.startswith("I-") and current and ner[2:] == current["entity_type"]:
                    current["entity_text"] += " " + tok.form
                    current["entity_lemma"] += " " + (tok.lemma if tok.lemma != "_" else tok.form)
                    current["end_token_index"] = tok.token_index
                else:
                    if current:
                        rows.append(current)
                        current = None
    ents = pd.DataFrame(rows)
    if not ents.empty:
        path = dirs["tables"] / "named_entities.tsv"
        ents.to_csv(path, sep="\t", index=False)
        artifacts.tables["named_entities"] = path
        counts = ents.groupby(["entity_type", "entity_text"], as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
        counts_path = dirs["tables"] / "named_entity_counts.tsv"
        counts.to_csv(counts_path, sep="\t", index=False)
        artifacts.tables["named_entity_counts"] = counts_path
    return ents


def write_entity_emotion_summary(entities: pd.DataFrame, segment_scores: pd.DataFrame, dirs: Mapping[str, Path], artifacts: RunArtifacts) -> pd.DataFrame:
    if entities.empty or segment_scores.empty:
        return pd.DataFrame()
    sentence_scores = segment_scores[segment_scores["unit"] == "sentence"].copy()
    if sentence_scores.empty:
        return pd.DataFrame()
    # segment_id for sentences is doc::sentence::n, so join by sent_id via sent_ids column.
    tmp = sentence_scores[["doc_id", "sent_ids"] + [c for c in sentence_scores.columns if c.startswith("emotion_") or c.startswith("sentiment_") or c.startswith("vad_") or c in {"lexicon_coverage", "dominant_emotion"}]].copy()
    tmp = tmp.rename(columns={"sent_ids": "sent_id"})
    merged = entities.merge(tmp, on=["doc_id", "sent_id"], how="left")
    if merged.empty:
        return pd.DataFrame()
    agg_cols = [c for c in merged.columns if c.startswith("emotion_") and c.endswith("_per_1k")]
    rows = []
    for (etype, etext), g in merged.groupby(["entity_type", "entity_text"]):
        row = {"entity_type": etype, "entity_text": etext, "occurrences": len(g), "doc_count": g["doc_id"].nunique()}
        for c in agg_cols:
            row[c] = pd.to_numeric(g[c], errors="coerce").mean()
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("occurrences", ascending=False)
    path = dirs["tables"] / "entity_emotion_summary.tsv"
    out.to_csv(path, sep="\t", index=False)
    artifacts.tables["entity_emotion_summary"] = path
    return out


def import_sklearn(artifacts: RunArtifacts):
    try:
        from sklearn.decomposition import LatentDirichletAllocation, PCA
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        from sklearn.cluster import MiniBatchKMeans, KMeans
        from sklearn.metrics import silhouette_score
        import joblib
        return {
            "LatentDirichletAllocation": LatentDirichletAllocation,
            "PCA": PCA,
            "CountVectorizer": CountVectorizer,
            "TfidfVectorizer": TfidfVectorizer,
            "MiniBatchKMeans": MiniBatchKMeans,
            "KMeans": KMeans,
            "silhouette_score": silhouette_score,
            "joblib": joblib,
        }
    except Exception as exc:
        artifacts.add_warning(f"scikit-learn/joblib not available; LDA and clustering skipped. Details: {exc}")
        return None


def umass_coherence(X: Any, feature_names: Sequence[str], topics: list[list[str]], eps: float = 1.0) -> float:
    """Compute a lightweight UMass-style topic coherence over segment documents."""
    if X is None or not topics:
        return float("nan")
    # Binary document-term matrix. Convert once to csc for fast column slicing.
    try:
        X_bin = (X > 0).astype(int).tocsc()
    except Exception:
        X_bin = (np.asarray(X) > 0).astype(int)
    term_index = {t: i for i, t in enumerate(feature_names)}
    scores = []
    for terms in topics:
        inds = [term_index[t] for t in terms if t in term_index]
        if len(inds) < 2:
            continue
        topic_scores = []
        for m in range(1, len(inds)):
            wi = inds[m]
            col_i = X_bin[:, wi]
            for l in range(0, m):
                wj = inds[l]
                col_j = X_bin[:, wj]
                try:
                    d_j = float(col_j.sum())
                    d_ij = float(col_i.multiply(col_j).sum())
                except Exception:
                    d_j = float(np.sum(col_j))
                    d_ij = float(np.sum(col_i * col_j))
                topic_scores.append(math.log((d_ij + eps) / max(eps, d_j)))
        if topic_scores:
            scores.append(float(np.mean(topic_scores)))
    return float(np.mean(scores)) if scores else float("nan")


def choose_topic_segments(segments: Sequence[Segment], unit: str, min_lemma_tokens: int) -> list[Segment]:
    chosen = [seg for seg in segments if seg.unit == unit and len(seg.lemma_text.split()) >= min_lemma_tokens]
    if not chosen and unit != "document":
        chosen = [seg for seg in segments if seg.unit == "document" and len(seg.lemma_text.split()) >= min_lemma_tokens]
    return chosen


def run_lda(
    segments: Sequence[Segment],
    segment_scores: pd.DataFrame,
    args: argparse.Namespace,
    dirs: Mapping[str, Path],
    artifacts: RunArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    topics_path = dirs["tables"] / "lda_topics.tsv"
    doc_topics_path = dirs["tables"] / "lda_segment_topics.tsv"
    summary_path = dirs["tables"] / "lda_topic_summary.tsv"
    if should_resume(args) and outputs_exist([topics_path, doc_topics_path, summary_path]):
        get_progress(args).stage_skip("LDA topic model", f"loaded {doc_topics_path}")
        artifacts.tables["lda_topics"] = topics_path
        artifacts.tables["lda_segment_topics"] = doc_topics_path
        artifacts.tables["lda_topic_summary"] = summary_path
        diag_path = dirs["tables"] / "lda_diagnostics.tsv"
        if diag_path.exists():
            artifacts.tables["lda_diagnostics"] = diag_path
        return read_table_if_exists(topics_path), read_table_if_exists(doc_topics_path), read_table_if_exists(summary_path)

    progress = get_progress(args)
    progress.stage_start("LDA topic model", f"unit={args.lda_unit}, grid={args.lda_grid or args.lda_topics}")
    sk = import_sklearn(artifacts)
    if sk is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    CountVectorizer = sk["CountVectorizer"]
    LatentDirichletAllocation = sk["LatentDirichletAllocation"]
    joblib = sk["joblib"]

    topic_segments = choose_topic_segments(segments, args.lda_unit, args.min_topic_lemmas)
    if len(topic_segments) < 2:
        artifacts.add_warning(f"Not enough {args.lda_unit} segments with at least {args.min_topic_lemmas} lemmas for LDA.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    texts = [seg.lemma_text for seg in topic_segments]
    min_df: int | float = args.lda_min_df
    if isinstance(min_df, int) and min_df > len(texts):
        min_df = 1
    vectorizer = CountVectorizer(
        analyzer=str.split,
        min_df=min_df,
        max_df=args.lda_max_df,
        max_features=args.lda_max_features if args.lda_max_features > 0 else None,
    )
    try:
        X = vectorizer.fit_transform(texts)
    except Exception as exc:
        artifacts.add_warning(f"LDA vectorization failed: {exc}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if X.shape[1] < 5:
        artifacts.add_warning(f"LDA skipped because the vocabulary is too small after filtering ({X.shape[1]} terms).")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    feature_names = vectorizer.get_feature_names_out()
    topic_grid = [int(x) for x in str(args.lda_grid or args.lda_topics).split(",") if str(x).strip()]
    if args.lda_topics not in topic_grid:
        topic_grid.insert(0, args.lda_topics)
    topic_grid = sorted(set(max(2, x) for x in topic_grid if x < max(2, X.shape[0])))
    if not topic_grid:
        topic_grid = [min(args.lda_topics, max(2, X.shape[0] - 1))]
    diagnostics = []
    fitted_models = {}
    for n_topics in topic_grid:
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method=args.lda_learning_method,
            max_iter=args.lda_max_iter,
            random_state=args.random_state,
            evaluate_every=-1,
            n_jobs=args.lda_n_jobs,
        )
        doc_topic = lda.fit_transform(X)
        top_terms = []
        for topic_idx, topic_weights in enumerate(lda.components_):
            inds = topic_weights.argsort()[::-1][: args.lda_top_words]
            top_terms.append([feature_names[i] for i in inds])
        coherence = umass_coherence(X, feature_names, top_terms)
        try:
            perplexity = float(lda.perplexity(X))
        except Exception:
            perplexity = np.nan
        diagnostics.append({"n_topics": n_topics, "perplexity": perplexity, "umass_coherence": coherence})
        fitted_models[n_topics] = (lda, doc_topic, top_terms)
        progress.progress("LDA topic model", len(diagnostics), len(topic_grid), force=True, detail=f"n_topics={n_topics}, perplexity={perplexity:.3f}, UMass={coherence:.3f}")
    diag_df = pd.DataFrame(diagnostics)
    diag_path = dirs["tables"] / "lda_diagnostics.tsv"
    diag_df.to_csv(diag_path, sep="\t", index=False)
    artifacts.tables["lda_diagnostics"] = diag_path

    selected_n = args.lda_topics
    if args.lda_auto_select and not diag_df.empty:
        # Higher UMass is better. Use perplexity as tiebreaker.
        tmp = diag_df.copy()
        tmp["umass_coherence"] = pd.to_numeric(tmp["umass_coherence"], errors="coerce")
        tmp["perplexity"] = pd.to_numeric(tmp["perplexity"], errors="coerce")
        tmp = tmp.sort_values(["umass_coherence", "perplexity"], ascending=[False, True])
        selected_n = int(tmp.iloc[0]["n_topics"])
    elif selected_n not in fitted_models:
        selected_n = topic_grid[0]
    lda, doc_topic, top_terms = fitted_models[selected_n]

    topic_rows = []
    for topic_idx, topic_weights in enumerate(lda.components_):
        inds = topic_weights.argsort()[::-1][: args.lda_top_words]
        label = " / ".join(feature_names[i] for i in inds[:5])
        for rank, i in enumerate(inds, start=1):
            topic_rows.append(
                {
                    "topic_id": topic_idx,
                    "topic_label": label,
                    "rank": rank,
                    "term": feature_names[i],
                    "weight": float(topic_weights[i]),
                    "n_topics_selected": selected_n,
                }
            )
    topics_df = pd.DataFrame(topic_rows)
    topics_path = dirs["tables"] / "lda_topics.tsv"
    topics_df.to_csv(topics_path, sep="\t", index=False)
    artifacts.tables["lda_topics"] = topics_path

    doc_rows = []
    for seg, probs in zip(topic_segments, doc_topic):
        dominant = int(np.argmax(probs))
        row = {
            "segment_id": seg.segment_id,
            "unit": seg.unit,
            "doc_id": seg.doc_id,
            "dominant_lda_topic": dominant,
            "dominant_lda_topic_score": float(probs[dominant]),
            "lemma_token_count": len(seg.lemma_text.split()),
        }
        for i, p in enumerate(probs):
            row[f"lda_topic_{i}"] = float(p)
        row.update(seg.metadata)
        doc_rows.append(row)
    doc_topics_df = pd.DataFrame(doc_rows)
    doc_topics_path = dirs["tables"] / "lda_segment_topics.tsv"
    doc_topics_df.to_csv(doc_topics_path, sep="\t", index=False)
    artifacts.tables["lda_segment_topics"] = doc_topics_path

    summary_rows = []
    topic_labels = topics_df.groupby("topic_id")["topic_label"].first().to_dict() if not topics_df.empty else {}
    for topic_id in range(selected_n):
        probs = doc_topics_df[f"lda_topic_{topic_id}"]
        row = {
            "topic_id": topic_id,
            "topic_label": topic_labels.get(topic_id, ""),
            "mean_probability": float(probs.mean()),
            "dominant_segment_count": int((doc_topics_df["dominant_lda_topic"] == topic_id).sum()),
        }
        if not segment_scores.empty:
            merged = doc_topics_df[["segment_id", f"lda_topic_{topic_id}"]].merge(segment_scores, on="segment_id", how="left")
            w = pd.to_numeric(merged[f"lda_topic_{topic_id}"], errors="coerce").fillna(0).to_numpy(dtype=float)
            for emo in EMOTIONS:
                col = f"emotion_{emo}_per_1k"
                if col in merged.columns:
                    vals = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=float)
                    mask = ~np.isnan(vals)
                    row[f"weighted_{col}"] = float(np.average(vals[mask], weights=w[mask])) if mask.any() and w[mask].sum() > 0 else np.nan
        summary_rows.append(row)
    topic_summary_df = pd.DataFrame(summary_rows)
    summary_path = dirs["tables"] / "lda_topic_summary.tsv"
    topic_summary_df.to_csv(summary_path, sep="\t", index=False)
    artifacts.tables["lda_topic_summary"] = summary_path

    if args.save_models:
        model_path = dirs["models"] / "lda_model.joblib"
        vectorizer_path = dirs["models"] / "lda_vectorizer.joblib"
        joblib.dump(lda, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        artifacts.models["lda_model"] = model_path
        artifacts.models["lda_vectorizer"] = vectorizer_path
    progress.stage_done("LDA topic model", f"selected {selected_n} topics; {len(doc_topics_df):,} segment-topic rows")
    return topics_df, doc_topics_df, topic_summary_df


def detect_device(device_arg: str) -> int:
    if device_arg == "cpu":
        return -1
    if device_arg.startswith("cuda"):
        if ":" in device_arg:
            return int(device_arg.split(":", 1)[1])
        return 0
    if device_arg == "auto":
        try:
            import torch  # type: ignore
            return 0 if torch.cuda.is_available() else -1
        except Exception:
            return -1
    try:
        return int(device_arg)
    except Exception:
        return -1


def choose_transformer_segments(segments: Sequence[Segment], unit: str, min_words: int) -> list[Segment]:
    chosen = [seg for seg in segments if seg.unit == unit and seg.text and len(seg.text.split()) >= min_words]
    if not chosen and unit != "window":
        chosen = [seg for seg in segments if seg.unit == "window" and seg.text and len(seg.text.split()) >= min_words]
    if not chosen:
        chosen = [seg for seg in segments if seg.unit == "document" and seg.text]
    return chosen


def run_transformer_topic_classifier(
    segments: Sequence[Segment],
    args: argparse.Namespace,
    dirs: Mapping[str, Path],
    artifacts: RunArtifacts,
) -> pd.DataFrame:
    pred_path = dirs["tables"] / "transformer_topic_predictions.tsv"
    agg_path = dirs["tables"] / "transformer_topic_document_summary.tsv"
    if should_resume(args) and pred_path.exists() and pred_path.stat().st_size > 0:
        get_progress(args).stage_skip("transformer topic classifier", f"loaded {pred_path}")
        artifacts.tables["transformer_topic_predictions"] = pred_path
        if agg_path.exists() and agg_path.stat().st_size > 0:
            artifacts.tables["transformer_topic_document_summary"] = agg_path
        return read_table_if_exists(pred_path)

    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:
        artifacts.add_warning(f"transformers is not installed; transformer topic classifier skipped. Details: {exc}")
        return pd.DataFrame()
    chosen = choose_transformer_segments(segments, args.transformer_unit, args.transformer_min_words)
    if args.transformer_max_segments and args.transformer_max_segments > 0 and len(chosen) > args.transformer_max_segments:
        artifacts.add_warning(f"Transformer classifier limited from {len(chosen):,} to {args.transformer_max_segments:,} segments by --transformer-max-segments.")
        chosen = chosen[: args.transformer_max_segments]
    if not chosen:
        artifacts.add_warning("No segments available for transformer topic classification.")
        return pd.DataFrame()

    progress = get_progress(args)
    progress.stage_start("transformer topic classifier", f"{len(chosen):,} {args.transformer_unit} segments, batch={args.transformer_batch_size}, shard={args.transformer_shard_size}")
    device = detect_device(args.transformer_device)
    try:
        clf = pipeline("text-classification", model=args.topic_classifier_model, tokenizer=args.topic_classifier_model, device=device)
    except Exception as exc:
        artifacts.add_warning(f"Could not load transformer topic classifier {args.topic_classifier_model!r}; skipped. Details: {exc}")
        return pd.DataFrame()

    ckpt = checkpoint_dirs(args, dirs)["transformer_classifier"]
    shard_size = max(1, int(args.transformer_shard_size))
    batch_size = max(1, int(args.transformer_batch_size))
    total_shards = math.ceil(len(chosen) / shard_size)
    parts: list[Path] = []

    for shard_idx, start in enumerate(range(0, len(chosen), shard_size), start=1):
        shard = chosen[start:start + shard_size]
        part_path = ckpt / f"transformer_topic_predictions.part-{shard_idx:06d}.tsv"
        done_marker = ckpt / f"transformer_classifier.part-{shard_idx:06d}.done.json"
        if should_resume(args) and done_marker.exists() and part_path.exists() and part_path.stat().st_size > 0:
            parts.append(part_path)
            progress.progress("transformer topic classifier", min(start + len(shard), len(chosen)), len(chosen), force=True, detail=f"resumed shard {shard_idx}/{total_shards}")
            continue

        rows: list[dict[str, Any]] = []
        for batch_start in range(0, len(shard), batch_size):
            batch = shard[batch_start:batch_start + batch_size]
            texts = [seg.text[: args.transformer_max_chars] for seg in batch]
            try:
                outputs = clf(texts, truncation=True, max_length=args.transformer_max_length, top_k=args.transformer_top_k)
            except TypeError:
                outputs = clf(texts, truncation=True, max_length=args.transformer_max_length, return_all_scores=True)
            except RuntimeError as exc:
                # CUDA OOM recovery: clear cache, retry individually on CPU if allowed.
                try:
                    import torch  # type: ignore
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                artifacts.add_warning(f"Transformer classifier failed on shard {shard_idx}, local batch {batch_start}; skipped batch. Details: {exc}")
                continue
            except Exception as exc:
                artifacts.add_warning(f"Transformer classifier failed on shard {shard_idx}, local batch {batch_start}; skipped batch. Details: {exc}")
                continue
            if isinstance(outputs, dict):
                outputs = [outputs]
            for seg, output in zip(batch, outputs):
                pred_list = [output] if isinstance(output, dict) else output
                if pred_list and isinstance(pred_list[0], list):
                    pred_list = pred_list[0]
                pred_list = sorted(pred_list, key=lambda x: float(x.get("score", 0)), reverse=True)[: args.transformer_top_k]
                for rank, pred in enumerate(pred_list, start=1):
                    row = {
                        "segment_id": seg.segment_id,
                        "unit": seg.unit,
                        "doc_id": seg.doc_id,
                        "rank": rank,
                        "label": pred.get("label", ""),
                        "score": float(pred.get("score", 0.0)),
                        "model": args.topic_classifier_model,
                        "text_preview": seg.text[:200].replace("\n", " ").replace("\t", " "),
                    }
                    row.update(seg.metadata)
                    rows.append(row)
            global_done = start + min(batch_start + len(batch), len(shard))
            progress.progress("transformer topic classifier", global_done, len(chosen))

        pd.DataFrame(rows).to_csv(part_path, sep="\t", index=False)
        parts.append(part_path)
        write_json(done_marker, {"stage": "transformer_classifier", "shard": shard_idx, "start": start, "count": len(shard), "finished_at": now_stamp()})
        progress.progress("transformer topic classifier", min(start + len(shard), len(chosen)), len(chosen), force=True, detail=f"wrote shard {shard_idx}/{total_shards}")
        del rows
        gc.collect()

    concat_tsv_files(parts, pred_path)
    pred_df = read_table_if_exists(pred_path)
    if not pred_df.empty:
        artifacts.tables["transformer_topic_predictions"] = pred_path
        doc_agg = aggregate_transformer_predictions(pred_df)
        doc_agg.to_csv(agg_path, sep="\t", index=False)
        artifacts.tables["transformer_topic_document_summary"] = agg_path
    progress.stage_done("transformer topic classifier", dataframe_headroom_note(len(pred_df), len(pred_df.columns)))
    return pred_df

def aggregate_transformer_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    top = pred_df[pred_df["rank"] == 1].copy()
    counts = top.groupby(["doc_id", "label"], as_index=False).agg(
        segment_count=("segment_id", "count"),
        mean_top_score=("score", "mean"),
    )
    # Dominant label per doc by count, score tie-breaker.
    counts = counts.sort_values(["doc_id", "segment_count", "mean_top_score"], ascending=[True, False, False])
    dominant = counts.groupby("doc_id", as_index=False).first().rename(
        columns={"label": "dominant_transformer_topic", "segment_count": "dominant_segment_count"}
    )
    wide = top.pivot_table(index="doc_id", columns="label", values="score", aggfunc="mean", fill_value=0).reset_index()
    wide.columns = [str(c) if c == "doc_id" else f"transformer_topic_score::{c}" for c in wide.columns]
    return dominant.merge(wide, on="doc_id", how="left")


def sentence_transformer_embeddings(texts: list[str], model_name: str, batch_size: int, device: str, artifacts: RunArtifacts) -> np.ndarray | None:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    try:
        kwargs: dict[str, Any] = {}
        if device != "auto":
            kwargs["device"] = "cpu" if device == "cpu" else device
        model = SentenceTransformer(model_name, **kwargs)
        emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        return np.asarray(emb)
    except Exception as exc:
        artifacts.add_warning(f"SentenceTransformer embedding failed for {model_name!r}: {exc}")
        return None


def hf_mean_pool_embeddings(texts: list[str], model_name: str, batch_size: int, device_arg: str, max_length: int, artifacts: RunArtifacts) -> np.ndarray | None:
    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as exc:
        artifacts.add_warning(f"transformers/torch unavailable for HF embeddings. Details: {exc}")
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as exc:
        artifacts.add_warning(f"Could not load embedding model {model_name!r}: {exc}")
        return None
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_arg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(device_arg if device_arg.startswith("cuda") else "cpu")
    model.to(device)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.cpu().numpy())
            if artifacts and len(outputs) % 20 == 0:
                print(f"  embedded {min(start + batch_size, len(texts)):,}/{len(texts):,} segments", file=sys.stderr)
    return np.vstack(outputs) if outputs else None


def load_embedding_backend(model_name: str, backend: str, device_arg: str, artifacts: RunArtifacts) -> tuple[str, Any] | tuple[None, None]:
    """Load an embedding backend once for checkpointed/sharded encoding."""
    if backend in {"auto", "sentence-transformers"}:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            kwargs: dict[str, Any] = {}
            if device_arg != "auto":
                kwargs["device"] = "cpu" if device_arg == "cpu" else device_arg
            return "sentence-transformers", SentenceTransformer(model_name, **kwargs)
        except Exception as exc:
            if backend == "sentence-transformers":
                artifacts.add_warning(f"Could not load SentenceTransformer {model_name!r}: {exc}")
                return None, None
            artifacts.add_warning(f"SentenceTransformer backend unavailable for {model_name!r}; trying HF AutoModel. Details: {exc}")
    if backend in {"auto", "hf-auto"}:
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            if device_arg == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif device_arg == "cpu":
                device = torch.device("cpu")
            else:
                device = torch.device(device_arg if str(device_arg).startswith("cuda") else "cpu")
            model.to(device)
            model.eval()
            return "hf-auto", {"torch": torch, "tokenizer": tokenizer, "model": model, "device": device}
        except Exception as exc:
            artifacts.add_warning(f"Could not load HF embedding backend {model_name!r}: {exc}")
            return None, None
    return None, None


def encode_texts_with_backend(backend_name: str, backend_obj: Any, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
    if backend_name == "sentence-transformers":
        emb = backend_obj.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(emb, dtype="float32")
    if backend_name == "hf-auto":
        torch = backend_obj["torch"]
        tokenizer = backend_obj["tokenizer"]
        model = backend_obj["model"]
        device = backend_obj["device"]
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                hidden = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                outputs.append(pooled.cpu().numpy().astype("float32"))
        return np.vstack(outputs) if outputs else np.zeros((0, 0), dtype="float32")
    raise ValueError(f"Unsupported embedding backend: {backend_name}")


def write_embedding_segment_manifest(segments: Sequence[Segment], path: Path) -> None:
    rows = []
    for i, seg in enumerate(segments):
        rows.append({
            "embedding_index": i,
            "segment_id": seg.segment_id,
            "unit": seg.unit,
            "doc_id": seg.doc_id,
            "lemma_token_count": len(seg.lemma_text.split()),
            "text_preview": seg.text[:200].replace("\n", " ").replace("\t", " "),
            **seg.metadata,
        })
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def compute_embedding_shards(
    chosen: Sequence[Segment],
    model_name: str,
    backend: str,
    args: argparse.Namespace,
    stage_dir: Path,
    artifacts: RunArtifacts,
    stage_name: str,
) -> tuple[list[Path], Path]:
    """Encode text segments into resumable .npy shards and a segment manifest."""
    progress = get_progress(args)
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = stage_dir / "segments.tsv"
    meta_path = stage_dir / "segments.meta.json"
    fp_src = "\n".join([model_name, backend, str(args.transformer_max_chars), str(args.transformer_max_length)] + [seg.segment_id for seg in chosen])
    fingerprint = hashlib.sha256(fp_src.encode("utf-8", errors="replace")).hexdigest()
    old_meta = read_json(meta_path)
    resume_ok = should_resume(args) and old_meta.get("fingerprint") == fingerprint
    if should_resume(args) and old_meta and not resume_ok:
        artifacts.add_warning(
            f"{stage_name} embedding checkpoint segment/model fingerprint changed; old embedding shards in {stage_dir} will be ignored. "
            "Use --force to delete/recompute explicitly if desired."
        )
    if not manifest_path.exists() or not resume_ok:
        write_embedding_segment_manifest(chosen, manifest_path)
        write_json(meta_path, {
            "fingerprint": fingerprint,
            "model_name": model_name,
            "backend": backend,
            "segment_count": len(chosen),
            "transformer_max_chars": args.transformer_max_chars,
            "transformer_max_length": args.transformer_max_length,
            "written_at": now_stamp(),
        })
    shard_size = max(1, int(args.embedding_shard_size))
    total_shards = math.ceil(len(chosen) / shard_size) if chosen else 0
    parts: list[Path] = []

    backend_name, backend_obj = load_embedding_backend(model_name, backend, args.transformer_device, artifacts)
    if backend_name is None:
        return [], manifest_path

    progress.stage_start(stage_name + " embeddings", f"{len(chosen):,} segments, model={model_name}, shard={shard_size}, batch={args.transformer_batch_size}")
    for shard_idx, start in enumerate(range(0, len(chosen), shard_size), start=1):
        shard = chosen[start:start + shard_size]
        emb_path = stage_dir / f"embeddings.part-{shard_idx:06d}.npy"
        done_marker = stage_dir / f"embeddings.part-{shard_idx:06d}.done.json"
        if resume_ok and done_marker.exists() and emb_path.exists() and emb_path.stat().st_size > 0:
            parts.append(emb_path)
            progress.progress(stage_name + " embeddings", min(start + len(shard), len(chosen)), len(chosen), force=True, detail=f"resumed shard {shard_idx}/{total_shards}")
            continue
        texts = [seg.text[: args.transformer_max_chars] for seg in shard]
        try:
            emb = encode_texts_with_backend(backend_name, backend_obj, texts, max(1, args.transformer_batch_size), args.transformer_max_length)
        except RuntimeError as exc:
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            artifacts.add_warning(f"{stage_name} embedding shard {shard_idx} failed; skipped. Details: {exc}")
            continue
        except Exception as exc:
            artifacts.add_warning(f"{stage_name} embedding shard {shard_idx} failed; skipped. Details: {exc}")
            continue
        np.save(emb_path, emb.astype("float32", copy=False))
        parts.append(emb_path)
        write_json(done_marker, {"stage": stage_name, "shard": shard_idx, "start": start, "count": len(shard), "shape": list(emb.shape), "finished_at": now_stamp()})
        progress.progress(stage_name + " embeddings", min(start + len(shard), len(chosen)), len(chosen), force=True, detail=f"wrote shard {shard_idx}/{total_shards}")
        del emb, texts
        gc.collect()
    progress.stage_done(stage_name + " embeddings", f"{len(parts):,}/{total_shards:,} shards")
    return parts, manifest_path


def iter_embedding_shards(paths: Sequence[Path]) -> Iterator[np.ndarray]:
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            yield np.load(path, mmap_mode="r")


def load_embedding_shards(paths: Sequence[Path]) -> np.ndarray | None:
    arrays = [np.asarray(arr, dtype="float32") for arr in iter_embedding_shards(paths)]
    if not arrays:
        return None
    return np.vstack(arrays).astype("float32", copy=False)


def embedding_total_shape(paths: Sequence[Path]) -> tuple[int, int]:
    n = 0
    dim = 0
    for arr in iter_embedding_shards(paths):
        if arr.ndim == 2:
            n += int(arr.shape[0])
            dim = int(arr.shape[1])
    return n, dim


def estimated_embedding_mb(paths: Sequence[Path]) -> float:
    n, dim = embedding_total_shape(paths)
    return n * dim * 4 / (1024 ** 2)

def run_transformer_embeddings(
    segments: Sequence[Segment],
    segment_scores: pd.DataFrame,
    args: argparse.Namespace,
    dirs: Mapping[str, Path],
    artifacts: RunArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_path = dirs["tables"] / "transformer_embedding_clusters.tsv"
    desc_path = dirs["tables"] / "transformer_cluster_summary.tsv"
    if should_resume(args) and cluster_path.exists() and cluster_path.stat().st_size > 0:
        get_progress(args).stage_skip("transformer embeddings + clustering", f"loaded {cluster_path}")
        artifacts.tables["transformer_embedding_clusters"] = cluster_path
        if desc_path.exists() and desc_path.stat().st_size > 0:
            artifacts.tables["transformer_cluster_summary"] = desc_path
        return read_table_if_exists(cluster_path), read_table_if_exists(desc_path)

    sk = import_sklearn(artifacts)
    if sk is None:
        return pd.DataFrame(), pd.DataFrame()
    chosen = choose_transformer_segments(segments, args.embedding_unit, args.transformer_min_words)
    if args.embedding_max_segments and args.embedding_max_segments > 0 and len(chosen) > args.embedding_max_segments:
        artifacts.add_warning(f"Embedding clustering limited from {len(chosen):,} to {args.embedding_max_segments:,} segments by --embedding-max-segments.")
        chosen = chosen[: args.embedding_max_segments]
    if len(chosen) < 3:
        artifacts.add_warning("Too few segments for transformer embedding clustering.")
        return pd.DataFrame(), pd.DataFrame()

    progress = get_progress(args)
    progress.stage_start("transformer embeddings + clustering", f"{len(chosen):,} {args.embedding_unit} segments")
    ckpt = checkpoint_dirs(args, dirs)["embeddings"]
    emb_paths, manifest_path = compute_embedding_shards(
        chosen,
        args.embedding_model,
        args.embedding_backend,
        args,
        ckpt,
        artifacts,
        "transformer",
    )
    if not emb_paths:
        artifacts.add_warning("Transformer embeddings skipped because no embedding backend worked.")
        return pd.DataFrame(), pd.DataFrame()
    artifacts.models["transformer_embedding_manifest"] = manifest_path
    n_emb, dim = embedding_total_shape(emb_paths)
    emb_mb = estimated_embedding_mb(emb_paths)
    progress.progress("transformer embeddings + clustering", n_emb, len(chosen), force=True, detail=f"embedding matrix approx {emb_mb:,.1f}MB, dim={dim}")

    MiniBatchKMeans = sk["MiniBatchKMeans"]
    PCA = sk["PCA"]
    TfidfVectorizer = sk["TfidfVectorizer"]
    silhouette_score = sk["silhouette_score"]

    n_clusters = args.embedding_clusters
    if n_clusters <= 0:
        n_clusters = max(2, min(30, int(math.sqrt(n_emb))))
    n_clusters = max(2, min(n_clusters, max(2, n_emb - 1)))
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=args.random_state, batch_size=min(2048, max(64, args.embedding_shard_size)))

    # Partial-fit KMeans over shards to avoid loading the entire matrix when not necessary.
    for pass_i in range(max(1, args.embedding_kmeans_passes)):
        seen = 0
        for arr in iter_embedding_shards(emb_paths):
            if arr.shape[0] >= n_clusters:
                km.partial_fit(np.asarray(arr, dtype="float32"))
            seen += arr.shape[0]
        progress.progress("transformer embeddings + clustering", seen, n_emb, force=True, detail=f"kmeans pass {pass_i + 1}/{max(1, args.embedding_kmeans_passes)}")

    # Sample for PCA and silhouette. Full PCA over all windows is often pointless and expensive.
    sample_max = max(0, int(args.embedding_pca_max_segments))
    sample_arrays: list[np.ndarray] = []
    sample_needed = sample_max if sample_max > 0 else min(n_emb, 50000)
    for arr in iter_embedding_shards(emb_paths):
        if sum(a.shape[0] for a in sample_arrays) >= sample_needed:
            break
        remaining = sample_needed - sum(a.shape[0] for a in sample_arrays)
        sample_arrays.append(np.asarray(arr[:remaining], dtype="float32"))
    sample_emb = np.vstack(sample_arrays) if sample_arrays else None
    pca = None
    if sample_emb is not None and sample_emb.shape[0] >= 2 and sample_emb.shape[1] >= 2:
        try:
            pca = PCA(n_components=2, random_state=args.random_state).fit(sample_emb)
        except Exception as exc:
            artifacts.add_warning(f"Could not fit PCA for embedding plot coordinates: {exc}")
    sil = np.nan
    if sample_emb is not None and len(sample_emb) > n_clusters and args.embedding_silhouette_sample != 0:
        try:
            sil_sample = sample_emb
            if args.embedding_silhouette_sample > 0 and len(sil_sample) > args.embedding_silhouette_sample:
                sil_sample = sil_sample[: args.embedding_silhouette_sample]
            sil_labels = km.predict(sil_sample)
            sil = float(silhouette_score(sil_sample, sil_labels)) if len(set(sil_labels)) > 1 else np.nan
        except Exception:
            sil = np.nan

    # Write cluster rows streaming by shard.
    if cluster_path.exists() and not should_resume(args):
        cluster_path.unlink()
    rows_written = 0
    labels_all: list[int] = []
    chosen_offset = 0
    fieldnames: list[str] | None = None
    for arr in iter_embedding_shards(emb_paths):
        emb = np.asarray(arr, dtype="float32")
        labels = km.predict(emb)
        labels_all.extend(int(x) for x in labels)
        coords = None
        if pca is not None:
            try:
                coords = pca.transform(emb)
            except Exception:
                coords = None
        rows: list[dict[str, Any]] = []
        for local_i, (label, seg) in enumerate(zip(labels, chosen[chosen_offset:chosen_offset + len(labels)])):
            xy = coords[local_i] if coords is not None else (np.nan, np.nan)
            row = {
                "segment_id": seg.segment_id,
                "unit": seg.unit,
                "doc_id": seg.doc_id,
                "transformer_cluster": int(label),
                "pca_x": float(xy[0]) if not pd.isna(xy[0]) else np.nan,
                "pca_y": float(xy[1]) if not pd.isna(xy[1]) else np.nan,
                "embedding_model": args.embedding_model,
                "silhouette_overall": sil,
                "text_preview": seg.text[:200].replace("\n", " ").replace("\t", " "),
            }
            row.update(seg.metadata)
            rows.append(row)
        if rows:
            if fieldnames is None:
                fieldnames = list(rows[0].keys())
            write_tsv_stream(cluster_path, rows, fieldnames=fieldnames, append=rows_written > 0)
            rows_written += len(rows)
        chosen_offset += len(labels)
        progress.progress("transformer embeddings + clustering", rows_written, n_emb)
        del emb, rows
        gc.collect()
    artifacts.tables["transformer_embedding_clusters"] = cluster_path
    clusters_df = read_table_if_exists(cluster_path)

    desc_rows = []
    try:
        lemma_texts = [seg.lemma_text for seg in chosen[: len(labels_all)]]
        vectorizer = TfidfVectorizer(analyzer=str.split, min_df=1, max_features=args.cluster_top_term_features if args.cluster_top_term_features > 0 else None)
        X = vectorizer.fit_transform(lemma_texts)
        terms = np.asarray(vectorizer.get_feature_names_out())
        labels_np = np.asarray(labels_all)
        for cl in sorted(set(labels_all)):
            mask = labels_np == cl
            mean_scores = np.asarray(X[mask].mean(axis=0)).ravel()
            top_idx = mean_scores.argsort()[::-1][: args.cluster_top_terms]
            label_terms = [terms[i] for i in top_idx if mean_scores[i] > 0]
            row = {
                "transformer_cluster": int(cl),
                "cluster_label": " / ".join(label_terms[:5]),
                "segment_count": int(mask.sum()),
                "top_terms": ", ".join(label_terms),
                "embedding_model": args.embedding_model,
            }
            if not segment_scores.empty and not clusters_df.empty:
                seg_ids = clusters_df.loc[clusters_df["transformer_cluster"].astype(str) == str(cl), ["segment_id"]]
                merged = seg_ids.merge(segment_scores, on="segment_id", how="left")
                for emo in EMOTIONS:
                    col = f"emotion_{emo}_per_1k"
                    if col in merged.columns:
                        row[f"mean_{col}"] = pd.to_numeric(merged[col], errors="coerce").mean()
            desc_rows.append(row)
    except Exception as exc:
        artifacts.add_warning(f"Could not compute transformer cluster descriptors: {exc}")
    desc_df = pd.DataFrame(desc_rows)
    if not desc_df.empty:
        desc_df.to_csv(desc_path, sep="\t", index=False)
        artifacts.tables["transformer_cluster_summary"] = desc_path
    progress.stage_done("transformer embeddings + clustering", dataframe_headroom_note(len(clusters_df), len(clusters_df.columns)))
    return clusters_df, desc_df

def run_bertopic(
    segments: Sequence[Segment],
    segment_scores: pd.DataFrame,
    args: argparse.Namespace,
    dirs: Mapping[str, Path],
    artifacts: RunArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run BERTopic with checkpointed embeddings and resumable final tables.

    BERTopic fitting itself is not internally resumable, but the expensive
    embedding step is. If final BERTopic tables already exist and --resume is
    enabled, the stage is skipped. Use --bertopic-max-segments or document-level
    units on limited RAM.
    """
    topics_path = dirs["tables"] / "bertopic_topics.tsv"
    seg_topics_path = dirs["tables"] / "bertopic_segment_topics.tsv"
    summary_path = dirs["tables"] / "bertopic_topic_summary.tsv"
    if should_resume(args) and seg_topics_path.exists() and seg_topics_path.stat().st_size > 0:
        get_progress(args).stage_skip("BERTopic", f"loaded {seg_topics_path}")
        if topics_path.exists():
            artifacts.tables["bertopic_topics"] = topics_path
        artifacts.tables["bertopic_segment_topics"] = seg_topics_path
        if summary_path.exists():
            artifacts.tables["bertopic_topic_summary"] = summary_path
        return read_table_if_exists(topics_path), read_table_if_exists(seg_topics_path), read_table_if_exists(summary_path)

    try:
        from bertopic import BERTopic  # type: ignore
    except Exception as exc:
        artifacts.add_warning(f"bertopic is not installed; BERTopic skipped. Details: {exc}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    sk = import_sklearn(artifacts)
    if sk is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    CountVectorizer = sk["CountVectorizer"]
    MiniBatchKMeans = sk.get("MiniBatchKMeans")

    chosen = choose_topic_segments(segments, args.bertopic_unit, args.bertopic_min_words)
    if len(chosen) < 5:
        artifacts.add_warning(f"Too few {args.bertopic_unit} segments with at least {args.bertopic_min_words} lemmas for BERTopic.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if args.bertopic_max_segments and args.bertopic_max_segments > 0 and len(chosen) > args.bertopic_max_segments:
        artifacts.add_warning(
            f"BERTopic limited from {len(chosen):,} to {args.bertopic_max_segments:,} segments by --bertopic-max-segments."
        )
        chosen = chosen[: args.bertopic_max_segments]

    progress = get_progress(args)
    progress.stage_start("BERTopic", f"{len(chosen):,} {args.bertopic_unit} segments")
    lemma_docs = [seg.lemma_text for seg in chosen]
    embedding_model_name = args.bertopic_embedding_model or args.embedding_model
    ckpt = checkpoint_dirs(args, dirs)["bertopic"]
    emb_paths, manifest_path = compute_embedding_shards(
        chosen,
        embedding_model_name,
        args.embedding_backend,
        args,
        ckpt,
        artifacts,
        "BERTopic",
    )
    if not emb_paths:
        artifacts.add_warning("BERTopic skipped because transformer embeddings could not be computed.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    artifacts.models["bertopic_embedding_manifest"] = manifest_path
    emb_mb = estimated_embedding_mb(emb_paths)
    if emb_mb > args.embedding_full_load_max_mb:
        artifacts.add_warning(
            f"BERTopic needs the full embedding matrix in memory ({emb_mb:,.1f}MB estimated), which exceeds "
            f"--embedding-full-load-max-mb={args.embedding_full_load_max_mb}. Skipping BERTopic fit; embeddings are checkpointed. "
            "Increase the limit, reduce --bertopic-max-segments, or use --bertopic-unit document."
        )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    emb = load_embedding_shards(emb_paths)
    if emb is None:
        artifacts.add_warning("BERTopic skipped because no embedding shards could be loaded.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if args.save_models and not args.no_save_embedding_matrices:
        emb_path = dirs["models"] / "bertopic_segment_embeddings.npy"
        np.save(emb_path, emb.astype("float32", copy=False))
        artifacts.models["bertopic_segment_embeddings"] = emb_path

    def build_vectorizer(min_df_override: int | float | None = None, max_df_override: int | float | None = None):
        min_df_value: int | float = (
            min_df_override
            if min_df_override is not None
            else (args.bertopic_min_df if args.bertopic_min_df > 0 else args.lda_min_df)
        )
        if isinstance(min_df_value, int) and min_df_value > len(lemma_docs):
            min_df_value = 1
        max_df_value: int | float = (
            max_df_override
            if max_df_override is not None
            else (args.bertopic_max_df if args.bertopic_max_df > 0 else args.lda_max_df)
        )
        return CountVectorizer(
            analyzer=str.split,
            min_df=min_df_value,
            max_df=max_df_value,
            max_features=args.bertopic_max_features if args.bertopic_max_features > 0 else None,
        )

    vectorizer_model = build_vectorizer()

    cluster_model = None
    if getattr(args, "bertopic_cluster_model", "hdbscan") == "kmeans":
        if MiniBatchKMeans is None:
            artifacts.add_warning("--bertopic-cluster-model=kmeans requested, but MiniBatchKMeans is unavailable; falling back to BERTopic default clustering.")
        else:
            requested_k = max(2, int(args.bertopic_kmeans_clusters))
            k = min(requested_k, max(2, len(chosen) // max(1, int(args.bertopic_min_topic_size))))
            k = min(k, len(chosen))
            if k < requested_k:
                artifacts.add_warning(f"BERTopic KMeans clusters reduced from {requested_k} to {k} for {len(chosen):,} segments and min_topic_size={args.bertopic_min_topic_size}.")
            cluster_model = MiniBatchKMeans(
                n_clusters=k,
                random_state=args.random_state,
                batch_size=min(4096, max(256, args.embedding_shard_size)),
                n_init=10,
            )
            artifacts.add_warning(f"BERTopic using forced MiniBatchKMeans clustering with k={k}; this avoids HDBSCAN giant-topic collapse in broad corpora.")

    nr_topics: int | str | None
    raw_nr_topics = str(args.bertopic_nr_topics).strip().lower()
    if raw_nr_topics in {"", "none", "null", "0"}:
        nr_topics = None
    elif raw_nr_topics == "auto":
        nr_topics = "auto"
    else:
        try:
            nr_topics = max(2, int(raw_nr_topics))
        except ValueError:
            artifacts.add_warning(f"Invalid --bertopic-nr-topics={args.bertopic_nr_topics!r}; using 'auto'.")
            nr_topics = "auto"

    kwargs: dict[str, Any] = {
        "language": args.bertopic_language,
        "embedding_model": None,
        "vectorizer_model": vectorizer_model,
        "min_topic_size": args.bertopic_min_topic_size,
        "nr_topics": nr_topics,
        "top_n_words": args.bertopic_top_words,
        "calculate_probabilities": args.bertopic_calculate_probabilities,
        "verbose": args.verbose,
    }
    if cluster_model is not None:
        kwargs["hdbscan_model"] = cluster_model
    if getattr(args, "bertopic_skip_dim_reduction", False):
        try:
            from bertopic.dimensionality import BaseDimensionalityReduction  # type: ignore
            kwargs["umap_model"] = BaseDimensionalityReduction()
            artifacts.add_warning("BERTopic dimensionality reduction skipped with BaseDimensionalityReduction; clustering is performed on original embedding space.")
        except Exception as exc:
            artifacts.add_warning(f"--bertopic-skip-dim-reduction requested but unavailable; falling back to BERTopic default dimensionality reduction. Details: {exc}")
    if args.bertopic_low_memory:
        kwargs["low_memory"] = True
    try:
        topic_model = BERTopic(**kwargs)
    except TypeError:
        kwargs.pop("low_memory", None)
        topic_model = BERTopic(**kwargs)

    progress.progress("BERTopic", 0, len(chosen), force=True, detail="fitting model; this substep cannot be resumed mid-fit")
    try:
        topics, probs = topic_model.fit_transform(lemma_docs, embeddings=emb)
    except MemoryError as exc:
        artifacts.add_warning(
            "BERTopic ran out of memory during fit. The embedding shards remain in checkpoints. "
            "Rerun with --resume plus a lower --bertopic-max-segments, --bertopic-unit document, smaller --bertopic-max-features, or --bertopic-low-memory."
        )
        raise
    except Exception as exc:
        msg = str(exc)
        if "max_df corresponds" in msg and "min_df" in msg:
            artifacts.add_warning(
                "BERTopic c-TF-IDF vectorizer failed because the number of discovered topic-documents was too small for "
                f"min_df={args.bertopic_min_df} and max_df={args.bertopic_max_df}. Retrying once with min_df=1, max_df=1.0. "
                "If this still produces only a few topics, rerun with --bertopic-cluster-model kmeans."
            )
            try:
                kwargs["vectorizer_model"] = build_vectorizer(min_df_override=1, max_df_override=1.0)
                try:
                    topic_model = BERTopic(**kwargs)
                except TypeError:
                    kwargs.pop("low_memory", None)
                    topic_model = BERTopic(**kwargs)
                topics, probs = topic_model.fit_transform(lemma_docs, embeddings=emb)
            except Exception as exc2:
                artifacts.add_warning(f"BERTopic fitting failed after safe vectorizer retry: {exc2}")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        else:
            artifacts.add_warning(f"BERTopic fitting failed: {exc}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    finally:
        del emb
        gc.collect()
    progress.progress("BERTopic", len(chosen), len(chosen), force=True, detail="fit complete")

    topic_info = topic_model.get_topic_info()
    topic_info = topic_info.copy() if topic_info is not None else pd.DataFrame()
    if not topic_info.empty:
        topic_info.columns = [normalize_col(c) for c in topic_info.columns]
        for col in topic_info.columns:
            topic_info[col] = topic_info[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, tuple, dict)) else x)
        if "topic" in topic_info.columns:
            topic_info = topic_info.rename(columns={"topic": "topic_id"})
        if "name" in topic_info.columns:
            topic_info = topic_info.rename(columns={"name": "topic_label"})
        path = dirs["tables"] / "bertopic_topic_info.tsv"
        topic_info.to_csv(path, sep="\t", index=False)
        artifacts.tables["bertopic_topic_info"] = path

    word_rows = []
    for topic_id in sorted(set(int(t) for t in topics)):
        words = topic_model.get_topic(topic_id) or []
        label = " / ".join(str(w) for w, _ in words[:5])
        for rank, (term, weight) in enumerate(words[: args.bertopic_top_words], start=1):
            word_rows.append(
                {
                    "topic_id": topic_id,
                    "topic_label": label,
                    "rank": rank,
                    "term": term,
                    "weight": float(weight),
                    "is_outlier_topic": topic_id == -1,
                    "model": "bertopic",
                    "embedding_model": embedding_model_name,
                }
            )
    topics_df = pd.DataFrame(word_rows)
    if not topics_df.empty:
        topics_df.to_csv(topics_path, sep="\t", index=False)
        artifacts.tables["bertopic_topics"] = topics_path

    prob_values: list[float | None] = [None] * len(chosen)
    try:
        if probs is not None:
            arr = np.asarray(probs)
            if arr.ndim == 1 and len(arr) == len(chosen):
                prob_values = [float(x) if not pd.isna(x) else None for x in arr]
            elif arr.ndim == 2 and arr.shape[0] == len(chosen):
                prob_values = [float(np.nanmax(row)) if row.size else None for row in arr]
    except Exception:
        prob_values = [None] * len(chosen)

    rows = []
    for seg, topic_id, prob in zip(chosen, topics, prob_values):
        row = {
            "segment_id": seg.segment_id,
            "unit": seg.unit,
            "doc_id": seg.doc_id,
            "bertopic_topic": int(topic_id),
            "bertopic_is_outlier": int(topic_id) == -1,
            "bertopic_probability": prob,
            "embedding_model": embedding_model_name,
            "text_preview": seg.text[:200].replace("\n", " ").replace("\t", " "),
        }
        row.update(seg.metadata)
        rows.append(row)
    segment_topics_df = pd.DataFrame(rows)
    segment_topics_df.to_csv(seg_topics_path, sep="\t", index=False)
    artifacts.tables["bertopic_segment_topics"] = seg_topics_path

    label_map = topics_df.groupby("topic_id")["topic_label"].first().to_dict() if not topics_df.empty else {}
    summary_rows = []
    for topic_id, g in segment_topics_df.groupby("bertopic_topic"):
        row = {
            "topic_id": int(topic_id),
            "topic_label": label_map.get(int(topic_id), ""),
            "segment_count": int(len(g)),
            "doc_count": int(g["doc_id"].nunique()) if "doc_id" in g.columns else 0,
            "is_outlier_topic": int(topic_id) == -1,
            "mean_probability": pd.to_numeric(g.get("bertopic_probability", pd.Series(dtype=float)), errors="coerce").mean(),
        }
        if not segment_scores.empty:
            merged = g[["segment_id"]].merge(segment_scores, on="segment_id", how="left")
            for emo in EMOTIONS:
                col = f"emotion_{emo}_per_1k"
                if col in merged.columns:
                    row[f"mean_{col}"] = pd.to_numeric(merged[col], errors="coerce").mean()
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("segment_count", ascending=False) if summary_rows else pd.DataFrame()
    if not summary_df.empty:
        summary_df.to_csv(summary_path, sep="\t", index=False)
        artifacts.tables["bertopic_topic_summary"] = summary_path

    if args.save_models:
        model_dir = dirs["models"] / "bertopic_model"
        try:
            if model_dir.exists():
                if model_dir.is_dir():
                    shutil.rmtree(model_dir)
                else:
                    model_dir.unlink()
            topic_model.save(str(model_dir), serialization="safetensors", save_ctfidf=True)
        except TypeError:
            try:
                topic_model.save(str(model_dir))
            except Exception as exc:
                artifacts.add_warning(f"Could not save BERTopic model: {exc}")
            else:
                artifacts.models["bertopic_model"] = model_dir
        except Exception as exc:
            artifacts.add_warning(f"Could not save BERTopic model: {exc}")
        else:
            artifacts.models["bertopic_model"] = model_dir
    progress.stage_done("BERTopic", dataframe_headroom_note(len(segment_topics_df), len(segment_topics_df.columns)))
    return topics_df, segment_topics_df, summary_df

def run_crosswalks(
    segment_scores: pd.DataFrame,
    lda_topics: pd.DataFrame,
    lda_segment_topics: pd.DataFrame,
    transformer_predictions: pd.DataFrame,
    transformer_clusters: pd.DataFrame,
    bertopic_topics: pd.DataFrame,
    bertopic_segment_topics: pd.DataFrame,
    dirs: Mapping[str, Path],
    artifacts: RunArtifacts,
) -> None:
    if not segment_scores.empty and not lda_segment_topics.empty:
        merged = lda_segment_topics.merge(segment_scores, on="segment_id", how="left", suffixes=("", "_emotion"))
        rows = []
        topic_cols = [c for c in lda_segment_topics.columns if c.startswith("lda_topic_")]
        labels = lda_topics.groupby("topic_id")["topic_label"].first().to_dict() if not lda_topics.empty else {}
        for tc in topic_cols:
            topic_id = int(tc.split("_")[-1])
            weights = pd.to_numeric(merged[tc], errors="coerce").fillna(0).to_numpy(dtype=float)
            row = {"topic_family": "lda", "topic_id": topic_id, "topic_label": labels.get(topic_id, ""), "weight_sum": float(weights.sum())}
            for emo in EMOTIONS:
                col = f"emotion_{emo}_per_1k"
                if col in merged.columns:
                    vals = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=float)
                    mask = ~np.isnan(vals)
                    row[col] = float(np.average(vals[mask], weights=weights[mask])) if mask.any() and weights[mask].sum() > 0 else np.nan
            rows.append(row)
        df = pd.DataFrame(rows)
        path = dirs["tables"] / "lda_emotion_crosswalk.tsv"
        df.to_csv(path, sep="\t", index=False)
        artifacts.tables["lda_emotion_crosswalk"] = path

    if not segment_scores.empty and not transformer_predictions.empty:
        top = transformer_predictions[transformer_predictions["rank"] == 1].copy()
        merged = top.merge(segment_scores, on="segment_id", how="left", suffixes=("", "_emotion"))
        rows = []
        for label, g in merged.groupby("label"):
            row = {"topic_family": "transformer_classifier", "topic_label": label, "segment_count": len(g), "mean_classifier_score": pd.to_numeric(g["score"], errors="coerce").mean()}
            for emo in EMOTIONS:
                col = f"emotion_{emo}_per_1k"
                if col in g.columns:
                    row[col] = pd.to_numeric(g[col], errors="coerce").mean()
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("segment_count", ascending=False)
        path = dirs["tables"] / "transformer_topic_emotion_crosswalk.tsv"
        df.to_csv(path, sep="\t", index=False)
        artifacts.tables["transformer_topic_emotion_crosswalk"] = path

    if not segment_scores.empty and not transformer_clusters.empty:
        merged = transformer_clusters.merge(segment_scores, on="segment_id", how="left", suffixes=("", "_emotion"))
        rows = []
        for cluster, g in merged.groupby("transformer_cluster"):
            row = {"topic_family": "transformer_cluster", "cluster_id": int(cluster), "segment_count": len(g)}
            for emo in EMOTIONS:
                col = f"emotion_{emo}_per_1k"
                if col in g.columns:
                    row[col] = pd.to_numeric(g[col], errors="coerce").mean()
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("segment_count", ascending=False)
        path = dirs["tables"] / "transformer_cluster_emotion_crosswalk.tsv"
        df.to_csv(path, sep="\t", index=False)
        artifacts.tables["transformer_cluster_emotion_crosswalk"] = path

    if not segment_scores.empty and not bertopic_segment_topics.empty:
        merged = bertopic_segment_topics.merge(segment_scores, on="segment_id", how="left", suffixes=("", "_emotion"))
        rows = []
        labels = bertopic_topics.groupby("topic_id")["topic_label"].first().to_dict() if not bertopic_topics.empty else {}
        for topic_id, g in merged.groupby("bertopic_topic"):
            row = {
                "topic_family": "bertopic",
                "topic_id": int(topic_id),
                "topic_label": labels.get(int(topic_id), ""),
                "segment_count": int(len(g)),
                "is_outlier_topic": int(topic_id) == -1,
            }
            for emo in EMOTIONS:
                col = f"emotion_{emo}_per_1k"
                if col in g.columns:
                    row[col] = pd.to_numeric(g[col], errors="coerce").mean()
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("segment_count", ascending=False)
        path = dirs["tables"] / "bertopic_emotion_crosswalk.tsv"
        df.to_csv(path, sep="	", index=False)
        artifacts.tables["bertopic_emotion_crosswalk"] = path



def _time_sort_dataframe_for_plot(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Return a copy sorted safely for time-like metadata values.

    Wikivir metadata can mix integers and strings in the same column, e.g. 19,
    "19", "19. stol.", "unknown".  Pandas cannot sort such mixed object
    columns directly on Python 3, so plotting needs an explicit numeric/string
    sort key.
    """
    if df.empty or time_col not in df.columns:
        return df.copy()

    out = df.copy()
    labels = out[time_col].astype(str).replace({"nan": "", "None": ""})

    def numeric_key(value: Any) -> float:
        if value is None:
            return float("nan")
        if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
            return float(value)
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "unknown", "_"}:
            return float("nan")
        # Prefer an explicit 3/4-digit year where present; otherwise century-like numbers.
        m = re.search(r"(?<!\d)(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})(?!\d)", text)
        if m:
            return float(m.group(1))
        m = re.search(r"(?<!\d)([0-9]{1,2})(?:\.|\s|$)", text)
        if m:
            return float(m.group(1))
        m = re.search(r"-?\d+(?:[\.,]\d+)?", text)
        if m:
            return float(m.group(0).replace(",", "."))
        return float("nan")

    out["__time_label"] = labels
    out["__time_numeric"] = [numeric_key(v) for v in out[time_col].tolist()]
    out["__time_is_missing_numeric"] = pd.to_numeric(out["__time_numeric"], errors="coerce").isna()
    out = out.sort_values(["__time_is_missing_numeric", "__time_numeric", "__time_label"], kind="mergesort")
    return out

def plot_outputs(
    doc_scores: pd.DataFrame,
    segment_scores: pd.DataFrame,
    lda_topic_summary: pd.DataFrame,
    transformer_predictions: pd.DataFrame,
    transformer_clusters: pd.DataFrame,
    bertopic_topic_summary: pd.DataFrame,
    dirs: Mapping[str, Path],
    artifacts: RunArtifacts,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        artifacts.add_warning(f"matplotlib not available; plots skipped. Details: {exc}")
        return

    if not doc_scores.empty:
        # Corpus emotion profile.
        vals = []
        for emo in EMOTIONS:
            col = f"emotion_{emo}_count"
            vals.append(float(pd.to_numeric(doc_scores.get(col, pd.Series(dtype=float)), errors="coerce").sum()))
        if sum(vals) > 0:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(EMOTIONS, vals)
            ax.set_title("Corpus emotion profile: SloEmoLex counts")
            ax.set_ylabel("Weighted lexicon count")
            ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
            path = dirs["plots"] / "emotion_profile_counts.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            artifacts.plots["emotion_profile_counts"] = path

        # VAD by time if year or century exists.
        for time_col in ("year", "century", "period"):
            if time_col in doc_scores.columns and doc_scores[time_col].replace("", np.nan).dropna().nunique() > 1:
                group = weighted_group_summary(doc_scores, time_col)
                if not group.empty and "vad_valence_mean" in group.columns:
                    try:
                        group = _time_sort_dataframe_for_plot(group, time_col)
                        xlabels = group.get("__time_label", group[time_col].astype(str)).astype(str)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(xlabels, group["vad_valence_mean"], marker="o", label="valence")
                        if "vad_arousal_mean" in group.columns:
                            ax.plot(xlabels, group["vad_arousal_mean"], marker="o", label="arousal")
                        if "vad_dominance_mean" in group.columns:
                            ax.plot(xlabels, group["vad_dominance_mean"], marker="o", label="dominance")
                        ax.set_title(f"VAD by {time_col}")
                        ax.set_ylabel("Mean score over lexicon hits")
                        ax.tick_params(axis="x", rotation=35)
                        ax.legend()
                        fig.tight_layout()
                        path = dirs["plots"] / f"vad_by_{time_col}.png"
                        fig.savefig(path, dpi=160)
                        plt.close(fig)
                        artifacts.plots[f"vad_by_{time_col}"] = path
                    except Exception as exc:
                        artifacts.add_warning(f"VAD-by-{time_col} plot skipped after plotting error: {exc}")
                        try:
                            plt.close("all")
                        except Exception:
                            pass
                    break

    if not lda_topic_summary.empty and "mean_probability" in lda_topic_summary.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = lda_topic_summary["topic_label"].astype(str).str.slice(0, 35)
        ax.bar(range(len(lda_topic_summary)), lda_topic_summary["mean_probability"])
        ax.set_xticks(range(len(lda_topic_summary)))
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_title("LDA topic prevalence")
        ax.set_ylabel("Mean topic probability")
        fig.tight_layout()
        path = dirs["plots"] / "lda_topic_prevalence.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        artifacts.plots["lda_topic_prevalence"] = path

    if not transformer_predictions.empty:
        top = transformer_predictions[transformer_predictions["rank"] == 1]
        counts = top["label"].value_counts().head(20)
        if not counts.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(counts)), counts.values)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index.astype(str), rotation=60, ha="right")
            ax.set_title("Transformer topic classifier: top labels")
            ax.set_ylabel("Segment count")
            fig.tight_layout()
            path = dirs["plots"] / "transformer_topic_labels.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            artifacts.plots["transformer_topic_labels"] = path

    if not transformer_clusters.empty and {"pca_x", "pca_y", "transformer_cluster"}.issubset(transformer_clusters.columns):
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(transformer_clusters["pca_x"], transformer_clusters["pca_y"], c=transformer_clusters["transformer_cluster"], s=18)
        ax.set_title("Transformer embedding clusters, PCA projection")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, label="cluster")
        fig.tight_layout()
        path = dirs["plots"] / "transformer_cluster_pca.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        artifacts.plots["transformer_cluster_pca"] = path

    if not bertopic_topic_summary.empty and "segment_count" in bertopic_topic_summary.columns:
        tmp = bertopic_topic_summary.copy()
        if "is_outlier_topic" in tmp.columns:
            tmp = tmp[~tmp["is_outlier_topic"].astype(bool)]
        tmp = tmp.sort_values("segment_count", ascending=False).head(30)
        if not tmp.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            labels = tmp.get("topic_label", pd.Series([""] * len(tmp))).astype(str).str.slice(0, 35)
            ax.bar(range(len(tmp)), tmp["segment_count"])
            ax.set_xticks(range(len(tmp)))
            ax.set_xticklabels(labels, rotation=60, ha="right")
            ax.set_title("BERTopic topic prevalence")
            ax.set_ylabel("Segment count")
            fig.tight_layout()
            path = dirs["plots"] / "bertopic_topic_prevalence.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            artifacts.plots["bertopic_topic_prevalence"] = path


def write_lexicon_coverage(
    lexicon: SloEmoLex | None,
    segments: Sequence[Segment],
    segment_scores: pd.DataFrame,
    dirs: Mapping[str, Path],
    artifacts: RunArtifacts,
) -> None:
    if lexicon is None:
        return
    all_topic_tokens = Counter()
    all_word_tokens = Counter()
    for seg in segments:
        if seg.unit != "document":
            continue
        for tok in seg.tokens:
            if tok.is_word:
                key = normalize_lexeme(tok.lemma if tok.lemma and tok.lemma != "_" else tok.form)
                all_word_tokens[key] += 1
                if tok.upos in DEFAULT_CONTENT_UPOS:
                    all_topic_tokens[key] += 1
    matched_keys = set(lexicon.lookup.keys())
    total = sum(all_word_tokens.values())
    hits = sum(c for key, c in all_word_tokens.items() if key in matched_keys)
    rows = [
        {"metric": "lexicon_rows", "value": lexicon.row_count},
        {"metric": "lexicon_keys", "value": lexicon.key_count},
        {"metric": "corpus_word_tokens", "value": total},
        {"metric": "corpus_word_token_hits", "value": hits},
        {"metric": "corpus_word_token_coverage", "value": hits / total if total else 0.0},
        {"metric": "lexicon_word_columns", "value": ",".join(lexicon.word_columns)},
        {"metric": "detected_emotion_columns", "value": json.dumps(lexicon.emotion_columns, ensure_ascii=False)},
        {"metric": "detected_intensity_columns", "value": json.dumps(lexicon.intensity_columns, ensure_ascii=False)},
        {"metric": "detected_sentiment_columns", "value": json.dumps(lexicon.sentiment_columns, ensure_ascii=False)},
        {"metric": "detected_vad_columns", "value": json.dumps(lexicon.vad_columns, ensure_ascii=False)},
    ]
    df = pd.DataFrame(rows)
    path = dirs["tables"] / "lexicon_coverage.tsv"
    df.to_csv(path, sep="\t", index=False)
    artifacts.tables["lexicon_coverage"] = path

    oov_rows = []
    for key, count in all_topic_tokens.most_common(1000):
        if key not in matched_keys:
            oov_rows.append({"lemma": key, "count": count})
    if oov_rows:
        oov = pd.DataFrame(oov_rows)
        oov_path = dirs["tables"] / "top_oov_content_lemmas.tsv"
        oov.to_csv(oov_path, sep="\t", index=False)
        artifacts.tables["top_oov_content_lemmas"] = oov_path


def write_report(
    args: argparse.Namespace,
    docs: list[Document],
    doc_scores: pd.DataFrame,
    lda_topics: pd.DataFrame,
    lda_topic_summary: pd.DataFrame,
    bertopic_topic_summary: pd.DataFrame,
    transformer_predictions: pd.DataFrame,
    transformer_clusters: pd.DataFrame,
    artifacts: RunArtifacts,
    dirs: Mapping[str, Path],
) -> Path:
    lines = []
    lines.append("# Wikivir emotion and topic analysis report")
    lines.append("")
    lines.append(f"Generated: {GENERATED_AT}")
    lines.append("")
    lines.append("## Input")
    lines.append("")
    lines.append(f"- CoNLL-U: `{args.input}`")
    lines.append(f"- Documents parsed: **{len(docs):,}**")
    lines.append(f"- SloEmoLex: `{args.sloemolex or 'not used'}`")
    lines.append("")
    if artifacts.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in artifacts.warnings:
            lines.append(f"- {w}")
        lines.append("")
    if not doc_scores.empty:
        lines.append("## Corpus-level emotion profile")
        lines.append("")
        totals = []
        for emo in EMOTIONS:
            col = f"emotion_{emo}_count"
            totals.append((emo, float(pd.to_numeric(doc_scores.get(col, pd.Series(dtype=float)), errors="coerce").sum())))
        total_sum = sum(v for _, v in totals)
        lines.append("| emotion | count | proportion |")
        lines.append("|---|---:|---:|")
        for emo, val in sorted(totals, key=lambda x: x[1], reverse=True):
            prop = val / total_sum if total_sum else 0.0
            lines.append(f"| {emo} | {val:.2f} | {prop:.3f} |")
        if "lexicon_coverage" in doc_scores.columns:
            cov = pd.to_numeric(doc_scores["lexicon_coverage"], errors="coerce").mean()
            lines.append("")
            lines.append(f"Mean document lexicon coverage: **{cov:.3f}**")
        lines.append("")
    if not lda_topic_summary.empty:
        lines.append("## LDA topics")
        lines.append("")
        lines.append("| topic | label | mean probability | dominant segments |")
        lines.append("|---:|---|---:|---:|")
        for _, row in lda_topic_summary.sort_values("mean_probability", ascending=False).head(30).iterrows():
            lines.append(
                f"| {int(row['topic_id'])} | {row.get('topic_label', '')} | {float(row.get('mean_probability', 0)):.3f} | {int(row.get('dominant_segment_count', 0))} |"
            )
        lines.append("")
    if not bertopic_topic_summary.empty:
        lines.append("## BERTopic topics")
        lines.append("")
        lines.append("| topic | label | segments | documents | outlier |")
        lines.append("|---:|---|---:|---:|---:|")
        tmp = bertopic_topic_summary.sort_values("segment_count", ascending=False).head(30)
        for _, row in tmp.iterrows():
            lines.append(
                f"| {int(row['topic_id'])} | {row.get('topic_label', '')} | {int(row.get('segment_count', 0))} | {int(row.get('doc_count', 0))} | {int(bool(row.get('is_outlier_topic', False)))} |"
            )
        lines.append("")
    if not transformer_predictions.empty:
        lines.append("## Transformer topic classifier")
        lines.append("")
        top = transformer_predictions[transformer_predictions["rank"] == 1]
        counts = top["label"].value_counts().head(20)
        lines.append("| label | segment count |")
        lines.append("|---|---:|")
        for label, count in counts.items():
            lines.append(f"| {label} | {int(count)} |")
        lines.append("")
    if not transformer_clusters.empty:
        lines.append("## Transformer embedding clusters")
        lines.append("")
        lines.append(f"Segments clustered: **{len(transformer_clusters):,}**")
        lines.append(f"Clusters: **{transformer_clusters['transformer_cluster'].nunique()}**")
        lines.append("")
    lines.append("## Output manifest")
    lines.append("")
    lines.append("### Tables")
    for name, path in sorted(artifacts.tables.items()):
        lines.append(f"- `{name}`: `{path.relative_to(dirs['base'])}`")
    lines.append("")
    lines.append("### Plots")
    for name, path in sorted(artifacts.plots.items()):
        lines.append(f"- `{name}`: `{path.relative_to(dirs['base'])}`")
    lines.append("")
    lines.append("### Models / arrays")
    for name, path in sorted(artifacts.models.items()):
        lines.append(f"- `{name}`: `{path.relative_to(dirs['base'])}`")
    report_path = dirs["base"] / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_manifest(
    args: argparse.Namespace,
    docs: list[Document],
    segments: Sequence[Segment],
    lexicon: SloEmoLex | None,
    artifacts: RunArtifacts,
    dirs: Mapping[str, Path],
) -> Path:
    manifest = {
        "generated_at": GENERATED_AT,
        "input": str(args.input),
        "input_sha256": sha256_file(Path(args.input)) if Path(args.input).exists() else None,
        "sloemolex": str(args.sloemolex) if args.sloemolex else None,
        "sloemolex_sha256": sha256_file(Path(args.sloemolex)) if args.sloemolex and Path(args.sloemolex).exists() else None,
        "documents": len(docs),
        "segments": len(segments),
        "lexicon": ({
            "word_columns": lexicon.word_columns,
            "emotion_columns": lexicon.emotion_columns,
            "intensity_columns": lexicon.intensity_columns,
            "sentiment_columns": lexicon.sentiment_columns,
            "vad_columns": lexicon.vad_columns,
            "row_count": lexicon.row_count,
            "key_count": lexicon.key_count,
        } if lexicon else None),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "tables": {k: str(v) for k, v in artifacts.tables.items()},
        "plots": {k: str(v) for k, v in artifacts.plots.items()},
        "models": {k: str(v) for k, v in artifacts.models.items()},
        "warnings": artifacts.warnings,
    }
    # Lexicon lookup is intentionally not dumped into the manifest.
    path = dirs["base"] / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_basic_corpus_tables(docs: list[Document], segments: Sequence[Segment], doc_meta: pd.DataFrame, dirs: Mapping[str, Path], artifacts: RunArtifacts) -> None:
    doc_rows = []
    for doc in docs:
        toks = doc.tokens
        word_toks = [t for t in toks if t.is_word]
        row = {
            "doc_id": doc.doc_id,
            "sentence_count": len(doc.sentences),
            "token_count": len(toks),
            "word_token_count": len(word_toks),
            "type_count_forms": len({normalize_lexeme(t.form) for t in word_toks}),
            "type_count_lemmas": len({normalize_lexeme(t.lemma if t.lemma != '_' else t.form) for t in word_toks}),
            **doc.metadata,
        }
        doc_rows.append(row)
    doc_df = pd.DataFrame(doc_rows)
    if not doc_meta.empty:
        meta_cols = [c for c in doc_meta.columns if c not in doc_df.columns or c == "doc_id"]
        if len(meta_cols) > 1:
            doc_df = doc_df.merge(doc_meta[meta_cols], on="doc_id", how="left")
    path = dirs["tables"] / "documents.tsv"
    doc_df.to_csv(path, sep="\t", index=False)
    artifacts.tables["documents"] = path

    meta_path = dirs["tables"] / "document_metadata.tsv"
    meta_cols = [c for c in doc_df.columns if c not in {"sentence_count", "token_count", "word_token_count", "type_count_forms", "type_count_lemmas"}]
    doc_df[meta_cols].to_csv(meta_path, sep="\t", index=False)
    artifacts.tables["document_metadata"] = meta_path

    coverage_df = metadata_presence_summary(docs)
    coverage_path = dirs["tables"] / "metadata_coverage.tsv"
    coverage_df.to_csv(coverage_path, sep="\t", index=False)
    artifacts.tables["metadata_coverage"] = coverage_path

    seg_df = segments_to_frame(segments)
    path = dirs["tables"] / "segments.tsv"
    seg_df.to_csv(path, sep="\t", index=False)
    artifacts.tables["segments"] = path

    # UPOS/deprel overview.
    upos = Counter(t.upos for doc in docs for t in doc.tokens)
    deprel = Counter(t.deprel for doc in docs for t in doc.tokens if t.deprel and t.deprel != "_")
    pd.DataFrame([{"upos": k, "count": v} for k, v in upos.most_common()]).to_csv(dirs["tables"] / "upos_counts.tsv", sep="\t", index=False)
    pd.DataFrame([{"deprel": k, "count": v} for k, v in deprel.most_common()]).to_csv(dirs["tables"] / "deprel_counts.tsv", sep="\t", index=False)
    artifacts.tables["upos_counts"] = dirs["tables"] / "upos_counts.tsv"
    artifacts.tables["deprel_counts"] = dirs["tables"] / "deprel_counts.tsv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emotion and topic analysis for CLASSLA-annotated Wikivir CoNLL-U corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path, help="CLASSLA-annotated CoNLL-U file.")
    p.add_argument("--metadata", type=Path, help="Optional EXTRA metadata CSV/TSV sidecar. Embedded CoNLL-U # comments are used by default and do not need this.")
    p.add_argument("--metadata-doc-id-column", help="Sidecar metadata doc ID column if it cannot be inferred.")
    p.add_argument("--require-embedded-metadata", action="store_true", help="Fail if the CoNLL-U does not contain useful document metadata comments such as # title/# author/# genre.")
    p.add_argument("--sloemolex", type=Path, help="Path to SloEmoLex_v1.tsv or compatible emotion lexicon.")
    p.add_argument("--sloemolex-word-columns", help="Comma-separated lexicon word columns to use for matching.")
    p.add_argument("--allow-missing-lexicon", action="store_true", help="Continue without SloEmoLex; only topic outputs are produced.")
    p.add_argument("--output-dir", type=Path, default=Path("analysis/wikivir_emotion_topics"), help="Output directory.")

    p.set_defaults(resume=True)
    p.add_argument("--resume", dest="resume", action="store_true", help="Resume from completed stage outputs and shard checkpoints. Enabled by default.")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore checkpoints and recompute everything.")
    p.add_argument("--force", action="store_true", help="Recompute stages even when outputs/checkpoints exist.")
    p.add_argument("--checkpoint-dir", type=Path, help="Checkpoint directory. Default: <output-dir>/checkpoints.")
    p.add_argument("--progress-every", type=int, default=1000, help="Print progress every N units/batches where applicable.")
    p.add_argument("--quiet-progress", action="store_true", help="Suppress progress ticker lines, keeping warnings and final paths.")
    p.add_argument("--low-memory", action="store_true", help="Apply safer defaults for RAM-constrained runs: smaller batches, fewer features, capped BERTopic/embedding segments.")

    p.add_argument("--segment-levels", default="document,window,sentence", help="Comma-separated levels to build: document, paragraph, sentence, window.")
    p.add_argument("--window-size", type=int, default=250, help="Word-token window size for long documents.")
    p.add_argument("--window-step", type=int, default=125, help="Word-token step for rolling windows.")
    p.add_argument("--content-upos", default=",".join(sorted(DEFAULT_CONTENT_UPOS)), help="UPOS tags kept for topic modelling.")
    p.add_argument("--lexicon-upos", default=",".join(sorted(DEFAULT_LEXICON_UPOS)), help="UPOS tags matched to emotion lexicon.")
    p.add_argument("--stopwords", type=Path, help="Optional stopword file, one item per line.")
    p.add_argument("--extra-stopwords", help="Comma-separated extra stopwords.")
    p.add_argument("--normalization", choices=["lower", "simple", "historical", "ascii"], default="simple", help="Lexeme normalization mode.")
    p.add_argument("--strip-diacritics", action="store_true", help="Also match without Slovene diacritics.")
    p.add_argument("--match-on", default="lemma,form", help="Comma-separated lexicon matching sources: lemma, form, lemma_or_form.")
    p.add_argument("--fuzzy-lexicon", action="store_true", help="Use rapidfuzz fuzzy matching for OOV lexemes; expensive, off by default.")
    p.add_argument("--fuzzy-threshold", type=int, default=94, help="Fuzzy matching threshold, 0-100.")
    p.add_argument("--export-token-matches", action="store_true", help="Write token-level SloEmoLex matches; can be large.")
    p.add_argument("--lexicon-shard-size", type=int, default=5000, help="Segments per SloEmoLex checkpoint shard.")
    p.add_argument("--load-token-matches", action="store_true", help="Load token_emotion_matches.tsv back into memory after writing. Usually unnecessary and memory-heavy.")

    p.add_argument("--no-lda", action="store_true", help="Skip LDA topic modelling.")
    p.add_argument("--lda-unit", choices=["document", "paragraph", "sentence", "window"], default="document", help="Segment unit for LDA.")
    p.add_argument("--lda-topics", type=int, default=20, help="Number of LDA topics.")
    p.add_argument("--lda-grid", default="", help="Optional comma-separated topic numbers for diagnostics/auto-selection.")
    p.add_argument("--lda-auto-select", action="store_true", help="Select topic number by UMass coherence over --lda-grid.")
    p.add_argument("--lda-top-words", type=int, default=20, help="Top words per LDA topic.")
    p.add_argument("--lda-min-df", type=int, default=2, help="LDA CountVectorizer min_df. Lowered automatically for tiny corpora.")
    p.add_argument("--lda-max-df", type=float, default=0.85, help="LDA CountVectorizer max_df.")
    p.add_argument("--lda-max-features", type=int, default=50000, help="Maximum topic vocabulary size; 0 means unlimited.")
    p.add_argument("--lda-max-iter", type=int, default=30, help="LDA fitting iterations.")
    p.add_argument("--lda-learning-method", choices=["batch", "online"], default="batch", help="LDA learning method.")
    p.add_argument("--lda-n-jobs", type=int, default=-1, help="Parallel jobs for sklearn LDA.")
    p.add_argument("--min-topic-lemmas", type=int, default=30, help="Minimum lemma tokens in a segment for topic modelling.")

    p.add_argument("--run-bertopic", action="store_true", help="Run BERTopic transformer topic modelling with lemma-based c-TF-IDF.")
    p.add_argument("--bertopic-unit", choices=["document", "paragraph", "sentence", "window"], default="window", help="Segment unit for BERTopic.")
    p.add_argument("--bertopic-language", default="multilingual", help="BERTopic language hint.")
    p.add_argument("--bertopic-embedding-model", default="", help="Embedding model for BERTopic; defaults to --embedding-model.")
    p.add_argument("--bertopic-min-words", type=int, default=30, help="Minimum lemma tokens in a segment for BERTopic.")
    p.add_argument("--bertopic-min-topic-size", type=int, default=15, help="BERTopic HDBSCAN minimum topic size.")
    p.add_argument("--bertopic-nr-topics", default="auto", help="BERTopic topic reduction: auto, none/0, or integer.")
    p.add_argument("--bertopic-top-words", type=int, default=20, help="Top words per BERTopic topic.")
    p.add_argument("--bertopic-min-df", type=int, default=1, help="BERTopic CountVectorizer min_df over lemma texts.")
    p.add_argument("--bertopic-max-df", type=float, default=0.90, help="BERTopic CountVectorizer max_df over lemma texts.")
    p.add_argument("--bertopic-max-features", type=int, default=50000, help="BERTopic c-TF-IDF vocabulary cap; 0 means unlimited.")
    p.add_argument("--bertopic-calculate-probabilities", action="store_true", help="Ask BERTopic to calculate topic probabilities; more informative but slower/heavier.")
    p.add_argument("--bertopic-low-memory", action="store_true", help="Use BERTopic low-memory mode when supported.")
    p.add_argument("--bertopic-max-segments", type=int, default=0, help="Optional cap for BERTopic segments. 0 = all.")
    p.add_argument("--bertopic-cluster-model", choices=["hdbscan", "kmeans"], default="hdbscan", help="BERTopic clustering backend. Use kmeans for broad corpora when HDBSCAN collapses into one giant topic.")
    p.add_argument("--bertopic-kmeans-clusters", type=int, default=80, help="Number of forced BERTopic clusters when --bertopic-cluster-model=kmeans.")
    p.add_argument("--bertopic-skip-dim-reduction", action="store_true", help="Skip UMAP dimensionality reduction in BERTopic. Useful with KMeans on full large-window corpora; uses BERTopic's BaseDimensionalityReduction when available.")

    p.add_argument("--run-transformers", action="store_true", help="Run both transformer classifier and transformer embedding clustering.")
    p.add_argument("--run-topic-classifier", action="store_true", help="Run transformer topic classifier.")
    p.add_argument("--topic-classifier-model", default="cjvt/sloberta-trendi-topics", help="HF text-classification model for Slovene topics.")
    p.add_argument("--transformer-unit", choices=["document", "paragraph", "sentence", "window"], default="window", help="Segment unit for transformer topic classifier.")
    p.add_argument("--transformer-top-k", type=int, default=5, help="Top classifier labels to export.")
    p.add_argument("--transformer-device", default="auto", help="auto, cpu, cuda, cuda:0, or integer pipeline device.")
    p.add_argument("--transformer-batch-size", type=int, default=8, help="Transformer inference batch size.")
    p.add_argument("--transformer-max-length", type=int, default=512, help="Transformer tokenizer max length.")
    p.add_argument("--transformer-max-chars", type=int, default=6000, help="Hard character cap per transformer segment before tokenization.")
    p.add_argument("--transformer-min-words", type=int, default=20, help="Minimum words in segment for transformer inference.")
    p.add_argument("--transformer-shard-size", type=int, default=1000, help="Segments per transformer-classifier checkpoint shard.")
    p.add_argument("--transformer-max-segments", type=int, default=0, help="Optional cap for transformer classifier segments. 0 = all.")

    p.add_argument("--run-embeddings", action="store_true", help="Run transformer embeddings + clustering.")
    p.add_argument("--embedding-unit", choices=["document", "paragraph", "sentence", "window"], default="window", help="Segment unit for embeddings.")
    p.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="SentenceTransformer or HF AutoModel for embeddings.")
    p.add_argument("--embedding-backend", choices=["auto", "sentence-transformers", "hf-auto"], default="auto", help="Embedding backend.")
    p.add_argument("--embedding-clusters", type=int, default=0, help="KMeans clusters. 0 = automatic sqrt(n) cap 30.")
    p.add_argument("--cluster-top-terms", type=int, default=25, help="Top terms for each transformer cluster.")
    p.add_argument("--cluster-top-term-features", type=int, default=30000, help="Max TF-IDF features for cluster labels.")
    p.add_argument("--embedding-shard-size", type=int, default=1000, help="Segments per embedding .npy checkpoint shard.")
    p.add_argument("--embedding-max-segments", type=int, default=0, help="Optional cap for embedding clustering segments. 0 = all.")
    p.add_argument("--embedding-kmeans-passes", type=int, default=2, help="Number of MiniBatchKMeans partial-fit passes over embedding shards.")
    p.add_argument("--embedding-pca-max-segments", type=int, default=50000, help="Max segments used to fit PCA coordinates for plots. 0 = auto sample.")
    p.add_argument("--embedding-silhouette-sample", type=int, default=10000, help="Max embeddings for silhouette estimate. 0 = skip silhouette.")
    p.add_argument("--embedding-full-load-max-mb", type=float, default=2048.0, help="Maximum estimated embedding matrix size to load fully for BERTopic. Lower this on RAM-limited machines.")
    p.add_argument("--no-save-embedding-matrices", action="store_true", help="Do not copy full embedding matrices into models/; checkpoint shards are still kept.")

    p.add_argument("--make-plots", action="store_true", help="Create PNG plots.")
    p.add_argument("--save-models", action="store_true", help="Save fitted sklearn models/vectorizers.")
    p.add_argument("--max-docs", type=int, default=0, help="Debug limit: keep only first N documents. 0 = all.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--verbose", action="store_true", help="More progress messages.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = RunArtifacts()
    dirs = ensure_dirs(args.output_dir)
    apply_low_memory_defaults(args, artifacts)
    checkpoint_dirs(args, dirs)
    progress = get_progress(args)
    progress.stage_start("full Wikivir analysis", f"output={args.output_dir}, resume={args.resume}, low_memory={args.low_memory}")

    if not args.input.exists():
        raise FileNotFoundError(f"Input CoNLL-U not found: {args.input}")
    if args.sloemolex and not args.sloemolex.exists():
        if args.allow_missing_lexicon:
            artifacts.add_warning(f"SloEmoLex file not found: {args.sloemolex}; continuing without lexicon analysis.")
            args.sloemolex = None
        else:
            raise FileNotFoundError(
                f"SloEmoLex file not found: {args.sloemolex}. Download SloEmoLex_v1.tsv and pass --sloemolex, "
                "or add --allow-missing-lexicon to run topics only."
            )
    elif not args.sloemolex and not args.allow_missing_lexicon:
        raise FileNotFoundError("No --sloemolex supplied. Pass SloEmoLex_v1.tsv or use --allow-missing-lexicon for topics only.")

    if args.run_transformers:
        args.run_topic_classifier = True
        args.run_embeddings = True

    progress.stage_start("read CoNLL-U", str(args.input))
    docs = parse_conllu(args.input, artifacts)
    if args.max_docs and args.max_docs > 0:
        docs = docs[: args.max_docs]
    if not docs:
        raise ValueError("No documents parsed from input CoNLL-U.")
    progress.stage_done("read CoNLL-U", f"{len(docs):,} documents")
    if not has_useful_embedded_metadata(docs):
        message = (
            "No useful embedded document metadata found in CoNLL-U comments. "
            "Expected comments such as # title = ..., # author = ..., # century = ..., # genre = .... "
            "If annotation stripped them, restore them from the original Wikivir XML/CoNLL-U with "
            "scripts/restore_wikivir_metadata.py."
        )
        if args.require_embedded_metadata:
            raise ValueError(message)
        artifacts.add_warning(message)

    metadata_df = load_metadata(args.metadata, args.metadata_doc_id_column) if args.metadata else pd.DataFrame()
    doc_meta_df = merge_metadata(docs, metadata_df, artifacts)
    metadata_by_doc = dataframe_to_unique_doc_mapping(doc_meta_df, artifacts, "document metadata")

    content_upos = parse_upos_set(args.content_upos, DEFAULT_CONTENT_UPOS)
    stopwords = load_stopwords(args.stopwords, args.extra_stopwords)
    levels = {x.strip() for x in args.segment_levels.split(",") if x.strip()}
    # Ensure requested units are available for enabled models.
    levels.add("document")
    if not args.no_lda:
        levels.add(args.lda_unit)
    if args.run_topic_classifier:
        levels.add(args.transformer_unit)
    if args.run_embeddings:
        levels.add(args.embedding_unit)
    if args.run_bertopic:
        levels.add(args.bertopic_unit)
    if args.export_token_matches or "sentence" in levels:
        # Keep sentence-level scores useful for entity-emotion output.
        levels.add("sentence")
    progress.stage_start("build segments", ", ".join(sorted(levels)))
    segments = build_segments(
        docs,
        levels=levels,
        content_upos=content_upos,
        stopwords=stopwords,
        normalization=args.normalization,
        strip_accents=args.strip_diacritics,
        window_size=args.window_size,
        window_step=args.window_step,
        metadata_by_doc=metadata_by_doc,
    )
    progress.stage_done("build segments", f"{len(segments):,} segments")
    write_basic_corpus_tables(docs, segments, doc_meta_df, dirs, artifacts)

    lexicon: SloEmoLex | None = None
    segment_scores = pd.DataFrame()
    doc_scores = pd.DataFrame()
    if args.sloemolex:
        progress.stage_start("load SloEmoLex", str(args.sloemolex))
        word_cols = [x.strip() for x in args.sloemolex_word_columns.split(",")] if args.sloemolex_word_columns else None
        lexicon = SloEmoLex.load(
            args.sloemolex,
            word_columns=word_cols,
            normalization=args.normalization,
            strip_accents=args.strip_diacritics,
            artifacts=artifacts,
        )
        progress.stage_done("load SloEmoLex", f"{lexicon.key_count:,} normalized keys from {lexicon.row_count:,} rows")
        segment_scores, matches = run_lexicon_analysis(segments, lexicon, args, dirs, artifacts)
        doc_scores = aggregate_document_scores(segment_scores, dirs, artifacts)
        write_grouped_emotion_summaries(doc_scores, dirs, artifacts)
        write_lexicon_coverage(lexicon, segments, segment_scores, dirs, artifacts)

    entities = extract_named_entities(docs, dirs, artifacts)
    if lexicon and not entities.empty and not segment_scores.empty:
        write_entity_emotion_summary(entities, segment_scores, dirs, artifacts)

    lda_topics = pd.DataFrame()
    lda_segment_topics = pd.DataFrame()
    lda_topic_summary = pd.DataFrame()
    if not args.no_lda:
        lda_topics, lda_segment_topics, lda_topic_summary = run_lda(segments, segment_scores, args, dirs, artifacts)

    transformer_predictions = pd.DataFrame()
    if args.run_topic_classifier:
        transformer_predictions = run_transformer_topic_classifier(segments, args, dirs, artifacts)

    transformer_clusters = pd.DataFrame()
    transformer_cluster_summary = pd.DataFrame()
    if args.run_embeddings:
        transformer_clusters, transformer_cluster_summary = run_transformer_embeddings(segments, segment_scores, args, dirs, artifacts)

    bertopic_topics = pd.DataFrame()
    bertopic_segment_topics = pd.DataFrame()
    bertopic_topic_summary = pd.DataFrame()
    if args.run_bertopic:
        bertopic_topics, bertopic_segment_topics, bertopic_topic_summary = run_bertopic(segments, segment_scores, args, dirs, artifacts)

    run_crosswalks(segment_scores, lda_topics, lda_segment_topics, transformer_predictions, transformer_clusters, bertopic_topics, bertopic_segment_topics, dirs, artifacts)

    if args.make_plots:
        progress.stage_start("plots")
        try:
            plot_outputs(doc_scores, segment_scores, lda_topic_summary, transformer_predictions, transformer_clusters, bertopic_topic_summary, dirs, artifacts)
            progress.stage_done("plots")
        except Exception as exc:
            artifacts.add_warning(f"plots stage failed but analysis tables are preserved and report generation will continue. Details: {exc}")
            progress.stage_done("plots", "skipped after non-fatal plotting error")

    report_path = write_report(args, docs, doc_scores, lda_topics, lda_topic_summary, bertopic_topic_summary, transformer_predictions, transformer_clusters, artifacts, dirs)
    artifacts.tables["report_md"] = report_path
    manifest_path = write_manifest(args, docs, segments, lexicon, artifacts, dirs)
    progress.stage_done("full Wikivir analysis", f"Report: {report_path}; Manifest: {manifest_path}")
    print(f"Done. Report: {report_path}", file=sys.stderr)
    print(f"Manifest: {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
