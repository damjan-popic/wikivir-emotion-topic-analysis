#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$(pwd)}"
CONLLU="${CONLLU:-$PROJECT_DIR/data/annotated/wikivir-classla.clean.conllu}"
META="${META:-$PROJECT_DIR/data/metadata/wikivir.cleaned.metadata.tsv}"
OUT="${OUT:-$PROJECT_DIR/reports/qc/post_annotation}"
mkdir -p "$OUT"

if [[ ! -s "$CONLLU" ]]; then
  echo "ERROR: CoNLL-U not found or empty: $CONLLU" >&2
  exit 1
fi
if [[ ! -s "$META" ]]; then
  echo "ERROR: metadata TSV not found or empty: $META" >&2
  exit 1
fi

printf 'Input CoNLL-U: %s\n' "$CONLLU" | tee "$OUT/run.log"
printf 'Input metadata: %s\n' "$META" | tee -a "$OUT/run.log"
printf 'Output dir: %s\n\n' "$OUT" | tee -a "$OUT/run.log"

# Basic counts and duplicate IDs without loading the corpus in memory.
grep -c '^# newdoc id = ' "$CONLLU" > "$OUT/conllu_doc_count.txt" || true
grep -c '^# sent_id = ' "$CONLLU" > "$OUT/conllu_sent_count.txt" || true

grep '^# newdoc id = ' "$CONLLU" | sed 's/^# newdoc id = //' | sort | uniq -d > "$OUT/duplicate_newdoc_ids.txt"
grep '^# sent_id = ' "$CONLLU" | sed 's/^# sent_id = //' | sort | uniq -d > "$OUT/duplicate_sent_ids.txt"

grep '^# ' "$CONLLU" | sed -n 's/^# \([^=]*\) = .*/\1/p' | sort | uniq -c | sort -nr > "$OUT/conllu_comment_key_inventory.txt"

grep -E '^# (gerne|cemtury|genere|gendre|auther|titel|centruy|centry|yer|pubication) = ' "$CONLLU" > "$OUT/known_bad_metadata_keys.txt" || true

# Streaming Python checks.
python - "$CONLLU" "$META" "$OUT" <<'PY'
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import get_close_matches
from pathlib import Path

conllu = Path(sys.argv[1])
meta = Path(sys.argv[2])
out = Path(sys.argv[3])

expected_comment_keys = {
    "newdoc id", "newpar id", "sent_id", "text",
    "title", "author", "genre", "century", "year",
    "publication", "source", "url", "xml_doc_index", "source_xml",
    "category", "language",
}

summary = {
    "conllu": str(conllu),
    "metadata": str(meta),
    "checks": {},
    "warnings": [],
    "stop_reasons": [],
}

# Metadata TSV coverage and doc ids.
with meta.open("r", encoding="utf-8", errors="replace", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    meta_rows = list(reader)
    meta_cols = reader.fieldnames or []

summary["checks"]["metadata_rows"] = len(meta_rows)
summary["checks"]["metadata_columns"] = meta_cols

with (out / "metadata_column_coverage.tsv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["column", "nonempty", "total", "coverage"])
    for col in meta_cols:
        nonempty = sum(1 for r in meta_rows if (r.get(col, "") or "").strip())
        cov = nonempty / len(meta_rows) if meta_rows else 0
        w.writerow([col, nonempty, len(meta_rows), f"{cov:.6f}"])
        if col in {"title", "genre", "century", "author"} and cov < 0.75:
            summary["warnings"].append(f"low metadata coverage for {col}: {cov:.3f}")

meta_ids = [r.get("doc_id", "").strip() for r in meta_rows if r.get("doc_id", "").strip()]
summary["checks"]["metadata_doc_ids"] = len(meta_ids)
summary["checks"]["metadata_duplicate_doc_ids"] = sum(c-1 for c in Counter(meta_ids).values() if c > 1)

# Value sanity.
problems = []
for i, row in enumerate(meta_rows, start=2):
    doc_id = row.get("doc_id", "")
    title = row.get("title", "")
    for col in ["title", "author", "genre", "century", "year", "publication", "category", "language"]:
        if col in row:
            val = (row.get(col, "") or "").strip()
            if val and "\t" in val:
                problems.append([i, doc_id, title, col, val, "tab in value"])
    century = (row.get("century", "") or "").strip()
    if century:
        nums = re.findall(r"\d+", century)
        if not nums:
            problems.append([i, doc_id, title, "century", century, "no numeric century"])
        else:
            for n in nums:
                v = int(n)
                if not (8 <= v <= 21):
                    problems.append([i, doc_id, title, "century", century, "century outside expected 8-21"])
    year = (row.get("year", "") or "").strip()
    if year:
        nums = re.findall(r"\d{3,4}", year)
        if nums:
            for n in nums:
                v = int(n)
                if not (800 <= v <= 2100):
                    problems.append([i, doc_id, title, "year", year, "year outside expected 800-2100"])
        else:
            problems.append([i, doc_id, title, "year", year, "no 3-4 digit year"])

with (out / "metadata_value_problems.tsv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["row", "doc_id", "title", "column", "value", "problem"])
    w.writerows(problems)
summary["checks"]["metadata_value_problem_rows"] = len(problems)

# Metadata distributions.
for col in ["author", "genre", "century", "year", "publication", "category", "language"]:
    if col in meta_cols:
        ctr = Counter((r.get(col, "") or "<EMPTY>").strip() or "<EMPTY>" for r in meta_rows)
        with (out / f"metadata_distribution_{col}.tsv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow([col, "count"])
            for val, count in ctr.most_common():
                w.writerow([val, count])

# Stream CoNLL-U checks.
comment_keys = Counter()
conllu_ids = []
sent_ids = []
doc_meta = []
current_doc = None
current_meta = {}
current_sent_id = ""
sent_rows = []

bad_columns = []
dep_errors = []
empty_docs = []
doc_stats = []
current_doc_stats = None
upos = Counter()
deprel = Counter()
ner = Counter()
lemma_empty = xpos_empty = feats_empty = token_total = 0

def finish_doc():
    global current_doc_stats, current_meta
    if current_doc_stats is not None:
        doc_stats.append(current_doc_stats.copy())
        if current_doc_stats["tokens"] == 0 or current_doc_stats["sentences"] == 0:
            empty_docs.append(current_doc_stats.copy())
    if current_meta:
        doc_meta.append(current_meta.copy())

def finish_sent():
    global sent_rows, current_sent_id
    if not sent_rows:
        return
    int_ids = set()
    for p in sent_rows:
        tid = p[0]
        if "-" in tid or "." in tid:
            continue
        try:
            int_ids.add(int(tid))
        except ValueError:
            dep_errors.append([current_sent_id, "bad_token_id", tid, ""])
    roots = 0
    for p in sent_rows:
        tid, form, lemma, upos_, xpos, feats, head, deprel_, deps, misc = p
        if "-" in tid or "." in tid:
            continue
        try:
            h = int(head)
        except ValueError:
            dep_errors.append([current_sent_id, "bad_head", tid, head])
            continue
        if h == 0:
            roots += 1
            if deprel_ != "root":
                dep_errors.append([current_sent_id, "head0_nonroot_deprel", tid, deprel_])
        elif h not in int_ids:
            dep_errors.append([current_sent_id, "head_out_of_sentence", tid, head])
    if roots != 1:
        dep_errors.append([current_sent_id, "root_count_not_one", str(roots), ""])
    sent_rows = []
    current_sent_id = ""

with conllu.open("r", encoding="utf-8", errors="replace") as f:
    for lineno, line in enumerate(f, start=1):
        line = line.rstrip("\n")
        if not line:
            finish_sent()
            continue
        if line.startswith("# "):
            if " = " in line:
                key, val = line[2:].split(" = ", 1)
                key = key.strip()
                val = val.strip()
                comment_keys[key] += 1
                if key == "newdoc id":
                    finish_sent()
                    finish_doc()
                    current_doc = val
                    conllu_ids.append(val)
                    current_meta = {"doc_id": val}
                    current_doc_stats = {"doc_id": val, "title": "", "sentences": 0, "tokens": 0, "lexical_tokens": 0}
                elif key == "sent_id":
                    current_sent_id = val
                    sent_ids.append(val)
                    if current_doc_stats is not None:
                        current_doc_stats["sentences"] += 1
                elif current_meta is not None and key in {"title", "author", "genre", "century", "year", "publication", "category", "language", "source", "url"}:
                    current_meta[key] = val
                    if key == "title" and current_doc_stats is not None:
                        current_doc_stats["title"] = val
            continue
        p = line.split("\t")
        if len(p) != 10:
            if len(bad_columns) < 10000:
                bad_columns.append([lineno, len(p), line[:200]])
            continue
        if p[0].isdigit():
            sent_rows.append(p)
            token_total += 1
            if current_doc_stats is not None:
                current_doc_stats["tokens"] += 1
                if p[3] in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}:
                    current_doc_stats["lexical_tokens"] += 1
            upos[p[3]] += 1
            deprel[p[7]] += 1
            if p[2] == "_": lemma_empty += 1
            if p[4] == "_": xpos_empty += 1
            if p[5] == "_": feats_empty += 1
            for part in p[9].split("|"):
                if part.startswith("NER="):
                    ner[part.split("=", 1)[1]] += 1
finish_sent()
finish_doc()

summary["checks"].update({
    "conllu_docs": len(conllu_ids),
    "conllu_duplicate_doc_ids": sum(c-1 for c in Counter(conllu_ids).values() if c > 1),
    "conllu_sentences": len(sent_ids),
    "conllu_duplicate_sent_ids": sum(c-1 for c in Counter(sent_ids).values() if c > 1),
    "token_rows": token_total,
    "bad_column_rows_sampled": len(bad_columns),
    "dependency_error_rows": len(dep_errors),
    "empty_or_zero_sentence_docs": len(empty_docs),
    "lemma_empty_rate": lemma_empty / token_total if token_total else 0,
    "xpos_empty_rate": xpos_empty / token_total if token_total else 0,
    "feats_empty_rate": feats_empty / token_total if token_total else 0,
})

# Write outputs.
def write_counter(counter, path, col="value"):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([col, "count"])
        for k, v in counter.most_common():
            w.writerow([k, v])

write_counter(comment_keys, out / "conllu_comment_keys.tsv", "key")
write_counter(upos, out / "upos_distribution.tsv", "upos")
write_counter(deprel, out / "deprel_distribution.tsv", "deprel")
write_counter(ner, out / "ner_distribution.tsv", "ner")

unknown_keys = sorted(set(comment_keys) - expected_comment_keys)
with (out / "suspicious_metadata_keys.tsv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["key", "count", "close_expected_keys"])
    for key in unknown_keys:
        close = ", ".join(get_close_matches(key, expected_comment_keys, n=4, cutoff=0.65))
        w.writerow([key, comment_keys[key], close])
summary["checks"]["unknown_comment_keys"] = len(unknown_keys)
if unknown_keys:
    summary["warnings"].append(f"unknown comment keys present: {len(unknown_keys)}")

with (out / "bad_column_rows.tsv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["line", "columns", "text_sample"])
    w.writerows(bad_columns)

with (out / "dependency_integrity_errors.tsv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["sent_id", "problem", "value1", "value2"])
    w.writerows(dep_errors[:100000])

with (out / "conllu_document_metadata.tsv").open("w", encoding="utf-8", newline="") as f:
    fieldnames = sorted(set().union(*(r.keys() for r in doc_meta))) if doc_meta else ["doc_id"]
    if "doc_id" in fieldnames:
        fieldnames = ["doc_id"] + [c for c in fieldnames if c != "doc_id"]
    w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
    w.writeheader()
    w.writerows(doc_meta)

with (out / "document_size_distribution.tsv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["doc_id", "title", "sentences", "tokens", "lexical_tokens"], delimiter="\t")
    w.writeheader()
    w.writerows(doc_stats)

# Doc id mismatch.
meta_id_set = set(meta_ids)
conllu_id_set = set(conllu_ids)
only_meta = sorted(meta_id_set - conllu_id_set)
only_conllu = sorted(conllu_id_set - meta_id_set)
(out / "doc_ids_only_in_metadata.txt").write_text("\n".join(only_meta), encoding="utf-8")
(out / "doc_ids_only_in_conllu.txt").write_text("\n".join(only_conllu), encoding="utf-8")
summary["checks"]["doc_ids_only_in_metadata"] = len(only_meta)
summary["checks"]["doc_ids_only_in_conllu"] = len(only_conllu)

# Missing authors/titles.
missing_author = [r for r in meta_rows if not (r.get("author", "") or "").strip()]
missing_title = [r for r in meta_rows if not (r.get("title", "") or "").strip()]
for name, rows in [("missing_author_docs.tsv", missing_author), ("missing_title_docs.tsv", missing_title)]:
    with (out / name).open("w", encoding="utf-8", newline="") as f:
        fieldnames = meta_cols or ["doc_id", "title"]
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
summary["checks"]["missing_author_docs"] = len(missing_author)
summary["checks"]["missing_title_docs"] = len(missing_title)

# Stop/go rules.
if len(conllu_ids) != len(meta_rows): summary["stop_reasons"].append("CoNLL-U document count does not match metadata row count")
if summary["checks"]["conllu_duplicate_doc_ids"]: summary["stop_reasons"].append("duplicate CoNLL-U doc IDs")
if summary["checks"]["conllu_duplicate_sent_ids"]: summary["stop_reasons"].append("duplicate sentence IDs")
if bad_columns: summary["stop_reasons"].append("bad CoNLL-U column rows")
if dep_errors: summary["stop_reasons"].append("dependency integrity errors")
if only_meta or only_conllu: summary["stop_reasons"].append("metadata/CoNLL-U doc ID mismatch")
if missing_title: summary["stop_reasons"].append("missing titles")

# Missing author is warning, not automatic stop, because anonymous/folk/etc. texts may be real.
if missing_author:
    summary["warnings"].append(f"missing author in {len(missing_author)} documents; inspect missing_author_docs.tsv")

with (out / "post_annotation_qc_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

printf '\nMain outputs:\n'
find "$OUT" -maxdepth 1 -type f | sort
printf '\nSummary:\n'
cat "$OUT/post_annotation_qc_summary.json"
