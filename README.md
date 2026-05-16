# Wikivir emotion and topic analysis

Standalone repo for Wikivir annotation, emotion analysis, and topic analysis. The preferred workflow is now: well-formed Wikivir XML → CLASSLA CoNLL-U with embedded metadata → SloEmoLex/LDA/transformer analysis.

The main script combines:

1. **Lexicon-based emotion analysis** with SloEmoLex 1.0.
2. **Unsupervised topic modelling** with LDA over CLASSLA lemmas.
3. **Transformer-based topic analysis**, either through a Slovene topic classifier or through transformer embeddings + clustering.

The pipeline is intentionally defensive: it reads document metadata directly from embedded CoNLL-U comments by default, accepts optional extra sidecar metadata, handles long documents through rolling windows, skips optional heavy models if dependencies are unavailable, writes a manifest, and produces plain TSV outputs that are easy to inspect.

## Layout

```text
wikivir-emotion-topic-analysis/
├── scripts/
│   ├── annotate_wikivir_xml_classla.py   # preferred clean XML → metadata-safe CoNLL-U annotator
│   ├── wikivir_emotion_topic_analysis.py
│   ├── restore_wikivir_metadata.py        # legacy/emergency metadata recovery only
│   ├── extract_conllu_metadata.py
│   └── download_sloemolex.py
├── configs/
│   └── wikivir_analysis.example.json
├── data/
│   ├── raw/              # put well-formed wikivir.xml here
│   ├── annotated/        # generated wikivir-classla.conllu goes here
│   ├── lexicons/         # put SloEmoLex_v1.tsv here
│   └── metadata/         # optional extra/recovery metadata; not the default source
├── analysis/             # generated outputs
├── tests/
├── requirements.txt
├── requirements-annotation.txt
├── requirements-transformers.txt
└── Makefile
```

## Install

Core analysis:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Transformer outputs:

```bash
pip install -r requirements-transformers.txt
```

Annotation dependencies, used only when producing fresh CLASSLA CoNLL-U from XML:

```bash
pip install -r requirements-annotation.txt
```

## Preferred clean workflow: XML → CLASSLA CoNLL-U with embedded metadata

Now that the Wikivir XML is well-formed, do **not** tag it as anonymous raw text and do **not** restore metadata afterwards. Run the dedicated XML annotator, which streams each `<doc>` element, keeps every XML attribute as CoNLL-U document metadata, and writes stable unique IDs:

```bash
python scripts/annotate_wikivir_xml_classla.py \
  --input data/raw/wikivir.xml \
  --output data/annotated/wikivir-classla.conllu \
  --metadata-output data/metadata/wikivir-metadata.tsv \
  --summary reports/wikivir-classla-summary.json \
  --errors reports/wikivir-classla-errors.tsv \
  --validation-report reports/wikivir-classla-validation.tsv \
  --models-dir ~/classla_resources_shared \
  --download-models \
  --require-clean-output \
  --overwrite
```

The beginning of each document in the generated CoNLL-U should look like this:

```conllu
# newdoc id = wikivir-000001
# source_xml = wikivir.xml
# xml_doc_index = 1
# title = Slovenja
# author = Jovan Vesel Koseski
# century = 19
# genre = priložnostne pesmi
# newpar id = wikivir-000001.p000001
# sent_id = wikivir-000001.s000001
# text = Slovenja
```

Before a full run, a metadata-only dry run is useful:

```bash
python scripts/annotate_wikivir_xml_classla.py \
  --input data/raw/wikivir.xml \
  --output data/annotated/wikivir-classla.conllu \
  --dry-run
```

For a quick CLASSLA smoke test, annotate only the first few documents:

```bash
python scripts/annotate_wikivir_xml_classla.py \
  --input data/raw/wikivir.xml \
  --output data/annotated/wikivir-classla.sample.conllu \
  --metadata-output data/metadata/wikivir-metadata.sample.tsv \
  --models-dir ~/classla_resources_shared \
  --limit-docs 5 \
  --download-models \
  --overwrite
```

Useful sanity checks after annotation:

```bash
grep -m 30 -E '^# (newdoc id|title|author|century|year|genre|publication|sent_id)' \
  data/annotated/wikivir-classla.conllu

grep '^# newdoc id' data/annotated/wikivir-classla.conllu \
  | sed 's/^# newdoc id = //' \
  | sort | uniq -d | head
```

The duplicate-ID command should print nothing.


On a GPU machine, install the correct PyTorch build for your CUDA version before or instead of the generic `torch` package in `requirements-transformers.txt`.

## Get SloEmoLex

SloEmoLex is publicly available through CLARIN.SI, but it is licensed **CC BY-NC-SA 4.0**, so do not vendor it into the repo unless that is appropriate for your use case.

```bash
python scripts/download_sloemolex.py --accept-license
```

This writes:

```text
data/lexicons/SloEmoLex_v1.tsv
data/lexicons/readme_sloEmoLex.txt
```

You can also download it manually from the CLARIN.SI item `http://hdl.handle.net/11356/1875`.

## Metadata: default is embedded CoNLL-U comments

For Wikivir, metadata belongs in the CoNLL-U comments, not in a mandatory sidecar file. A good document start looks like this:

```conllu
# newdoc id = wikivir-000001
# title = Slovenja
# author = Jovan Vesel Koseski
# century = 19
# genre = priložnostne pesmi
# sent_id = wikivir-000001.1
# text = Slovenja
1	Slovenja	Slovenja	PROPN	...
```

The analysis script reads `# title`, `# author`, `# century`, `# year`, `# genre`, `# publication`, `# source`, `# url`, and any other document-level `# key = value` comments automatically. The `--metadata` option is only for **extra** sidecar metadata or emergency recovery workflows.

Check whether your annotated corpus still has embedded metadata:

```bash
grep -m 20 -E '^# (newdoc|title|author|century|year|genre|publication|source|url)' \
  data/annotated/wikivir-classla.conllu
```

Extract embedded metadata to inspect it:

```bash
python scripts/extract_conllu_metadata.py \
  --input data/annotated/wikivir-classla.conllu \
  --output analysis/wikivir/tables/document_metadata_from_conllu.tsv \
  --coverage-output analysis/wikivir/tables/metadata_coverage_from_conllu.tsv
```

If you are dealing with an older already-annotated file where CLASSLA annotation stripped metadata, restore it from the original Wikivir XML. This is now a legacy/emergency path; the preferred path is `annotate_wikivir_xml_classla.py` above. The restore script has an XML recovery mode for Wikivir files that are not strictly well-formed XML, and it now also repairs duplicated/generic `# newdoc id` values while keeping traceability in `annotated_doc_id` / `source_doc_id` comments:

```bash
python scripts/restore_wikivir_metadata.py \
  --annotated data/annotated/wikivir-classla.conllu \
  --source-xml data/raw/wikivir.xml \
  --xml-parse-mode recover \
  --match-mode by-order \
  --doc-id-mode auto \
  --output data/annotated/wikivir-classla.with-metadata.conllu \
  --metadata-output data/metadata/wikivir-metadata-restored.tsv \
  --replace-existing
```

`--xml-parse-mode auto` is the default: it tries a strict XML parser first and falls back to the tolerant `<doc ...>` start-tag scanner. Use `recover` when the XML is known to contain bad control characters, unescaped ampersands, or other parser-breaking junk.

`--doc-id-mode auto` keeps already-good document IDs, but if the annotated CoNLL-U has duplicated or useless IDs, it uses XML/source IDs or generated order IDs like `doc-000001`. This prevents pandas crashes and, more importantly, stops metadata from being overwritten during restoration.

Then run analysis on `wikivir-classla.with-metadata.conllu`.

Sanity-check uniqueness before the analysis run:

```bash
grep '^# newdoc id' data/annotated/wikivir-classla.with-metadata.conllu | sort | uniq -d | head
```

That command should print nothing.

## Minimal run

```bash
python scripts/wikivir_emotion_topic_analysis.py \
  --input data/annotated/wikivir-classla.conllu \
  --sloemolex data/lexicons/SloEmoLex_v1.tsv \
  --output-dir analysis/wikivir \
  --make-plots
```

## Full run with LDA + transformer models

```bash
python scripts/wikivir_emotion_topic_analysis.py \
  --input data/annotated/wikivir-classla.conllu \
  --sloemolex data/lexicons/SloEmoLex_v1.tsv \
  --output-dir analysis/wikivir \
  --segment-levels document,paragraph,window,sentence \
  --window-size 250 \
  --window-step 125 \
  --lda-unit document \
  --lda-topics 30 \
  --lda-grid 10,20,30,40,50 \
  --lda-auto-select \
  --run-transformers \
  --topic-classifier-model cjvt/sloberta-trendi-topics \
  --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --embedding-clusters 0 \
  --export-token-matches \
  --make-plots \
  --save-models \
  --require-embedded-metadata
```

For very long literary works, transformer and lexicon windows are usually more informative than whole-document classification. The defaults therefore build rolling windows even when LDA is run on documents.

## Important options

`--normalization historical` applies cautious historical cleanup before lexicon matching. It does **not** modernize old Slovene morphology. For old Wikivir texts, inspect `tables/top_oov_content_lemmas.tsv` and consider adding a hand-built normalization/variant lexicon if coverage is low.

`--strip-diacritics` adds accent-insensitive matching. This can improve recall but may introduce false positives.

`--fuzzy-lexicon` uses `rapidfuzz` for approximate matching. It is off by default because it can be slow and can introduce questionable matches in historical texts.

`--allow-missing-lexicon` lets you run LDA and transformer-only topic analysis without SloEmoLex.

`--metadata` and `--metadata-doc-id-column` are only for optional extra sidecar metadata. They are not needed when metadata is already embedded in the CoNLL-U comments.

`--require-embedded-metadata` fails early when the CoNLL-U has no useful document metadata comments. This is handy in batch runs so you do not accidentally analyze a metadata-stripped corpus.

## Main outputs

Tables are written to `analysis/wikivir/tables/`:

- `documents.tsv` — parsed documents, token counts, document metadata.
- `document_metadata.tsv` — embedded/merged metadata by document.
- `metadata_coverage.tsv` — how many documents contain each metadata field.
- `segments.tsv` — generated document/paragraph/sentence/window segments.
- `segment_emotion_scores.tsv` — SloEmoLex emotion/sentiment/VAD scores by segment.
- `document_emotion_scores.tsv` — document-level emotion profile.
- `lexicon_coverage.tsv` — lexicon coverage and detected SloEmoLex columns.
- `top_oov_content_lemmas.tsv` — frequent content lemmas not found in SloEmoLex.
- `token_emotion_matches.tsv` — token-level matches, only with `--export-token-matches`.
- `emotion_by_author.tsv`, `emotion_by_century.tsv`, `emotion_by_genre.tsv`, etc. — written when those metadata fields exist.
- `named_entities.tsv` and `named_entity_counts.tsv` — entity spans from CLASSLA NER.
- `entity_emotion_summary.tsv` — entity-level sentence emotion context, when sentence scores are available.
- `lda_diagnostics.tsv` — LDA perplexity and UMass coherence by topic count.
- `lda_topics.tsv` — topic-term weights.
- `lda_segment_topics.tsv` — per-segment topic distributions.
- `lda_topic_summary.tsv` — topic prevalence and emotion-weighted topic summaries.
- `transformer_topic_predictions.tsv` — top-k topic classifier labels per segment.
- `transformer_topic_document_summary.tsv` — document-level aggregate of classifier labels.
- `transformer_embedding_clusters.tsv` — cluster labels and PCA coordinates from embeddings.
- `transformer_cluster_summary.tsv` — cluster descriptors and emotion summaries.
- `*_crosswalk.tsv` — emotion/topic cross-tabulations.

The root output directory also contains:

- `report.md` — compact human-readable report.
- `manifest.json` — arguments, file hashes, warnings, output paths.

Plots are written to `analysis/wikivir/plots/`.

## Notes on transformer topic labels

The default classifier `cjvt/sloberta-trendi-topics` is trained on Slovene news texts. It is useful as a broad topical signal, but for literary Wikivir data its labels should be treated as approximate. The embedding-clustering path is more corpus-internal and can be more suitable for literary material.

## Smoke test

```bash
pytest -q
```
