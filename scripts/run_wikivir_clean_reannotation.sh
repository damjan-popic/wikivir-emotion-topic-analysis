#!/usr/bin/env bash
set -euo pipefail

cd "${WIKIVIR_REPO:-$HOME/projects/wikivir-emotion-topic-analysis}"
source .venv/bin/activate
mkdir -p data/raw data/annotated data/metadata reports logs

RAW_XML="${RAW_XML:-data/raw/wikivir.xml}"
CLEAN_XML="${CLEAN_XML:-data/raw/wikivir.cleaned.xml}"
ANNOTATED="${ANNOTATED:-data/annotated/wikivir-classla.clean.conllu}"
MODELS_DIR="${MODELS_DIR:-$HOME/classla_resources_shared}"

python scripts/normalize_wikivir_xml_metadata.py \
  --input "$RAW_XML" \
  --output "$CLEAN_XML" \
  --metadata-output data/metadata/wikivir.cleaned.metadata.tsv \
  --fixes-output reports/wikivir_metadata_fixes.tsv \
  --coverage-output reports/wikivir_xml_metadata_coverage.tsv \
  --summary reports/wikivir_xml_metadata_summary.json \
  --min-coverage title=0.95,genre=0.80,century=0.70 \
  --fail-on-threshold

python scripts/annotate_wikivir_xml_classla.py \
  --input "$CLEAN_XML" \
  --output "$ANNOTATED" \
  --metadata-output data/metadata/wikivir.classla.metadata.tsv \
  --summary reports/wikivir_classla_summary.json \
  --errors reports/wikivir_classla_errors.tsv \
  --validation-report reports/wikivir_classla_validation.tsv \
  --models-dir "$MODELS_DIR" \
  --download-models \
  --require-clean-output \
  --overwrite

python scripts/check_wikivir_conllu_integrity.py \
  --input "$ANNOTATED" \
  --summary reports/wikivir_conllu_integrity_summary.json \
  --errors-output reports/wikivir_conllu_integrity_errors.tsv \
  --metadata-output reports/wikivir_conllu_document_metadata.tsv \
  --coverage-output reports/wikivir_conllu_metadata_coverage.tsv \
  --min-docs 1000 \
  --min-coverage title=0.95,genre=0.80,century=0.70 \
  --fail-on-error

echo "Clean annotated corpus ready: $ANNOTATED"
