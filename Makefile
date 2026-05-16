PYTHON ?= python
INPUT ?= data/annotated/wikivir-classla.conllu
SLOEMOLEX ?= data/lexicons/SloEmoLex_v1.tsv
OUT ?= analysis/wikivir
RAW_XML ?= data/raw/wikivir.xml
MODELS_DIR ?= ~/classla_resources_shared
RESTORED_INPUT ?= data/annotated/wikivir-classla.with-metadata.conllu

install:
	$(PYTHON) -m pip install -r requirements.txt

install-annotation:
	$(PYTHON) -m pip install -r requirements-annotation.txt

install-transformers:
	$(PYTHON) -m pip install -r requirements-transformers.txt

annotate-dry-run:
	$(PYTHON) scripts/annotate_wikivir_xml_classla.py --input $(RAW_XML) --output $(INPUT) --dry-run

annotate-sample:
	$(PYTHON) scripts/annotate_wikivir_xml_classla.py --input $(RAW_XML) --output data/annotated/wikivir-classla.sample.conllu --metadata-output data/metadata/wikivir-metadata.sample.tsv --models-dir $(MODELS_DIR) --limit-docs 5 --download-models --overwrite

annotate:
	$(PYTHON) scripts/annotate_wikivir_xml_classla.py --input $(RAW_XML) --output $(INPUT) --metadata-output data/metadata/wikivir-metadata.tsv --summary reports/wikivir-classla-summary.json --errors reports/wikivir-classla-errors.tsv --validation-report reports/wikivir-classla-validation.tsv --models-dir $(MODELS_DIR) --download-models --require-clean-output --overwrite

download-sloemolex:
	$(PYTHON) scripts/download_sloemolex.py --accept-license

check-metadata:
	$(PYTHON) scripts/extract_conllu_metadata.py --input $(INPUT) --output $(OUT)/tables/document_metadata_from_conllu.tsv --coverage-output $(OUT)/tables/metadata_coverage_from_conllu.tsv --fail-if-empty

restore-metadata-from-xml:
	$(PYTHON) scripts/restore_wikivir_metadata.py --annotated $(INPUT) --source-xml $(RAW_XML) --output $(RESTORED_INPUT) --metadata-output data/metadata/wikivir-metadata-restored.tsv --replace-existing

run:
	$(PYTHON) scripts/wikivir_emotion_topic_analysis.py --input $(INPUT) --sloemolex $(SLOEMOLEX) --output-dir $(OUT) --make-plots

run-full:
	$(PYTHON) scripts/wikivir_emotion_topic_analysis.py --input $(INPUT) --sloemolex $(SLOEMOLEX) --output-dir $(OUT) --require-embedded-metadata --segment-levels document,paragraph,window,sentence --lda-grid 10,20,30,40 --lda-auto-select --run-transformers --export-token-matches --make-plots --save-models

test:
	$(PYTHON) -m pytest -q
