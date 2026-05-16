# Methodology notes

## Corpus assumptions

The input is expected to be CLASSLA-style CoNLL-U. The script uses:

- `FORM` and `LEMMA` for lexicon matching and topic features,
- `UPOS` for content-word filtering,
- `MISC/NER` for named-entity extraction,
- `# newdoc id = ...` and metadata comments for document grouping.

If no `# newdoc id` comments are available, the parser attempts a fallback split based on `# sent_id` prefixes.

## Emotion analysis

The lexicon layer uses SloEmoLex-compatible wide TSV files. The loader auto-detects columns for:

- Plutchik association values: anger, anticipation, disgust, fear, joy, sadness, surprise, trust;
- positive and negative sentiment;
- VAD dimensions: valence, arousal, dominance;
- optional emotion intensity columns.

The default matching order is `lemma,form`. This is important for Wikivir because CLASSLA lemmas usually improve matching, but historical spellings and poetry can make surface-form fallback useful. Token-level matches can be exported for validation with `--export-token-matches`.

## Topic modelling

The LDA model uses lemmatized content words. By default, content words are `NOUN, PROPN, VERB, ADJ, ADV, INTJ`. Stopwords are removed using a small built-in list plus any user-provided list.

The script can evaluate several topic counts with `--lda-grid` and write `lda_diagnostics.tsv`, including perplexity and a lightweight UMass coherence estimate. `--lda-auto-select` selects the topic count with the best UMass coherence in the grid.

## Transformer topic analysis

The transformer path has two independent components:

1. A supervised topic classifier, by default `cjvt/sloberta-trendi-topics`. Because this model was trained on Slovene news, labels should be interpreted cautiously for literary corpora.
2. Embedding-based clustering, by default with a multilingual SentenceTransformer model. This gives a corpus-internal topic view that is often better for literary data.

For long texts, transformer analysis should normally run over windows or paragraphs, not entire documents.

## Recommended validation steps

1. Inspect `lexicon_coverage.tsv` and `top_oov_content_lemmas.tsv`.
2. Spot-check `token_emotion_matches.tsv` when using historical normalization or fuzzy matching.
3. Compare LDA topic labels with transformer clusters rather than trusting one method.
4. For literary texts, treat supervised news-topic labels as broad signals, not ground truth.
5. Report model names, lexicon version, matching options, and coverage statistics in publications.


## Embedded metadata policy

Wikivir document metadata is treated as part of the corpus object and should be preserved in CoNLL-U comments. The analysis pipeline therefore reads document-level `# key = value` comments from the CoNLL-U input by default. External TSV/CSV metadata is supported only as an optional augmentation or recovery source. If annotation strips metadata comments, use `scripts/restore_wikivir_metadata.py` with the original XML or original CoNLL-U before running the analysis.
