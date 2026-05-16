# Output dictionary

## Emotion columns

For each Plutchik emotion (`anger`, `anticipation`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, `trust`):

- `emotion_<emotion>_count`: sum of lexicon association values over matched tokens.
- `emotion_<emotion>_per_1k`: count normalized per 1,000 lexical tokens.
- `emotion_<emotion>_prop`: share of the segment's total emotion association mass.
- `intensity_<emotion>_mean`: mean SloEmoLex intensity score over matched tokens, if intensity columns are present.

Sentiment:

- `sentiment_positive_count`, `sentiment_negative_count`: positive/negative association sums.
- `sentiment_positive_per_1k`, `sentiment_negative_per_1k`: normalized values.
- `sentiment_balance`: `(positive - negative) / (positive + negative)`.

VAD:

- `vad_valence_mean`, `vad_arousal_mean`, `vad_dominance_mean`: mean scores over lexicon hits with available VAD.

Coverage:

- `lexicon_hit_count`: matched tokens.
- `lexicon_unique_hit_count`: distinct matched lexicon keys.
- `lexicon_coverage`: matched tokens divided by lexical tokens eligible for lexicon matching.

## LDA columns

- `lda_topics.tsv`: one row per top term per topic.
- `lda_segment_topics.tsv`: topic probabilities for every modelled segment.
- `dominant_lda_topic`: topic with the highest probability for a segment.
- `dominant_lda_topic_score`: that topic's probability.
- `lda_topic_summary.tsv`: prevalence and emotion-weighted summaries by topic.

## Transformer columns

- `transformer_topic_predictions.tsv`: top-k labels from the selected text-classification model.
- `transformer_topic_document_summary.tsv`: document-level aggregation of top labels.
- `transformer_embedding_clusters.tsv`: unsupervised cluster ID and PCA coordinates per segment.
- `transformer_cluster_summary.tsv`: top terms and emotion means per embedding cluster.
