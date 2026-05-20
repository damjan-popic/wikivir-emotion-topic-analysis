# Wikivir review wiring status

This pack is for manual validation after the cleaned Wikivir CLASSLA/research run. No plots or interpretation are included.

## Automated checks

- Hardening summary `ok`: **True**
- Hardening warnings: **0**
- Hardening errors: **0**
- Topic review rows: **4143**
- Topics: **120**
- Topic score source: **{'centroid_cosine': 4143}**
- Topic score min/median/max: **0.4648284152615292 / 0.8596730969775491 / 0.9446368673558284**
- Glued preview hits in topic review: **0**
- Topic-score alignment: **{'both': 4143, 'left_only': 0, 'right_only': 0}**
- Strict emotion file rows: **400**
- Strict deduplicated emotion rows: **329**
- Strict emotion thresholds: lexical >= **80**, words >= **80**

## Metadata coverage

| metadata_key   |   documents_with_key |   documents_with_nonempty_value |   coverage |
|:---------------|---------------------:|--------------------------------:|-----------:|
| author         |                16940 |                           16940 | 0.798755   |
| category       |                   33 |                              33 | 0.00155602 |
| century        |                17164 |                           17164 | 0.809317   |
| genre          |                18955 |                           18955 | 0.893767   |
| language       |                   48 |                              48 | 0.0022633  |
| publication    |                 1492 |                            1492 | 0.0703508  |
| source_xml     |                21208 |                           21208 | 1          |
| title          |                21208 |                           21208 | 1          |
| xml_doc_index  |                21208 |                           21208 | 1          |
| year           |                 4508 |                            4508 | 0.212561   |

## Lexicon coverage

| metric                     | value                                                                                                                                                                                                                                                                  |
|:---------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| lexicon_rows               | 19998                                                                                                                                                                                                                                                                  |
| lexicon_keys               | 16711                                                                                                                                                                                                                                                                  |
| corpus_word_tokens         | 61814387                                                                                                                                                                                                                                                               |
| corpus_word_token_hits     | 16513272                                                                                                                                                                                                                                                               |
| corpus_word_token_coverage | 0.26714285785928765                                                                                                                                                                                                                                                    |
| lexicon_word_columns       | Slovenian word(s)                                                                                                                                                                                                                                                      |
| detected_emotion_columns   | {"anger": ["Anger"], "anticipation": ["Anticipation"], "disgust": ["Disgust"], "fear": ["Fear"], "joy": ["Joy"], "sadness": ["Sadness"], "surprise": ["Surprise"], "trust": ["Trust"]}                                                                                 |
| detected_intensity_columns | {"anger": ["Anger-intensity"], "anticipation": ["Anticipation-intensity"], "disgust": ["Disgust-intensity"], "fear": ["Fear-intensity"], "joy": ["Joy-intensity"], "sadness": ["Sadness-intensity"], "surprise": ["Surprise-intensity"], "trust": ["Trust-intensity"]} |
| detected_sentiment_columns | {"positive": ["Positive"], "negative": ["Negative"]}                                                                                                                                                                                                                   |
| detected_vad_columns       | {"valence": ["Valence"], "arousal": ["Arousal"], "dominance": ["Dominance"]}                                                                                                                                                                                           |

## Manual validation order

1. Fill `topic_manual_label_sheet.tsv`: assign a Slovene label, optional English gloss, topic type, validity, and notes.
2. Use `topic_examples_for_manual_review.tsv` when a topic label is unclear.
3. Use `emotion_extremes_manual_review_lexical80_dedup.tsv` to validate high-emotion examples. Prefer this strict deduplicated sheet over the older non-strict file.
4. Use `oov_manual_review_sheet.tsv` to classify high-frequency OOV lemmas as archaic spelling, name/proper noun leakage, normal lemma absent from SloEmoLex, function/general lemma, or noise.

## Recommended topic_type values

`theme`, `motif`, `genre_register`, `work_cycle`, `character_name`, `author_source`, `historical_language`, `translation_foreign_names`, `mixed_noise`.

## Stop condition before plots

Do not produce final plots until topic labels are manually filled and high-emotion examples have at least a light pass for false positives.