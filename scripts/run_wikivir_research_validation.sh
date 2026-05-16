#!/usr/bin/env bash
set -euo pipefail
cd "${WIKIVIR_REPO:-$HOME/projects/wikivir-emotion-topic-analysis}"
source .venv/bin/activate
mkdir -p logs

OUT="analysis/wikivir_research_v1"
TS="$(date +%Y%m%d_%H%M)"

# Lightweight validation helpers from produced TSVs.
python - <<'PY'
from pathlib import Path
import pandas as pd
out = Path('analysis/wikivir_research_v1')
tables = out / 'tables'
review = out / 'review'
review.mkdir(parents=True, exist_ok=True)

# Topic review candidates: top and random windows per BERTopic/KMeans topic
seg_path = tables / 'bertopic_segment_topics.tsv'
segments_path = tables / 'segments.tsv'
if seg_path.exists() and segments_path.exists():
    seg = pd.read_csv(seg_path, sep='\t', low_memory=False)
    segments = pd.read_csv(segments_path, sep='\t', low_memory=False)
    merged = seg.merge(segments, on='segment_id', how='left')
    rows = []
    for topic, g in merged.groupby('bertopic_topic'):
        g = g.copy()
        if 'bertopic_probability' in g:
            g['_score'] = pd.to_numeric(g['bertopic_probability'], errors='coerce').fillna(0)
            top = g.sort_values('_score', ascending=False).head(25)
        else:
            top = g.head(25)
        rnd = g.sample(min(10, len(g)), random_state=42)
        rows.append(top.assign(review_source='top'))
        rows.append(rnd.assign(review_source='random'))
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(review / 'topic_label_review.tsv', sep='\t', index=False)

# Emotion extremes for manual false-positive checking
emo_path = tables / 'segment_emotion_scores.tsv'
if emo_path.exists() and segments_path.exists():
    emo = pd.read_csv(emo_path, sep='\t', low_memory=False)
    segments = pd.read_csv(segments_path, sep='\t', low_memory=False)
    merged = emo.merge(segments, on='segment_id', how='left')
    cols = [c for c in merged.columns if c.startswith('emotion_') and c.endswith('_per_1k')]
    rows=[]
    for col in cols:
        rows.append(merged.sort_values(col, ascending=False).head(50).assign(review_emotion=col))
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(review / 'emotion_extremes_review.tsv', sep='\t', index=False)

# OOV review
for name in ['top_oov_content_lemmas.tsv', 'lexicon_coverage.tsv', 'metadata_coverage.tsv']:
    p = tables / name
    if p.exists():
        df = pd.read_csv(p, sep='\t', low_memory=False)
        df.head(5000).to_csv(review / name.replace('.tsv', '_review.tsv'), sep='\t', index=False)

print('Review sheets written to', review)
PY
