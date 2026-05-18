#!/usr/bin/env bash
set -euo pipefail

cd "${WIKIVIR_REPO:-$HOME/projects/wikivir-emotion-topic-analysis}"
source .venv/bin/activate
mkdir -p logs analysis

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

INPUT="${INPUT:-data/annotated/wikivir-classla.clean.conllu}"
OUT="${OUT:-analysis/wikivir_research_v2_clean}"
LOG="logs/wikivir_research_v2_clean_$(date +%Y%m%d_%H%M).log"

# No --make-plots here. We validate first, label/review, and only plot after the data is structurally clean.
stdbuf -oL -eL python scripts/wikivir_emotion_topic_analysis.py \
  --input "$INPUT" \
  --sloemolex data/lexicons/SloEmoLex_v1.tsv \
  --output-dir "$OUT" \
  --require-embedded-metadata \
  --resume \
  --progress-every 1000 \
  --segment-levels document,window \
  --window-size 300 \
  --window-step 150 \
  --normalization historical \
  --match-on lemma,form \
  --lexicon-shard-size 10000 \
  --lda-unit document \
  --lda-grid 20,30,40,60,80,100 \
  --lda-auto-select \
  --lda-topics 60 \
  --lda-top-words 30 \
  --lda-min-df 3 \
  --lda-max-df 0.80 \
  --lda-max-features 100000 \
  --lda-max-iter 80 \
  --lda-learning-method batch \
  --lda-n-jobs 8 \
  --run-embeddings \
  --embedding-unit window \
  --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --embedding-backend auto \
  --embedding-max-segments 0 \
  --embedding-clusters 120 \
  --embedding-kmeans-passes 5 \
  --embedding-shard-size 2000 \
  --embedding-pca-max-segments 100000 \
  --embedding-silhouette-sample 50000 \
  --embedding-full-load-max-mb 4096 \
  --cluster-top-terms 40 \
  --cluster-top-term-features 100000 \
  --run-bertopic \
  --bertopic-unit window \
  --bertopic-language multilingual \
  --bertopic-embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --bertopic-min-words 40 \
  --bertopic-cluster-model kmeans \
  --bertopic-kmeans-clusters 120 \
  --bertopic-skip-dim-reduction \
  --bertopic-nr-topics none \
  --bertopic-top-words 40 \
  --bertopic-min-df 5 \
  --bertopic-max-df 0.85 \
  --bertopic-max-features 100000 \
  --bertopic-max-segments 0 \
  --bertopic-low-memory \
  --save-models \
  --no-save-embedding-matrices \
  --verbose \
  2>&1 | tee "$LOG"

echo "Analysis tables/checkpoints ready, with no plots yet: $OUT"
