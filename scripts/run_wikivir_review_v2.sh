#!/usr/bin/env bash
set -euo pipefail

cd "${WIKIVIR_REPO:-$HOME/projects/wikivir-emotion-topic-analysis}"
source .venv/bin/activate
mkdir -p logs
OUT="${OUT:-analysis/wikivir_research_v2_clean}"
LOG="logs/wikivir_review_v2_$(date +%Y%m%d_%H%M).log"

stdbuf -oL -eL python scripts/generate_wikivir_review_sheets_v2.py \
  --analysis-dir "$OUT" \
  --output-dir "$OUT/review" \
  --topic-top-n 25 \
  --topic-random-n 10 \
  --emotion-top-n 50 \
  --emotion-min-words 80 \
  --emotion-units window,document \
  --oov-top-n 500 \
  2>&1 | tee "$LOG"

echo "Review sheets ready: $OUT/review"
