#!/usr/bin/env bash
set -euo pipefail
cd "${WIKIVIR_REPO:-$HOME/projects/wikivir-emotion-topic-analysis}"
source .venv/bin/activate
mkdir -p logs
OUT="${OUT:-analysis/wikivir_research_v2}"
LOG="logs/wikivir_research_v2_validation_$(date +%Y%m%d_%H%M).log"

stdbuf -oL -eL python scripts/make_wikivir_review_sheets.py \
  --analysis-dir "$OUT" \
  --review-dir "$OUT/review" \
  --topic-top-n 25 \
  --topic-random-n 10 \
  --topic-min-words 40 \
  --emotion-unit window \
  --emotion-min-words 100 \
  --emotion-min-hits 2 \
  --emotion-min-coverage 0.05 \
  --emotion-top-n 50 \
  --oov-top-n 500 \
  --preview-chars 500 \
  2>&1 | tee "$LOG"

echo "[done] Review sheets written to: $OUT/review"
echo "[done] Log: $LOG"
