#!/usr/bin/env python3
"""Run wikivir_emotion_topic_analysis.py from a JSON config file."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BOOLEAN_FLAGS = {
    "allow_missing_lexicon",
    "require_embedded_metadata",
    "strip_diacritics",
    "fuzzy_lexicon",
    "export_token_matches",
    "no_lda",
    "lda_auto_select",
    "run_transformers",
    "run_topic_classifier",
    "run_embeddings",
    "make_plots",
    "save_models",
    "verbose",
}


def key_to_flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    script = Path(__file__).resolve().parent / "wikivir_emotion_topic_analysis.py"
    cmd = [sys.executable, str(script)]
    for key, value in cfg.items():
        if value is None or value is False:
            continue
        flag = key_to_flag(key)
        if key in BOOLEAN_FLAGS:
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    if args.dry_run:
        print(" ".join(cmd))
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
