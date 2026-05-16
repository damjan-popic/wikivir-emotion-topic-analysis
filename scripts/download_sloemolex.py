#!/usr/bin/env python3
"""Download SloEmoLex 1.0 from CLARIN.SI.

The lexicon is publicly available, but it is licensed CC BY-NC-SA 4.0.
Use this downloader only if your intended use is compatible with that licence.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

SLOEMOLEX_URL = "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1875/SloEmoLex_v1.tsv?isAllowed=y&sequence=4"
README_URL = "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1875/readme_sloEmoLex.txt?isAllowed=y&sequence=5"
EXPECTED_MD5 = "724424032e86ccb2a771be2361b7f26b"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, output: Path, retries: int = 3) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wikivir-emotion-topic-analysis/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, output.open("wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            return
        except Exception as exc:  # pragma: no cover
            last_error = exc
            print(f"download attempt {attempt}/{retries} failed: {exc}", file=sys.stderr)
            time.sleep(2 * attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {last_error}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download SloEmoLex_v1.tsv from CLARIN.SI.")
    ap.add_argument("--output", type=Path, default=Path("data/lexicons/SloEmoLex_v1.tsv"))
    ap.add_argument("--readme-output", type=Path, default=Path("data/lexicons/readme_sloEmoLex.txt"))
    ap.add_argument("--accept-license", action="store_true", help="Confirm that your use is compatible with CC BY-NC-SA 4.0.")
    ap.add_argument("--skip-readme", action="store_true")
    args = ap.parse_args()
    if not args.accept_license:
        print("SloEmoLex is licensed CC BY-NC-SA 4.0. Re-run with --accept-license after checking compatibility.", file=sys.stderr)
        return 2
    download(SLOEMOLEX_URL, args.output)
    got = md5(args.output)
    if got != EXPECTED_MD5:
        print(f"warning: MD5 mismatch for {args.output}: expected {EXPECTED_MD5}, got {got}", file=sys.stderr)
    else:
        print(f"downloaded {args.output} md5={got}", file=sys.stderr)
    if not args.skip_readme:
        download(README_URL, args.readme_output)
        print(f"downloaded {args.readme_output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
