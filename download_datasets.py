#!/usr/bin/env python3
"""Download CLINC150, BANKING77, Typed Decisions and API-Bank into data/raw.

Files that already exist are skipped, so re-running is safe.
"""

import urllib.request
from pathlib import Path

OUTPUT_DIR = Path("data/raw")

TYPED_DECISIONS = "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/e742f0a6998cf98bae8086d1dfc139bc601b284a/"
API_BANK = "https://huggingface.co/datasets/liminghao1630/API-Bank/resolve/12e8158b7628c168f07e8f31fbbe3445e99f44cf/"

FILES = {
    "clinc150/data_full.json": (
        "https://raw.githubusercontent.com/clinc/oos-eval/"
        "828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json"
    ),
    "clinc150/domains.json": (
        "https://raw.githubusercontent.com/clinc/oos-eval/"
        "828f8093932c8fe6ca7936c3d2e52903b1c523de/data/domains.json"
    ),
    "banking77/train.csv": (
        "https://raw.githubusercontent.com/PolyAI-LDN/"
        "task-specific-datasets/master/banking_data/train.csv"
    ),
    "banking77/test.csv": (
        "https://raw.githubusercontent.com/PolyAI-LDN/"
        "task-specific-datasets/master/banking_data/test.csv"
    ),
    "typed_decisions/README.md": TYPED_DECISIONS + "README.md",
    "typed_decisions/train.parquet": TYPED_DECISIONS + "all/train-00000-of-00001.parquet",
    "typed_decisions/test.parquet": TYPED_DECISIONS + "all/test-00000-of-00001.parquet",
}

for name in [
    "test-data/level-1-api.json",
    "test-data/level-1-response.json",
    "test-data/level-2-api.json",
    "test-data/level-2-response.json",
    "test-data/level-3.json",
    "training-data/lv1-api-train.json",
    "training-data/lv1-response-train.json",
    "training-data/lv1-train.json",
    "training-data/lv2-api-train.json",
    "training-data/lv2-response-train.json",
    "training-data/lv2-train.json",
    "training-data/lv3-api-train.json",
    "training-data/lv3-response-train.json",
    "training-data/lv3-train.json",
]:
    FILES[f"api_bank/{name}"] = API_BANK + name


def main() -> None:
    for name, url in FILES.items():
        destination = OUTPUT_DIR / name
        if destination.exists():
            print(f"[skip] {destination} already exists")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {url}")
        # Write to a temp file first so an interrupted download isn't mistaken for a finished one.
        partial = destination.with_name(destination.name + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(destination)
        print(f"[saved] {destination}")


if __name__ == "__main__":
    main()
