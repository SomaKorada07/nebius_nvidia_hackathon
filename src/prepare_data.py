#!/usr/bin/env python3
"""Standardize CLINC150 and BANKING77 into one format and split into train/validation/test.

Usage:
    python src/prepare_data.py --config config.yaml [--force]

Reads the raw files written by download_datasets.py and writes:
    data/standardized/{train,validation,test}.parquet
    data/standardized/labels.json

A fraction of each dataset's labels is marked "unseen": their rows are dropped from
train so the model can later be tested on labels it never trained on.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

SPLITS = ("train", "validation", "test")

CLINC_SIZES = {
    "train": 15_000,
    "val": 3_000,
    "test": 4_500,
    "oos_train": 100,
    "oos_val": 100,
    "oos_test": 1_000,
}
CLINC_SPLITS = {
    "train": "train",
    "val": "validation",
    "test": "test",
    "oos_train": "train",
    "oos_val": "validation",
    "oos_test": "test",
}
BANKING_SIZES = {"train": 10_003, "test": 3_080}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def make_row(source: str, origin_split: str, index: int, split: str, text: str, label: str | None) -> dict:
    return {
        "example_id": f"{source}-{origin_split}-{index:05d}",
        "source": source,
        "split": split,
        "text": text.strip(),
        "label": label,
        "label_id": f"{source}:{label}" if label is not None else None,
        "is_oos": label is None,
    }


def choose_unseen(groups: list[list[str]], fraction: float, seed: int) -> set[str]:
    """Sample the same fraction of labels from every group."""
    rng = random.Random(seed)
    unseen = set()
    for group in groups:
        unseen.update(rng.sample(sorted(group), round(len(group) * fraction)))
    return unseen


def load_clinc150(raw_dir: Path) -> tuple[pd.DataFrame, list[list[str]]]:
    data = json.loads((raw_dir / "clinc150" / "data_full.json").read_text())
    sizes = {name: len(data[name]) for name in CLINC_SIZES}
    check(sizes == CLINC_SIZES, f"Unexpected CLINC150 split sizes: {sizes}")

    rows = []
    for origin_split, split in CLINC_SPLITS.items():
        for index, (text, label) in enumerate(data[origin_split]):
            check((label == "oos") == origin_split.startswith("oos"), f"Bad label {label!r} in {origin_split}")
            rows.append(make_row("clinc150", origin_split, index, split, text, None if label == "oos" else label))
    frame = pd.DataFrame(rows)

    # Domains are used only to balance the unseen-label choice: 3 of 15 intents per domain.
    domains = json.loads((raw_dir / "clinc150" / "domains.json").read_text())
    labels = set(frame.label.dropna())
    check(len(labels) == 150, f"Expected 150 CLINC150 labels, found {len(labels)}")
    check(set().union(*domains.values()) == labels, "CLINC150 domains do not cover the labels")
    return frame, list(domains.values())


def load_banking77(raw_dir: Path, validation_fraction: float, seed: int) -> tuple[pd.DataFrame, list[list[str]]]:
    files = {name: pd.read_csv(raw_dir / "banking77" / f"{name}.csv") for name in BANKING_SIZES}
    sizes = {name: len(frame) for name, frame in files.items()}
    check(sizes == BANKING_SIZES, f"Unexpected BANKING77 sizes: {sizes}")

    train = files["train"]
    _, validation_index = train_test_split(
        train.index, test_size=validation_fraction, stratify=train["category"], random_state=seed
    )
    validation_index = set(validation_index)

    rows = [
        make_row("banking77", "train", i, "validation" if i in validation_index else "train", r.text, r.category)
        for i, r in train.iterrows()
    ]
    rows += [make_row("banking77", "test", i, "test", r.text, r.category) for i, r in files["test"].iterrows()]
    frame = pd.DataFrame(rows)

    labels = set(frame.label)
    check(len(labels) == 77, f"Expected 77 BANKING77 labels, found {len(labels)}")
    return frame, [sorted(labels)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--force", action="store_true", help="Rebuild even if outputs exist")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    raw_dir = Path(config["raw_dir"])
    out_dir = Path(config["standardized_dir"])
    outputs = [out_dir / f"{split}.parquet" for split in SPLITS] + [out_dir / "labels.json"]
    if all(path.exists() for path in outputs) and not args.force:
        print(f"[skip] standardized data already exists in {out_dir} (use --force to rebuild)")
        return

    seed, fraction = config["seed"], config["unseen_label_fraction"]
    frames, labels = [], {}
    for source, (frame, groups) in {
        "clinc150": load_clinc150(raw_dir),
        "banking77": load_banking77(raw_dir, config["validation_fraction"], seed),
    }.items():
        unseen = choose_unseen(groups, fraction, seed)
        for label in sorted(set(frame.label.dropna())):
            labels[f"{source}:{label}"] = {
                "source": source,
                "label": label,
                "group": "unseen" if label in unseen else "seen",
            }
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["label_group"] = [
        "oos" if is_oos else labels[label_id]["group"] for is_oos, label_id in zip(data.is_oos, data.label_id)
    ]
    data = data[~((data.split == "train") & (data.label_group == "unseen"))]

    check(data.example_id.is_unique, "Duplicate example IDs")
    check(data.loc[~data.is_oos, "label_id"].isin(list(labels)).all(), "In-scope row with unknown label")

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        data[data.split == split].reset_index(drop=True).to_parquet(out_dir / f"{split}.parquet", index=False)
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")

    print(data.groupby(["split", "source", "label_group"]).size().unstack(fill_value=0).to_string())
    print(f"[done] wrote {', '.join(str(path) for path in outputs)}")


if __name__ == "__main__":
    main()
