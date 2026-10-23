#!/usr/bin/env python3
"""Build the candidate sets (options + target) used to train and evaluate every method.

Usage:
    python src/build_candidates.py --config config.yaml [--force]

Each row pairs one example text with a shuffled list of 5 or 10 candidate labels from the
same dataset, and a target: the gold label ID, or null for NONE_OF_THESE.

train       One row per example per epoch, resampled every epoch. Only seen labels are
            used, as gold or as negatives. open_set_fraction of rows drop the gold label;
            hard_negative_fraction draw negatives from the gold label's nearest labels.
validation  Fixed variants for every example and size: closed, gold_removed_random,
test        gold_removed_hard (the K labels most similar to gold) and, for CLINC150
            out-of-scope texts, natural_oos.

Reflex v3 trains its NONE head on simulated unseen labels (the real unseen labels stay untouched):
v3_split.json    pseudo_unseen_fraction of the training labels are pseudo-unseen; of the other
                 (base) labels, none_holdout_fraction of the texts are held out of the scorer.
v3_stage1        scorer training sets: base labels only, held-out texts excluded.
v3_stage2        NONE-head training sets, balanced over {seen, pseudo-unseen} x {present, removed}.

Writes data/processed/{train,validation,test,v3_stage1,v3_stage2}.parquet and v3_split.json;
existing outputs are kept unless --force.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SPLITS = ("train", "validation", "test")
HARD_POOL_SIZE = 20  # training hard negatives are sampled from the gold label's 20 nearest seen labels
TRAIN_TYPES = {
    (False, False): "closed",
    (False, True): "closed_hard",
    (True, False): "gold_removed_random",
    (True, True): "gold_removed_hard",
}


def label_neighbours(labels: dict, emb_dir: Path) -> dict[str, list[str]]:
    """For every label, the other labels of its dataset, most similar first.

    Similarity uses the raw label-name embeddings, so candidate sets stay identical across
    versions whatever text is later used to represent candidates.
    """
    index = pd.read_parquet(emb_dir / "candidate_index.parquet")
    vectors = np.load(emb_dir / "candidate_embeddings.npy")
    row = dict(zip(index.text, index.row))
    neighbours = {}
    for source in sorted({info["source"] for info in labels.values()}):
        ids = sorted(label_id for label_id, info in labels.items() if info["source"] == source)
        matrix = vectors[[row[labels[label_id]["label"]] for label_id in ids]]
        # Apple's Accelerate BLAS raises spurious FP warnings here; the result matches float64 to ~1e-6.
        with np.errstate(all="ignore"):
            similarity = matrix @ matrix.T
        for i, label_id in enumerate(ids):
            neighbours[label_id] = [ids[j] for j in np.argsort(-similarity[i], kind="stable") if j != i]
    return neighbours


def make_set(rng: random.Random, row, negatives: list[str], gold: str | None, kind: str, epoch: int | None = None) -> dict:
    candidates = list(negatives) + ([gold] if gold else [])
    rng.shuffle(candidates)  # puts the gold label in a uniformly random slot
    suffix = f"-e{epoch:02d}" if epoch is not None else ""
    result = {
        "set_id": f"{row.example_id}-{kind}-k{len(candidates)}{suffix}",
        "example_id": row.example_id,
        "source": row.source,
        "split": row.split,
        "text": row.text,
        "label_id": row.label_id,
        "label_group": row.label_group,
        "open_set_type": kind,
        "num_candidates": len(candidates),
        "candidate_ids": candidates,
        "target_id": gold,
        "is_none": gold is None,
    }
    if epoch is not None:
        result["epoch"] = epoch
    return result


def build_train(frame: pd.DataFrame, labels: dict, neighbours: dict, config: dict,
                allowed: list[str] | None = None, name: str = "train") -> pd.DataFrame:
    """Per-epoch training sets using only `allowed` labels (default: all seen labels)."""
    if allowed is None:
        allowed = [label_id for label_id, info in labels.items() if info["group"] == "seen"]
    seen = sorted(allowed)
    allowed_set = set(seen)
    seen_by_source = {s: [l for l in seen if labels[l]["source"] == s] for s in set(frame.source)}
    others = {l: [o for o in seen_by_source[labels[l]["source"]] if o != l] for l in seen}
    hard = {l: [n for n in neighbours[l] if n in allowed_set][:HARD_POOL_SIZE] for l in seen}

    rows = []
    for epoch in range(config["epochs"]):
        rng = random.Random(f"{config['seed']}-{name}-{epoch}")
        for row in frame.itertuples():
            k = rng.choice(config["candidate_sizes"])
            if row.is_oos:
                rows.append(make_set(rng, row, rng.sample(seen_by_source[row.source], k), None, "natural_oos", epoch))
                continue
            removed = rng.random() < config["open_set_fraction"]
            use_hard = rng.random() < config["hard_negative_fraction"]
            pool = hard[row.label_id] if use_hard else others[row.label_id]
            negatives = rng.sample(pool, k if removed else k - 1)
            gold = None if removed else row.label_id
            rows.append(make_set(rng, row, negatives, gold, TRAIN_TYPES[removed, use_hard], epoch))
    return pd.DataFrame(rows)


def split_v3(frame: pd.DataFrame, labels: dict, config: dict) -> dict:
    """Pseudo-unseen training labels, and the base-label texts held out of the v3 scorer."""
    rng = random.Random(f"{config['seed']}-v3-split")
    pseudo = []
    for source in sorted(set(frame.source)):
        seen = sorted(l for l, info in labels.items() if info["source"] == source and info["group"] == "seen")
        pseudo += rng.sample(seen, round(len(seen) * config["pseudo_unseen_fraction"]))
    base = sorted(l for l, info in labels.items() if info["group"] == "seen" and l not in pseudo)
    in_scope = frame[~frame.is_oos]
    held_out = []
    for label_id in base:
        ids = sorted(in_scope.example_id[in_scope.label_id == label_id])
        held_out += rng.sample(ids, round(len(ids) * config["none_holdout_fraction"]))
    return {"base_labels": base, "pseudo_unseen_labels": sorted(pseudo), "held_out_example_ids": sorted(held_out)}


def build_stage1(frame: pd.DataFrame, labels: dict, neighbours: dict, config: dict, split: dict) -> pd.DataFrame:
    """v3 scorer sets: base labels only (as gold or candidate); held-out texts excluded."""
    keep = frame.label_id.isin(split["base_labels"]) & ~frame.example_id.isin(split["held_out_example_ids"])
    return build_train(frame[keep | frame.is_oos], labels, neighbours, config, split["base_labels"], "v3-stage1")


def build_stage2(frame: pd.DataFrame, labels: dict, neighbours: dict, config: dict, split: dict) -> pd.DataFrame:
    """v3 NONE-head sets, equally many per {seen, pseudo-unseen} x {gold present, gold removed}.

    Seen queries are the held-out texts of base labels. Candidates come from base and pseudo-unseen
    labels, and every set holds at least one pseudo-unseen candidate (a wrong one unless it is the
    gold), so an unfamiliar option never tells by itself whether the answer is present.
    """
    base, pseudo = set(split["base_labels"]), set(split["pseudo_unseen_labels"])
    allowed = sorted(base | pseudo)
    held_out = set(split["held_out_example_ids"])
    in_scope = frame[~frame.is_oos]
    groups = {
        "seen": list(in_scope[in_scope.label_id.isin(base) & in_scope.example_id.isin(held_out)].itertuples()),
        "pseudo_unseen": list(in_scope[in_scope.label_id.isin(pseudo)].itertuples()),
    }
    by_source = {s: [l for l in allowed if labels[l]["source"] == s] for s in set(frame.source)}
    others = {l: [o for o in by_source[labels[l]["source"]] if o != l] for l in allowed}
    hard = {l: [n for n in neighbours[l] if n in base or n in pseudo][:HARD_POOL_SIZE] for l in allowed}

    rows = []
    for epoch in range(config["epochs"]):
        rng = random.Random(f"{config['seed']}-v3-stage2-{epoch}")
        for group, texts in groups.items():
            for removed in (False, True):
                for row in rng.choices(texts, k=config["none_sets_per_group"]):
                    k = rng.choice(config["candidate_sizes"])
                    use_hard = rng.random() < config["hard_negative_fraction"]
                    pool = hard[row.label_id] if use_hard else others[row.label_id]
                    negatives = rng.sample(pool, k if removed else k - 1)
                    gold = None if removed else row.label_id
                    if gold not in pseudo and not pseudo & set(negatives):
                        spare = [l for l in others[row.label_id] if l in pseudo and l not in negatives]
                        negatives[rng.randrange(len(negatives))] = rng.choice(spare)
                    result = make_set(rng, row, negatives, gold, TRAIN_TYPES[removed, use_hard], epoch)
                    result["set_id"] += f"-{group}-{len(rows)}"  # texts are resampled, so add a counter
                    result["none_group"] = group
                    rows.append(result)
    return pd.DataFrame(rows)


def build_eval(frame: pd.DataFrame, labels: dict, neighbours: dict, config: dict, split: str) -> pd.DataFrame:
    by_source = {s: sorted(l for l, info in labels.items() if info["source"] == s) for s in set(frame.source)}
    rng = random.Random(f"{config['seed']}-{split}")
    rows = []
    for row in frame.itertuples():
        for k in config["candidate_sizes"]:
            if row.is_oos:
                rows.append(make_set(rng, row, rng.sample(by_source[row.source], k), None, "natural_oos"))
                continue
            others = [l for l in by_source[row.source] if l != row.label_id]
            rows.append(make_set(rng, row, rng.sample(others, k - 1), row.label_id, "closed"))
            rows.append(make_set(rng, row, rng.sample(others, k), None, "gold_removed_random"))
            rows.append(make_set(rng, row, neighbours[row.label_id][:k], None, "gold_removed_hard"))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--force", action="store_true", help="Rebuild even if outputs exist")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    data_dir, out_dir = Path(config["standardized_dir"]), Path(config["processed_dir"])
    outputs = {name: out_dir / f"{name}.parquet" for name in (*SPLITS, "v3_stage1", "v3_stage2")}
    split_path = out_dir / "v3_split.json"
    if all(path.exists() for path in [*outputs.values(), split_path]) and not args.force:
        print(f"[skip] candidate sets already exist in {out_dir} (use --force to rebuild)")
        return

    labels = json.loads((data_dir / "labels.json").read_text())
    neighbours = label_neighbours(labels, Path(config["embeddings_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {split: pd.read_parquet(data_dir / f"{split}.parquet") for split in SPLITS}
    if args.force or not split_path.exists():
        split_path.write_text(json.dumps(split_v3(frames["train"], labels, config), indent=2) + "\n")
    v3 = json.loads(split_path.read_text())
    builders = {
        "train": lambda: build_train(frames["train"], labels, neighbours, config),
        "validation": lambda: build_eval(frames["validation"], labels, neighbours, config, "validation"),
        "test": lambda: build_eval(frames["test"], labels, neighbours, config, "test"),
        "v3_stage1": lambda: build_stage1(frames["train"], labels, neighbours, config, v3),
        "v3_stage2": lambda: build_stage2(frames["train"], labels, neighbours, config, v3),
    }
    for name, path in outputs.items():
        if path.exists() and not args.force:
            print(f"[skip] {path} already exists")
            continue
        sets = builders[name]()
        sets.to_parquet(path, index=False)
        counts = sets.groupby(["source", "open_set_type"]).size().unstack(fill_value=0)
        print(f"[saved] {path}: {len(sets)} sets\n{counts.to_string()}\n")


if __name__ == "__main__":
    main()
