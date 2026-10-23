#!/usr/bin/env python3
"""Train the ordered baseline and the Reflex heads on cached embeddings, then calibrate.

Usage:
    python src/train.py --config config.yaml [--heads set_head ordered_head reflex_v3] [--force]

All heads see the same candidate sets in the same order. Early stopping, temperature
scaling and the cosine threshold use validation rows with seen labels and natural OOS
only; unseen-label rows are left for the final experiments.

Reflex v3 (ReflexRankingHead) trains in three steps:
  1. scorer on base labels (v3_stage1): ranking cross-entropy + pair_match_weight * pair-match BCE;
  2. NONE head on the frozen scorer (v3_stage2), with pseudo-unseen labels: (K + 1)-way cross-entropy;
  3. scorer retrained on all training labels (train), reusing the NONE head from step 2.

Writes to artifacts/: set_head.pt, ordered_head.pt, reflex_v3.pt (step 3), reflex_v3_base.pt
(steps 1-2), calibration.json, training_log.json. The JSON files are updated, not replaced.
"""

import argparse
import copy
import json
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from data import CandidateSets, Embeddings, encode_sets, load_embeddings
from device import resolve_device
from models import CosineBaseline, OrderedHead, ReflexRankingHead, ReflexSetHead, save_head

EVAL_BATCH_SIZE = 1024
HEADS = ("set_head", "ordered_head", "reflex_v3")


@torch.no_grad()
def predict_logits(model, sets: CandidateSets, embeddings: Embeddings, device: str) -> torch.Tensor:
    model.eval()
    batches = torch.arange(len(sets)).split(EVAL_BATCH_SIZE)
    return torch.cat([model(*embeddings.gather(sets, index, device)).cpu() for index in batches])


def accuracy_by_type(logits: torch.Tensor, sets: CandidateSets, frame: pd.DataFrame) -> dict[str, float]:
    correct = pd.Series((logits.argmax(dim=1) == sets.target).numpy())
    by_type = correct.groupby(frame.open_set_type.to_numpy()).mean()
    return {**by_type.round(4).to_dict(), "all": round(correct.mean(), 4)}


def fit_temperature(logits: torch.Tensor, target: torch.Tensor) -> float:
    grid = torch.logspace(-3, 2, 501)
    losses = torch.stack([F.cross_entropy(logits / t, target) for t in grid])
    return float(grid[losses.argmin()])


def fit_threshold(scores: torch.Tensor, target: torch.Tensor, none_index: int) -> float:
    """Cosine threshold that maximizes validation accuracy; below it the answer is NONE_OF_THESE."""
    best_score, best_position = scores.max(dim=1)
    thresholds = torch.quantile(best_score, torch.linspace(0, 1, 1001))
    accuracy = torch.stack(
        [(torch.where(best_score >= t, best_position, none_index) == target).float().mean() for t in thresholds]
    )
    return float(thresholds[accuracy.argmax()])


def cross_entropy_loss(model, state, candidates, mask, target):
    return F.cross_entropy(model(state, candidates, mask), target)


def scorer_loss(model, state, candidates, mask, target, pair_weight: float):
    """Ranking cross-entropy among candidates (gold present only) + pair-match BCE on every real pair."""
    scores = model.candidate_logits(state, candidates, mask)
    kmax = mask.shape[1]
    present = target < kmax
    ranking = F.cross_entropy(scores[present], target[present]) if present.any() else scores.new_zeros(())
    pair_target = F.one_hot(target, kmax + 1)[:, :kmax].to(scores.dtype)  # gold slot 1, others 0
    pair = F.binary_cross_entropy_with_logits(scores[mask], pair_target[mask])
    return ranking + pair_weight * pair


@torch.no_grad()
def mean_loss(model, sets: CandidateSets, embeddings: Embeddings, device: str, loss_fn) -> float:
    model.eval()
    batches = torch.arange(len(sets)).split(EVAL_BATCH_SIZE)
    total = sum(loss_fn(model, *embeddings.gather(sets, index, device), sets.target[index].to(device)).item() * len(index)
                for index in batches)
    return total / len(sets)


def train_head(model, train: CandidateSets, epoch_rows: list[torch.Tensor], val: CandidateSets,
               embeddings: Embeddings, config: dict, device: str, loss_fn=cross_entropy_loss,
               parameters=None) -> list[dict]:
    generator = torch.Generator().manual_seed(config["seed"])
    optimizer = torch.optim.AdamW(model.parameters() if parameters is None else parameters, lr=config["learning_rate"])
    best_loss, best_state, stale, history = float("inf"), None, 0, []

    for epoch, rows in enumerate(epoch_rows):
        model.train()
        total = 0.0
        for index in rows[torch.randperm(len(rows), generator=generator)].split(config["train_batch_size"]):
            loss = loss_fn(model, *embeddings.gather(train, index, device), train.target[index].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)

        val_logits = predict_logits(model, val, embeddings, device)
        record = {"epoch": epoch, "train_loss": round(total / len(rows), 4)}
        if loss_fn is cross_entropy_loss:
            record["val_loss"] = round(F.cross_entropy(val_logits, val.target).item(), 4)
            record["val_accuracy"] = round((val_logits.argmax(dim=1) == val.target).float().mean().item(), 4)
        else:  # scorer only: the NONE logit is not trained yet, so report candidate ranking instead
            present = val.target < val.mask.shape[1]
            record["val_loss"] = round(mean_loss(model, val, embeddings, device, loss_fn), 4)
            record["val_rank1"] = round((val_logits[present, :-1].argmax(dim=1) == val.target[present]).float().mean().item(), 4)
        history.append(record)
        metric = "val_accuracy" if "val_accuracy" in record else "val_rank1"
        print(f"  epoch {epoch:2d}  train_loss {record['train_loss']:.4f}  "
              f"val_loss {record['val_loss']:.4f}  {metric} {record[metric]:.4f}")

        if record["val_loss"] < best_loss:
            best_loss, best_state, stale = record["val_loss"], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config["early_stopping_patience"]:
                print(f"  early stop; keeping epoch {epoch - stale}")
                break

    model.load_state_dict(best_state)
    return history


def epoch_index(frame: pd.DataFrame, epochs: int) -> list[torch.Tensor]:
    if frame.epoch.nunique() < epochs:
        raise RuntimeError("candidate sets have fewer epochs than config; rerun build_candidates.py --force")
    epoch = torch.from_numpy(frame.epoch.to_numpy())
    return [torch.nonzero(epoch == e).squeeze(1) for e in range(epochs)]


def train_reflex_v3(config: dict, embeddings: Embeddings, device: str, train: CandidateSets,
                    train_rows: list[torch.Tensor], val: CandidateSets, val_frame: pd.DataFrame,
                    max_k: int) -> tuple[ReflexRankingHead, ReflexRankingHead, dict]:
    processed = Path(config["processed_dir"])
    split = json.loads((processed / "v3_split.json").read_text())
    dim, hidden = embeddings.states.shape[1], config["head_hidden_dim"]
    loss = partial(scorer_loss, pair_weight=config["pair_match_weight"])
    log = {}

    def new_head() -> ReflexRankingHead:
        torch.manual_seed(config["seed"])
        return ReflexRankingHead(dim, hidden, config["none_head_hidden_dim"]).to(device)

    print("[train] reflex_v3 step 1: scorer on base labels")
    stage1 = pd.read_parquet(processed / "v3_stage1.parquet")
    base_val = (val_frame.label_id.isin(split["base_labels"]) | (val_frame.label_group == "oos")).to_numpy()
    base = new_head()
    log["reflex_v3_scorer_base"] = train_head(
        base, encode_sets(stage1, embeddings, max_k), epoch_index(stage1, config["epochs"]),
        encode_sets(val_frame[base_val], embeddings, max_k), embeddings, config, device, loss_fn=loss)

    print("[train] reflex_v3 step 2: NONE head on the frozen scorer (seen + pseudo-unseen labels)")
    stage2 = pd.read_parquet(processed / "v3_stage2.parquet")
    for name, parameter in base.named_parameters():
        parameter.requires_grad_(name.startswith("none_head."))
    log["reflex_v3_none_head"] = train_head(
        base, encode_sets(stage2, embeddings, max_k), epoch_index(stage2, config["epochs"]),
        val, embeddings, config, device, parameters=list(base.none_head.parameters()))
    base.requires_grad_(True)

    print("[train] reflex_v3 step 3: scorer on all training labels, NONE head from step 2")
    final = new_head()
    log["reflex_v3_scorer_final"] = train_head(final, train, train_rows, val, embeddings, config, device, loss_fn=loss)
    final.none_head.load_state_dict(base.none_head.state_dict())
    return final, base, log


def update_json(path: Path, values: dict) -> None:
    current = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**current, **values}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--heads", nargs="+", choices=HEADS, default=list(HEADS))
    parser.add_argument("--force", action="store_true", help="Retrain even if outputs exist")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    out_dir = Path(config["artifacts_dir"])
    files = {"set_head": ["set_head.pt"], "ordered_head": ["ordered_head.pt"],
             "reflex_v3": ["reflex_v3.pt", "reflex_v3_base.pt"]}
    outputs = [out_dir / name for head in args.heads for name in files[head]] + [out_dir / "calibration.json"]
    if all(path.exists() for path in outputs) and not args.force:
        print(f"[skip] trained heads already exist in {out_dir} (use --force to retrain)")
        return

    device = resolve_device(config["device"])
    max_k = max(config["candidate_sizes"])
    embeddings = load_embeddings(config, device)
    processed = Path(config["processed_dir"])
    train_frame = pd.read_parquet(processed / "train.parquet")
    val_frame = pd.read_parquet(processed / "validation.parquet")
    val_frame = val_frame[val_frame.label_group != "unseen"].reset_index(drop=True)

    train = encode_sets(train_frame, embeddings, max_k)
    val = encode_sets(val_frame, embeddings, max_k)
    epoch_rows = epoch_index(train_frame, config["epochs"])
    print(f"[data] {len(train)} training sets over {config['epochs']} epochs, {len(val)} validation sets, device {device}")

    cosine = CosineBaseline().to(device)
    threshold = fit_threshold(predict_logits(cosine, val, embeddings, device)[:, :max_k], val.target, max_k)
    cosine_logits = predict_logits(CosineBaseline(threshold).to(device), val, embeddings, device)
    calibration = {"cosine": {"threshold": threshold, "temperature": fit_temperature(cosine_logits, val.target)}}
    results = {"cosine": accuracy_by_type(cosine_logits, val, val_frame)}

    dim, hidden = embeddings.states.shape[1], config["head_hidden_dim"]
    makers = {"set_head": lambda: ReflexSetHead(dim, hidden), "ordered_head": lambda: OrderedHead(dim, hidden, max_k)}
    trained, log = {}, {}
    for name in [head for head in ("set_head", "ordered_head") if head in args.heads]:
        torch.manual_seed(config["seed"])
        trained[name] = makers[name]().to(device)
        print(f"[train] {name}")
        log[name] = train_head(trained[name], train, epoch_rows, val, embeddings, config, device)
    if "reflex_v3" in args.heads:
        trained["reflex_v3"], trained["reflex_v3_base"], v3_log = train_reflex_v3(
            config, embeddings, device, train, epoch_rows, val, val_frame, max_k)
        log.update(v3_log)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, model in trained.items():
        logits = predict_logits(model, val, embeddings, device)
        calibration[name] = {"temperature": fit_temperature(logits, val.target)}
        results[name] = accuracy_by_type(logits, val, val_frame)
        save_head(out_dir / f"{name}.pt", model)

    update_json(out_dir / "calibration.json", calibration)
    update_json(out_dir / "training_log.json", log)
    print(f"\nValidation accuracy (seen labels + natural OOS):\n{pd.DataFrame(results).to_string()}")
    print(f"[done] wrote {', '.join(str(path) for path in outputs)}")


if __name__ == "__main__":
    main()
