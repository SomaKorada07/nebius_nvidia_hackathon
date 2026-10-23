#!/usr/bin/env python3
"""Reproduce every number and figure in checkpoints/v1/REPORT.md.

Usage (from the repository root, at git tag v1):
    python checkpoints/v1/analysis.py [--seeds 17 18 19 20 21]

Reads the frozen v1 heads in checkpoints/v1/weights/ and the validation candidate sets and
writes checkpoints/v1/results/metrics.json, checkpoints/v1/figures/*.png and
checkpoints/v1/MANIFEST.json. With --seeds it also retrains both heads once per seed in a
temporary directory (about 25 s each on an M4 Max) to measure seed variance.

The test split is never read.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from datetime import date
from importlib.metadata import version
from pathlib import Path

CKPT = Path(__file__).resolve().parent
ROOT = CKPT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score

from data import CandidateSets, encode_sets, load_embeddings
from device import resolve_device
from models import CosineBaseline, load_head
from train import predict_logits

TYPES = ["closed", "gold_removed_hard", "gold_removed_random", "natural_oos"]
METHODS = ["cosine", "reflex", "ordered"]
CAL_KEY = {"cosine": "cosine", "reflex": "set_head", "ordered": "ordered_head"}
BOOTSTRAP = 1000
SHUFFLES = 10

# Plot styling (validated categorical palette; neutral ink for the untrained baseline)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COLOR = {"reflex": "#2a78d6", "ordered": "#eb6834", "ordered_spec": "#1baf7a", "cosine": "#52514e"}
NAME = {"reflex": "Reflex set head", "ordered": "Ordered head", "ordered_spec": "Ordered head (spec version)",
        "cosine": "Cosine baseline"}


# ---------------------------------------------------------------- metrics

def boot(values: np.ndarray, groups: np.ndarray, seed: int = 0) -> dict:
    """Mean with a 95% cluster-bootstrap CI; examples (not sets) are resampled."""
    codes, inverse = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inverse, weights=values), np.bincount(inverse)
    index = np.random.default_rng(seed).integers(0, len(codes), (BOOTSTRAP, len(codes)))
    stats = sums[index].sum(1) / counts[index].sum(1)
    low, high = np.percentile(stats, [2.5, 97.5])
    return {"value": round(float(values.mean()), 4), "ci95": [round(float(low), 4), round(float(high), 4)],
            "n_sets": int(len(values)), "n_examples": int(len(codes))}


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    return round(float(roc_auc_score(labels, np.r_[positive, negative])), 4)


def method_metrics(logits: np.ndarray, temperature: float, frame: pd.DataFrame, sets: CandidateSets) -> dict:
    target, mask = sets.target.numpy(), sets.mask.numpy()
    kmax = mask.shape[1]
    kind, group = frame.open_set_type.to_numpy(), frame.label_group.to_numpy()
    examples, source, k = frame.example_id.to_numpy(), frame.source.to_numpy(), frame.num_candidates.to_numpy()
    pred = logits.argmax(1)
    correct = (pred == target).astype(float)
    cand = np.where(mask, logits[:, :kmax], -np.inf)
    present = cand.max(1) - logits[:, kmax]  # threshold-free "an option is correct" score
    sel = group != "unseen"

    out = {"accuracy": {t: boot(correct[sel & (kind == t)], examples[sel & (kind == t)]) for t in TYPES}}
    out["accuracy"]["all"] = boot(correct[sel], examples[sel])

    out["seen_vs_unseen"] = {
        g: {t: boot(correct[(group == g) & (kind == t)], examples[(group == g) & (kind == t)])
            for t in TYPES[:3]} for g in ("seen", "unseen")
    }

    none_pred, none_true = pred[sel] == kmax, target[sel] == kmax
    tp = (none_pred & none_true).sum()
    precision, recall = tp / max(none_pred.sum(), 1), tp / none_true.sum()
    closed_wrong = sel & (kind == "closed") & (correct == 0)
    out["none"] = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(2 * precision * recall / (precision + recall)), 4),
        "false_accept_rate": round(float(1 - recall), 4),
        "closed_errors_that_are_rejections": round(float((pred[closed_wrong] == kmax).mean()), 4),
    }

    out["threshold_free"] = {}
    for g in ("seen", "unseen"):
        closed, hard, rand = [(group == g) & (kind == t) for t in ("closed", "gold_removed_hard", "gold_removed_random")]
        out["threshold_free"][g] = {
            "rank1_closed": boot((cand.argmax(1) == target).astype(float)[closed], examples[closed]),
            "auroc_closed_vs_hard_removed": auroc(present[closed], present[hard]),
            "auroc_closed_vs_random_removed": auroc(present[closed], present[rand]),
        }

    by = {}
    for s in ("clinc150", "banking77"):
        by[s] = {t: round(float(correct[sel & (source == s) & (kind == t)].mean()), 4)
                 for t in TYPES if (sel & (source == s) & (kind == t)).any()}
        by[s]["all"] = round(float(correct[sel & (source == s)].mean()), 4)
    out["by_dataset"] = by
    out["by_list_size"] = {str(size): round(float(correct[sel & (k == size)].mean()), 4) for size in (5, 10)}

    t_logits, t_target = torch.from_numpy(logits[sel]), torch.from_numpy(target[sel])
    probs = F.softmax(t_logits / temperature, 1)
    confidence, hit = probs.max(1).values.numpy(), (probs.argmax(1) == t_target).numpy()
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(abs(hit[bins == b].mean() - confidence[bins == b].mean()) * (bins == b).mean()
              for b in range(15) if (bins == b).any())
    out["calibration"] = {
        "temperature": round(temperature, 4),
        "nll_before": round(F.cross_entropy(t_logits, t_target).item(), 4),
        "nll_after": round(F.cross_entropy(t_logits / temperature, t_target).item(), 4),
        "ece_after": round(float(ece), 4),
    }
    return out


def headline(m: dict) -> dict:
    return {
        "accuracy_all": m["accuracy"]["all"]["value"],
        "accuracy_closed_seen": m["seen_vs_unseen"]["seen"]["closed"]["value"],
        "accuracy_closed_unseen": m["seen_vs_unseen"]["unseen"]["closed"]["value"],
        "rank1_unseen": m["threshold_free"]["unseen"]["rank1_closed"]["value"],
        "auroc_hard_unseen": m["threshold_free"]["unseen"]["auroc_closed_vs_hard_removed"],
        "none_f1": m["none"]["f1"],
    }


# ---------------------------------------------------------------- permutation stability

def shuffled(sets: CandidateSets, rng: np.random.Generator) -> tuple[CandidateSets, np.ndarray]:
    """Shuffle the real candidates of every set (padding stays last); return sets and the slot permutation."""
    mask = sets.mask.numpy()
    keys = np.where(mask, rng.random(mask.shape), np.inf)
    perm = np.argsort(keys, axis=1, kind="stable")  # new slot j holds original slot perm[:, j]
    inverse = np.argsort(perm, axis=1)
    target = sets.target.numpy()
    kmax = mask.shape[1]
    rows = np.arange(len(target))
    new_target = np.where(target < kmax, inverse[rows, np.minimum(target, kmax - 1)], target)
    new = CandidateSets(sets.state_idx, torch.from_numpy(np.take_along_axis(sets.cand_idx.numpy(), perm, 1)),
                        sets.mask, torch.from_numpy(new_target))
    return new, perm


def stability(model, temperature: float, sets: CandidateSets, emb, device: str) -> dict:
    kmax = sets.mask.shape[1]
    rows = np.arange(len(sets))
    base_probs = F.softmax(predict_logits(model, sets, emb, device) / temperature, 1).numpy()
    base = base_probs.argmax(1)  # original slot or kmax (NONE)
    base_p = base_probs[rows, base]
    target = sets.target.numpy()
    flips, correct_all, prob_change = [], base == target, np.zeros(len(sets))
    rng = np.random.default_rng(0)
    for _ in range(SHUFFLES):
        new, perm = shuffled(sets, rng)
        probs = F.softmax(predict_logits(model, new, emb, device) / temperature, 1).numpy()
        pred = probs.argmax(1)
        answer = np.where(pred < kmax, perm[rows, np.minimum(pred, kmax - 1)], kmax)  # back to original slots
        flips.append(answer != base)
        correct_all &= answer == target
        inverse = np.argsort(perm, axis=1)
        slot = np.where(base < kmax, inverse[rows, np.minimum(base, kmax - 1)], kmax)
        prob_change = np.maximum(prob_change, np.abs(probs[rows, slot] - base_p))
    flips = np.stack(flips)
    return {
        "flip_rate": round(float(flips.mean()), 4),
        "sets_with_any_flip": round(float(flips.any(0).mean()), 4),
        "accuracy_unshuffled": round(float((base == target).mean()), 4),
        "worst_shuffle_accuracy": round(float(correct_all.mean()), 4),
        "max_prob_change_mean": round(float(prob_change.mean()), 6),
        "max_prob_change_max": round(float(prob_change.max()), 6),
    }


# ---------------------------------------------------------------- diagnostics

def familiarity(logits: np.ndarray, cosine_logits: np.ndarray, frame: pd.DataFrame, sets: CandidateSets,
                labels: dict) -> dict:
    """Reflex scores at equal cosine similarity, for seen versus unseen labels."""
    kind, group = frame.open_set_type.to_numpy(), frame.label_group.to_numpy()
    target = sets.target.numpy()
    gold = pd.DataFrame([(group[i], cosine_logits[i, target[i]], logits[i, target[i]])
                         for i in np.where(kind == "closed")[0]], columns=["group", "cosine", "logit"])
    gold["bin"] = pd.cut(gold.cosine, [-1, 0.3, 0.4, 0.5, 1]).astype(str)
    negatives = pd.DataFrame([(labels[l]["group"], cosine_logits[i, j], logits[i, j])
                              for i in np.where((kind == "gold_removed_random") & (group == "seen"))[0]
                              for j, l in enumerate(frame.candidate_ids.iloc[i])],
                             columns=["group", "cosine", "logit"])
    negatives["bin"] = pd.cut(negatives.cosine, [-1, 0.2, 0.3, 0.4, 1]).astype(str)
    table = lambda d: d.pivot_table(index="bin", columns="group", values="logit", aggfunc="mean").round(2).to_dict()
    pred = logits.argmax(1)
    kmax = sets.mask.shape[1]
    unseen_closed = np.where((kind == "closed") & (group == "unseen"))[0]
    wrong = unseen_closed[pred[unseen_closed] != target[unseen_closed]]
    picked = [frame.candidate_ids.iloc[i][pred[i]] for i in wrong if pred[i] != kmax]
    return {
        "gold_logit_by_cosine": table(gold),
        "negative_logit_by_cosine": table(negatives),
        "unseen_closed_errors": {
            "errors": int(len(wrong)), "sets": int(len(unseen_closed)),
            "rejected_share": round(float((pred[wrong] == kmax).mean()), 4),
            "wrong_label_share": round(float((pred[wrong] != kmax).mean()), 4),
            "wrong_label_was_seen_share": round(float(np.mean([labels[p]["group"] == "seen" for p in picked])), 4),
        },
    }


# ---------------------------------------------------------------- figures

def style() -> None:
    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": "#c3c2b7", "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "legend.frameon": False, "hatch.linewidth": 1.0,
    })


def kept_marker(ax, x, y, color):
    ax.plot([x], [y], "o", ms=7, color=color, mec=SURFACE, mew=1.5, zorder=5)


def plot_curves(log: dict, cosine_accuracy: float) -> None:
    kept = {name: min(h, key=lambda r: r["val_loss"])["epoch"] for name, h in log.items()}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, name in zip(axes, ["reflex", "ordered", "ordered_spec"]):
        h = pd.DataFrame(log[name]).set_index("epoch")
        ax.plot(h.index, h.train_loss, "--", color=COLOR[name], lw=1.6)
        ax.plot(h.index, h.val_loss, "-", color=COLOR[name], lw=1.6)
        kept_marker(ax, kept[name], h.val_loss[kept[name]], COLOR[name])
        ax.annotate(f"kept: epoch {kept[name]}\nval loss {h.val_loss[kept[name]]:.3f}",
                    (kept[name], h.val_loss[kept[name]]), xytext=(0, 14), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_title(NAME[name] + (" — discarded" if name == "ordered_spec" else ""))
        ax.set_xlabel("epoch")
        ax.set_xticks(range(0, 15, 2))
        ax.set_xlim(-0.5, 14.5)
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_ylim(0, 2.2)
    handles = [plt.Line2D([], [], color=INK2, ls="--", lw=1.6), plt.Line2D([], [], color=INK2, ls="-", lw=1.6),
               plt.Line2D([], [], color=INK2, ls="", marker="o", ms=7, mec=SURFACE)]
    fig.legend(handles, ["train loss", "validation loss", "checkpoint kept (lowest validation loss)"],
               loc="upper right", ncol=3, fontsize=8, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Figure 1. Loss per epoch", x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(CKPT / "figures" / "fig1_loss.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["reflex", "ordered", "ordered_spec"]:
        h = pd.DataFrame(log[name]).set_index("epoch")
        ax.plot(h.index, h.val_accuracy, "-", color=COLOR[name], lw=1.6, label=NAME[name])
        kept_marker(ax, kept[name], h.val_accuracy[kept[name]], COLOR[name])
        ax.annotate(NAME[name], (h.index[-1], h.val_accuracy.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.axhline(cosine_accuracy, color=COLOR["cosine"], ls=":", lw=1.2, label="Cosine baseline (no training)")
    ax.annotate(f"cosine baseline {cosine_accuracy:.3f}", (17.9, cosine_accuracy), xytext=(0, -10),
                textcoords="offset points", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (seen labels + OOS)")
    ax.set_xticks(range(0, 15, 2))
    ax.set_xlim(-0.5, 18)
    ax.set_ylim(0.6, 0.92)
    ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.0, 0.12))
    ax.set_title("Figure 2. Validation accuracy per epoch")
    fig.tight_layout()
    fig.savefig(CKPT / "figures" / "fig2_validation_accuracy.png", dpi=200)
    plt.close(fig)


def plot_seen_unseen(metrics: dict) -> None:
    panels = [("Closed-set accuracy", lambda m, g: m["seen_vs_unseen"][g]["closed"]),
              ("Gold ranked first (threshold-free)", lambda m, g: m["threshold_free"][g]["rank1_closed"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    width = 0.36
    for ax, (title, get) in zip(axes, panels):
        for i, method in enumerate(METHODS):
            for j, g in enumerate(("seen", "unseen")):
                stat = get(metrics[method], g)
                x = i + (j - 0.5) * (width + 0.02)
                ax.bar(x, stat["value"], width, color=COLOR[method] if g == "seen" else SURFACE,
                       edgecolor=COLOR[method], linewidth=1.4, hatch=None if g == "seen" else "///")
                low, high = stat["ci95"]
                ax.errorbar(x, stat["value"], yerr=[[stat["value"] - low], [high - stat["value"]]],
                            color=INK2, lw=1, capsize=2)
                ax.annotate(f"{stat['value']:.3f}", (x, high), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(METHODS)), [NAME[m] for m in METHODS])
        ax.set_title(title)
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("validation, closed sets")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=INK2, edgecolor=INK2),
               plt.Rectangle((0, 0), 1, 1, facecolor=SURFACE, edgecolor=INK2, hatch="///")]
    fig.legend(handles, ["seen labels (trained on)", "unseen labels (held out)"], loc="upper right", ncol=2,
               fontsize=8, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Figure 3. Seen versus unseen labels (95% CI)", x=0.01, ha="left", fontsize=11,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(CKPT / "figures" / "fig3_seen_vs_unseen.png", dpi=200)
    plt.close(fig)


def plot_stability(stab: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (key, title) in zip(axes, [("sets_with_any_flip", "Sets whose answer changes in ≥1 of 10 shuffles"),
                                       ("worst_shuffle_accuracy", "Worst-shuffle accuracy")]):
        values = [stab[m][key] for m in METHODS]
        ax.bar(range(len(METHODS)), values, 0.55, color=[COLOR[m] for m in METHODS])
        for i, v in enumerate(values):
            ax.annotate(f"{v:.3f}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(range(len(METHODS)), [NAME[m] for m in METHODS])
        ax.set_title(title)
        ax.set_ylim(0, max(1.0, max(values) * 1.15) if key == "worst_shuffle_accuracy" else max(values) * 1.25 + 0.01)
    fig.suptitle("Figure 4. Permutation stability (all 24,206 validation sets)", x=0.01, ha="left", fontsize=11,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(CKPT / "figures" / "fig4_permutation_stability.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- manifest

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(config: dict, device: str) -> None:
    files = [Path("data/raw/clinc150/data_full.json"), Path("data/raw/clinc150/domains.json"),
             Path("data/raw/banking77/train.csv"), Path("data/raw/banking77/test.csv")]
    for folder in (config["standardized_dir"], config["processed_dir"], config["embeddings_dir"]):
        files += sorted(p for p in Path(folder).iterdir() if p.is_file())
    files += sorted((CKPT / "weights").iterdir()) + [CKPT / "config.yaml"]
    manifest = {
        "version": "v1",
        "created": date.today().isoformat(),
        "code": "git tag v1",
        "encoder": {"model_id": config["model_id"], "revision": config["model_revision"]},
        "seed": config["seed"],
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(), "device": device,
            **{pkg: version(pkg) for pkg in ("torch", "transformers", "sentence-transformers", "numpy", "pandas",
                                             "scikit-learn", "pyarrow")},
        },
        "sha256": {str(p.relative_to(ROOT) if p.is_absolute() else p): sha256(ROOT / p) for p in files},
    }
    (CKPT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


# ---------------------------------------------------------------- main

def evaluate_heads(weights_dir: Path, calibration: dict, val: pd.DataFrame, sets, emb, device) -> dict:
    models = {"cosine": CosineBaseline(calibration["cosine"]["threshold"]).to(device),
              "reflex": load_head(weights_dir / "set_head.pt", device),
              "ordered": load_head(weights_dir / "ordered_head.pt", device)}
    return {m: predict_logits(models[m], sets, emb, device).numpy() for m in METHODS}, models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="*", default=[], help="Retrain with these seeds for variance")
    args = parser.parse_args()

    import os
    os.chdir(ROOT)
    config = yaml.safe_load((CKPT / "config.yaml").read_text())
    device = resolve_device(config["device"])
    labels = json.loads((Path(config["standardized_dir"]) / "labels.json").read_text())
    emb = load_embeddings(config, device)
    val = pd.read_parquet(Path(config["processed_dir"]) / "validation.parquet")
    sets = encode_sets(val, emb, max(config["candidate_sizes"]))
    calibration = json.loads((CKPT / "results" / "calibration.json").read_text())

    logits, models = evaluate_heads(CKPT / "weights", calibration, val, sets, emb, device)
    metrics = {m: method_metrics(logits[m], calibration[CAL_KEY[m]]["temperature"], val, sets) for m in METHODS}

    examples = val.example_id.to_numpy()
    sel = (val.label_group != "unseen").to_numpy()
    target = sets.target.numpy()
    correct = {m: (logits[m].argmax(1) == target).astype(float) for m in METHODS}
    paired = {
        "reflex_minus_ordered_all": boot((correct["reflex"] - correct["ordered"])[sel], examples[sel]),
        "reflex_minus_cosine_all": boot((correct["reflex"] - correct["cosine"])[sel], examples[sel]),
    }
    unseen_closed = ((val.label_group == "unseen") & (val.open_set_type == "closed")).to_numpy()
    paired["reflex_minus_ordered_closed_unseen"] = boot(
        (correct["reflex"] - correct["ordered"])[unseen_closed], examples[unseen_closed])

    stab = {m: stability(models[m], calibration[CAL_KEY[m]]["temperature"], sets, emb, device) for m in METHODS}
    diagnostics = familiarity(logits["reflex"], logits["cosine"], val, sets, labels)

    seeds = {}
    for seed in args.seeds:
        with tempfile.TemporaryDirectory() as tmp:
            seed_config = {**config, "seed": seed, "artifacts_dir": tmp}
            (Path(tmp) / "config.yaml").write_text(yaml.safe_dump(seed_config))
            subprocess.run([sys.executable, "src/train.py", "--config", f"{tmp}/config.yaml", "--force"],
                           check=True, capture_output=True)
            seed_cal = json.loads((Path(tmp) / "calibration.json").read_text())
            seed_logits, _ = evaluate_heads(Path(tmp), seed_cal, val, sets, emb, device)
            seeds[seed] = {m: headline(method_metrics(seed_logits[m], seed_cal[CAL_KEY[m]]["temperature"], val, sets))
                           for m in ("reflex", "ordered")}
        print(f"[seed {seed}] reflex {seeds[seed]['reflex']['accuracy_all']}  ordered {seeds[seed]['ordered']['accuracy_all']}")
    seed_summary = {}
    if seeds:
        for m in ("reflex", "ordered"):
            frame = pd.DataFrame([seeds[s][m] for s in seeds])
            seed_summary[m] = {k: {"mean": round(float(frame[k].mean()), 4), "std": round(float(frame[k].std()), 4)}
                               for k in frame.columns}

    results = {
        "version": "v1", "split": "validation", "bootstrap_resamples": BOOTSTRAP, "shuffles": SHUFFLES,
        "params": {m: sum(p.numel() for p in models[m].parameters()) for m in METHODS},
        "methods": metrics, "paired_differences": paired, "permutation_stability": stab,
        "reflex_familiarity_diagnostics": diagnostics,
        "seed_runs": {str(s): v for s, v in seeds.items()}, "seed_summary": seed_summary,
    }
    (CKPT / "results" / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")

    style()
    log = json.loads((CKPT / "results" / "training_log.json").read_text())
    spec = json.loads((CKPT / "results" / "ordered_head_spec_version_log.json").read_text())["history"]
    plot_curves({"reflex": log["set_head"], "ordered": log["ordered_head"], "ordered_spec": spec},
                metrics["cosine"]["accuracy"]["all"]["value"])
    plot_seen_unseen(metrics)
    plot_stability(stab)
    write_manifest(config, device)
    print(f"[done] {CKPT / 'results' / 'metrics.json'}, figures, MANIFEST.json")


if __name__ == "__main__":
    main()
