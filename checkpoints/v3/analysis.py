#!/usr/bin/env python3
"""Reproduce every number and figure in checkpoints/v3/REPORT.md.

Usage (from the repository root, at git tag v3):
    python checkpoints/v3/analysis.py [--seeds 17 18 19 20 21]

Evaluates the frozen v3 weights (seed 17) and, on identical candidate sets, the frozen v2 weights
(cosine threshold, ordered head, Reflex v2). With --seeds it retrains Reflex v3 once per seed with
pair-match on and off, in temporary directories. Five-seed reference numbers for v2, the ordered
head and cosine come from checkpoints/v2/results/metrics.json (same protocol and data).
Writes results/metrics.json, figures/*.png and MANIFEST.json.

The test split is never read.
"""

import argparse
import hashlib
import json
import os
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
import yaml

from data import encode_sets, load_embeddings
from device import resolve_device
from evaluation import boot, headline, method_metrics, rejection_diagnostics, rejection_tradeoff, stability
from models import CosineBaseline, load_head
from train import predict_logits

V2 = ROOT / "checkpoints" / "v2"
TABLE = ["cosine", "ordered", "reflex_v2", "reflex_v3"]
GROUPS = ("base", "pseudo", "unseen")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COLOR = {"cosine": "#52514e", "ordered": "#eb6834", "reflex_v2": "#2a78d6", "reflex_v3": "#1baf7a"}
NAME = {"cosine": "Cosine baseline", "ordered": "Ordered head", "reflex_v2": "Reflex v2", "reflex_v3": "Reflex v3"}


def label_groups(val: pd.DataFrame, split: dict) -> np.ndarray:
    """base / pseudo (training labels simulated as unseen for the NONE head) / unseen / oos, per row."""
    return np.where(val.label_id.isin(split["pseudo_unseen_labels"]), "pseudo",
                    np.where(val.label_id.isin(split["base_labels"]), "base",
                             np.where(val.label_group == "unseen", "unseen", "oos")))


def group_breakdown(logits: np.ndarray, sets, kind: np.ndarray, groups: np.ndarray) -> dict:
    target = sets.target.numpy()
    pred, cand = logits.argmax(1), np.where(sets.mask.numpy(), logits[:, :-1], -np.inf)
    out = {}
    for g in GROUPS:
        closed, hard = (groups == g) & (kind == "closed"), (groups == g) & (kind == "gold_removed_hard")
        out[g] = {"closed_accuracy": round(float((pred == target)[closed].mean()), 4),
                  "hard_removed_accuracy": round(float((pred == target)[hard].mean()), 4),
                  "rank1": round(float((cand.argmax(1) == target)[closed].mean()), 4)}
    return out


def summarize(rows: list[dict]) -> dict:
    frame = pd.json_normalize(rows)
    return {k: {"mean": round(float(frame[k].mean()), 4), "std": round(float(frame[k].std()), 4)} for k in frame.columns}


# ---------------------------------------------------------------- figures

def style() -> None:
    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": "#c3c2b7", "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "legend.frameon": False,
    })


def plot_training(log: dict) -> None:
    panels = [("reflex_v3_scorer_base", "Step 1: scorer on base labels", "ranking + pair-match loss"),
              ("reflex_v3_none_head", "Step 2: NONE head (frozen scorer)", "(K+1)-way cross-entropy"),
              ("reflex_v3_scorer_final", "Step 3: scorer on all labels", "ranking + pair-match loss")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (key, title, ylabel) in zip(axes, panels):
        h = pd.DataFrame(log[key]).set_index("epoch")
        kept = int(h.val_loss.idxmin())
        ax.plot(h.index, h.train_loss, "--", color=COLOR["reflex_v3"], lw=1.6)
        ax.plot(h.index, h.val_loss, "-", color=COLOR["reflex_v3"], lw=1.6)
        ax.plot([kept], [h.val_loss[kept]], "o", ms=7, color=COLOR["reflex_v3"], mec=SURFACE, mew=1.5, zorder=5)
        ax.text(0.97, 0.95, f"kept: epoch {kept}\nval loss {h.val_loss[kept]:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=INK2)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(0, 15, 2))
        ax.set_xlim(-0.5, 14.5)
        ax.set_ylim(0, max(h.train_loss.max(), h.val_loss.max()) * 1.15)
    handles = [plt.Line2D([], [], color=INK2, ls="--", lw=1.6), plt.Line2D([], [], color=INK2, ls="-", lw=1.6),
               plt.Line2D([], [], color=INK2, ls="", marker="o", ms=7, mec=SURFACE)]
    fig.legend(handles, ["train loss", "validation loss", "checkpoint kept"], loc="upper right", ncol=3, fontsize=8,
               bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Figure 1. Reflex v3 training, three steps (seed 17)", x=0.01, ha="left", fontsize=11,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(CKPT / "figures" / "fig1_training.png", dpi=200)
    plt.close(fig)


def plot_comparison(summary: dict) -> None:
    panels = [("accuracy_all", "Accuracy, seen labels + OOS"), ("accuracy_closed_unseen", "Closed accuracy, unseen labels"),
              ("accuracy_hard_removed_unseen", "Hard-removed accuracy, unseen labels")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        for i, method in enumerate(TABLE):
            stat = summary[method][key]
            ax.bar(i, stat["mean"], 0.62, color=COLOR[method])
            if stat["std"] > 0:
                ax.errorbar(i, stat["mean"], yerr=stat["std"], color=INK2, lw=1, capsize=2)
            ax.annotate(f"{100 * stat['mean']:.1f}", (i, stat["mean"] + stat["std"]), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(TABLE)), [NAME[m].replace(" baseline", "\nbaseline").replace(" head", "\nhead") for m in TABLE],
                      fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("validation accuracy")
    fig.suptitle("Figure 2. Five-seed comparison, candidates as name + description (mean ± sd)", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(CKPT / "figures" / "fig2_comparison.png", dpi=200)
    plt.close(fig)


def plot_tradeoff(tradeoff: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, group, title in zip(axes, ("seen", "unseen"), ("Seen labels (+ natural OOS)", "Unseen labels")):
        for method in TABLE:
            curve = tradeoff[method][group]["curve"]
            ax.plot([p["false_accept"] for p in curve], [p["closed_accuracy"] for p in curve], color=COLOR[method],
                    lw=1.8, label=NAME[method])
            own = tradeoff[method][group]["operating_point"]
            ax.plot([own["false_accept"]], [own["closed_accuracy"]], "o", ms=7, color=COLOR[method], mec=SURFACE,
                    mew=1.5, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("false-accept rate on missing-answer sets")
        ax.set_xlim(0, 0.6)
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="x", color=GRID, lw=0.6)
    axes[0].set_ylabel("closed-set accuracy")
    handles = [plt.Line2D([], [], color=COLOR[m], lw=1.8) for m in TABLE]
    handles.append(plt.Line2D([], [], color=INK2, ls="", marker="o", ms=7, mec=SURFACE))
    fig.legend(handles, [NAME[m] for m in TABLE] + ["each model's own operating point"], loc="upper left", ncol=5,
               fontsize=8, bbox_to_anchor=(0.005, 0.93))
    fig.suptitle("Figure 3. Rejection trade-off as the NONE logit is shifted (seed 17)", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(CKPT / "figures" / "fig3_tradeoff.png", dpi=200)
    plt.close(fig)


def plot_stability(stab: dict) -> None:
    methods = ["cosine", "ordered", "reflex_v3"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (key, title) in zip(axes, [("sets_with_any_flip", "Sets whose answer changes in ≥1 of 10 shuffles"),
                                       ("worst_shuffle_accuracy", "Worst-shuffle accuracy")]):
        values = [stab[m][key] for m in methods]
        ax.bar(range(len(methods)), values, 0.55, color=[COLOR[m] for m in methods])
        for i, v in enumerate(values):
            ax.annotate(f"{100 * v:.1f}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(range(len(methods)), [NAME[m] for m in methods])
        ax.set_title(title)
        ax.set_ylim(0, 1.0 if key == "worst_shuffle_accuracy" else max(values) * 1.3 + 0.01)
    fig.suptitle("Figure 4. Permutation stability (all 24,206 validation sets, seed 17)", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold")
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
             Path("data/raw/banking77/train.csv"), Path("data/raw/banking77/test.csv"),
             Path(config["label_descriptions"])]
    for folder in (config["standardized_dir"], config["processed_dir"], config["embeddings_dir"]):
        files += sorted(p.relative_to(ROOT) for p in (ROOT / folder).iterdir() if p.is_file())
    files += [p.relative_to(ROOT) for p in sorted((CKPT / "weights").iterdir())] + [(CKPT / "config.yaml").relative_to(ROOT)]
    manifest = {
        "version": "v3",
        "created": date.today().isoformat(),
        "code": "git tag v3",
        "encoder": {"model_id": config["model_id"], "revision": config["model_revision"]},
        "candidate_text": config["candidate_text"],
        "seed": config["seed"],
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(), "device": device,
            **{pkg: version(pkg) for pkg in ("torch", "transformers", "sentence-transformers", "numpy", "pandas",
                                             "scikit-learn", "pyarrow")},
        },
        "sha256": {str(p): sha256(ROOT / p) for p in files},
    }
    (CKPT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="*", default=[], help="Seeds for the five-seed runs")
    args = parser.parse_args()

    os.chdir(ROOT)
    config = yaml.safe_load((CKPT / "config.yaml").read_text())
    device = resolve_device(config["device"])
    max_k = max(config["candidate_sizes"])
    val = pd.read_parquet(Path(config["processed_dir"]) / "validation.parquet")
    split = json.loads((Path(config["processed_dir"]) / "v3_split.json").read_text())
    emb = load_embeddings(config, device)
    sets = encode_sets(val, emb, max_k)
    target, examples, kind = sets.target.numpy(), val.example_id.to_numpy(), val.open_set_type.to_numpy()
    groups = label_groups(val, split)

    v2_cal = json.loads((V2 / "results" / "calibration.json").read_text())
    v3_cal = json.loads((CKPT / "results" / "calibration.json").read_text())
    models = {
        "cosine": CosineBaseline(v2_cal["cosine"]["threshold"]).to(device),
        "ordered": load_head(V2 / "weights" / "ordered_head.pt", device),
        "reflex_v2": load_head(V2 / "weights" / "set_head.pt", device),
        "reflex_v3": load_head(CKPT / "weights" / "reflex_v3.pt", device),
        "reflex_v3_no_retrain": load_head(CKPT / "weights" / "reflex_v3_base.pt", device),
    }
    temps = {"cosine": v2_cal["cosine"]["temperature"], "ordered": v2_cal["ordered_head"]["temperature"],
             "reflex_v2": v2_cal["set_head"]["temperature"], "reflex_v3": v3_cal["reflex_v3"]["temperature"],
             "reflex_v3_no_retrain": v3_cal["reflex_v3_base"]["temperature"]}
    logits = {m: predict_logits(model, sets, emb, device).numpy() for m, model in models.items()}

    metrics = {m: method_metrics(logits[m], temps[m], val, sets) for m in logits}
    correct = {m: (logits[m].argmax(1) == target).astype(float) for m in logits}
    sel = (val.label_group != "unseen").to_numpy()
    subsets = {"all": sel,
               "closed_unseen": ((val.label_group == "unseen") & (val.open_set_type == "closed")).to_numpy(),
               "hard_removed_unseen": ((val.label_group == "unseen") & (val.open_set_type == "gold_removed_hard")).to_numpy()}
    paired = {f"reflex_v3_minus_{other}_{name}": boot((correct["reflex_v3"] - correct[other])[mask], examples[mask])
              for other in ("reflex_v2", "ordered", "cosine") for name, mask in subsets.items()}
    stab = {m: stability(models[m], temps[m], sets, emb, device) for m in ("cosine", "ordered", "reflex_v3")}
    rejection = {m: rejection_diagnostics(logits[m], temps[m], val, sets) for m in TABLE}
    tradeoff = {m: rejection_tradeoff(logits[m], val, sets) for m in TABLE}
    breakdown = {m: group_breakdown(logits[m], sets, kind, groups) for m in ("reflex_v3", "reflex_v3_no_retrain")}

    runs = {}
    for weight in (1.0, 0.0) if args.seeds else []:
        for seed in args.seeds:
            with tempfile.TemporaryDirectory() as tmp:
                run_config = {**config, "seed": seed, "pair_match_weight": weight, "artifacts_dir": tmp}
                (Path(tmp) / "config.yaml").write_text(yaml.safe_dump(run_config))
                subprocess.run([sys.executable, "src/train.py", "--config", f"{tmp}/config.yaml", "--heads",
                                "reflex_v3", "--force"], check=True, capture_output=True)
                cal = json.loads((Path(tmp) / "calibration.json").read_text())
                for variant, file in (("final", "reflex_v3"), ("no_retrain", "reflex_v3_base")):
                    run_logits = predict_logits(load_head(Path(tmp) / f"{file}.pt", device), sets, emb, device).numpy()
                    runs[f"pair_match_{weight:g}/{variant}/{seed}"] = {
                        **headline(method_metrics(run_logits, cal[file]["temperature"], val, sets)),
                        "groups": group_breakdown(run_logits, sets, kind, groups)}
            print(f"[pair_match {weight:g} seed {seed}] reflex_v3 "
                  f"{runs[f'pair_match_{weight:g}/final/{seed}']['accuracy_all']}", flush=True)
    summary = {}
    for key in sorted({k.rsplit("/", 1)[0] for k in runs}):
        summary[key] = summarize([runs[f"{key}/{s}"] for s in args.seeds])

    v2_runs = json.loads((V2 / "results" / "metrics.json").read_text())["ablation_runs"]
    reference = {}
    for method, source in (("cosine", "cosine"), ("ordered", "ordered"), ("reflex_v2", "reflex")):
        reference[method] = summarize([v2_runs[f"name_description/{s}"][source] for s in (17, 18, 19, 20, 21)])
    if summary:
        reference["reflex_v3"] = {k: v for k, v in summary["pair_match_1/final"].items() if not k.startswith("groups")}

    results = {
        "version": "v3", "split": "validation", "candidate_text": config["candidate_text"],
        "params": {m: sum(p.numel() for p in models[m].parameters()) for m in ("ordered", "reflex_v2", "reflex_v3")},
        "temperatures": temps, "cosine_threshold": v2_cal["cosine"]["threshold"],
        "label_split": {"base_labels": len(split["base_labels"]), "pseudo_unseen_labels": len(split["pseudo_unseen_labels"]),
                        "held_out_texts": len(split["held_out_example_ids"])},
        "methods": metrics, "paired_differences": paired, "permutation_stability": stab,
        "rejection_diagnostics": rejection, "rejection_tradeoff": tradeoff, "group_breakdown": breakdown,
        "seed_runs": runs, "seed_summary": summary, "five_seed_comparison": reference,
    }
    (CKPT / "results" / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")

    style()
    plot_training(json.loads((CKPT / "results" / "training_log.json").read_text()))
    if summary:
        plot_comparison(reference)
    plot_tradeoff(tradeoff)
    plot_stability(stab)
    write_manifest(config, device)
    print(f"[done] {CKPT / 'results' / 'metrics.json'}, figures, MANIFEST.json")


if __name__ == "__main__":
    main()
