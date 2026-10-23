#!/usr/bin/env python3
"""Reproduce every number and figure in checkpoints/v2/REPORT.md.

Usage (from the repository root, at git tag v2):
    python checkpoints/v2/analysis.py [--seeds 17 18 19 20 21]

Evaluates the frozen v2 weights (seed 17, candidates embedded as "name: description") and,
for comparison on identical candidate sets, the frozen v1 Reflex weights (label names). With
--seeds it runs the ablation: candidate text {name, name_description} x {Reflex set head,
ordered head}, one training run per seed and text in a temporary directory.
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
from evaluation import boot, headline, method_metrics, rejection_diagnostics, stability
from models import CosineBaseline, load_head
from train import predict_logits

V1 = ROOT / "checkpoints" / "v1"
TABLE = ["cosine", "ordered", "reflex_v1", "reflex_v2"]  # seed-17 tables
ABLATION = ["cosine", "ordered", "reflex"]
HEAD_FILE = {"ordered": "ordered_head", "reflex": "set_head"}
MODES = ["name", "name_description"]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COLOR = {"cosine": "#52514e", "ordered": "#eb6834", "reflex": "#2a78d6", "reflex_v2": "#2a78d6"}
NAME = {"cosine": "Cosine baseline", "ordered": "Ordered head", "reflex": "Reflex set head", "reflex_v2": "Reflex v2"}


def run_logits(weights_dir: Path, calibration: dict, sets, emb, device) -> tuple[dict, dict, dict]:
    models = {"cosine": CosineBaseline(calibration["cosine"]["threshold"]).to(device)}
    models.update({m: load_head(weights_dir / f"{f}.pt", device) for m, f in HEAD_FILE.items()})
    temps = {"cosine": calibration["cosine"]["temperature"],
             **{m: calibration[f]["temperature"] for m, f in HEAD_FILE.items()}}
    return {m: predict_logits(models[m], sets, emb, device).numpy() for m in ABLATION}, models, temps


# ---------------------------------------------------------------- figures

def style() -> None:
    plt.rcParams.update({
        "font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": "#c3c2b7", "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "legend.frameon": False, "hatch.linewidth": 1.0,
    })


def plot_loss(log: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, method in zip(axes, ["reflex", "ordered"]):
        h = pd.DataFrame(log[HEAD_FILE[method]]).set_index("epoch")
        kept = int(h.val_loss.idxmin())
        ax.plot(h.index, h.train_loss, "--", color=COLOR[method], lw=1.6)
        ax.plot(h.index, h.val_loss, "-", color=COLOR[method], lw=1.6)
        ax.plot([kept], [h.val_loss[kept]], "o", ms=7, color=COLOR[method], mec=SURFACE, mew=1.5, zorder=5)
        ax.annotate(f"kept: epoch {kept}\nval loss {h.val_loss[kept]:.3f}", (kept, h.val_loss[kept]),
                    xytext=(0, 14), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_title(NAME["reflex_v2" if method == "reflex" else method])
        ax.set_xlabel("epoch")
        ax.set_xticks(range(0, 15, 2))
        ax.set_xlim(-0.5, 14.5)
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_ylim(0, 2.0)
    handles = [plt.Line2D([], [], color=INK2, ls="--", lw=1.6), plt.Line2D([], [], color=INK2, ls="-", lw=1.6),
               plt.Line2D([], [], color=INK2, ls="", marker="o", ms=7, mec=SURFACE)]
    fig.legend(handles, ["train loss", "validation loss", "checkpoint kept"], loc="upper right", ncol=3, fontsize=8,
               bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Figure 1. Loss per epoch (v2, seed 17)", x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(CKPT / "figures" / "fig1_loss.png", dpi=200)
    plt.close(fig)


def plot_ablation(summary: dict) -> None:
    panels = [("accuracy_all", "Accuracy, seen labels + OOS"), ("accuracy_closed_unseen", "Closed accuracy, unseen labels"),
              ("accuracy_hard_removed_unseen", "Hard-removed accuracy, unseen labels")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    width = 0.36
    for ax, (key, title) in zip(axes, panels):
        for i, method in enumerate(ABLATION):
            for j, mode in enumerate(MODES):
                stat = summary[mode][method][key]
                x = i + (j - 0.5) * (width + 0.02)
                ax.bar(x, stat["mean"], width, color=COLOR[method] if mode == "name_description" else SURFACE,
                       edgecolor=COLOR[method], linewidth=1.4, hatch=None if mode == "name_description" else "///")
                if stat["std"] > 0:
                    ax.errorbar(x, stat["mean"], yerr=stat["std"], color=INK2, lw=1, capsize=2)
                ax.annotate(f"{100 * stat['mean']:.1f}", (x, stat["mean"] + stat["std"]), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(ABLATION)), [NAME[m] for m in ABLATION], fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("validation accuracy")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SURFACE, edgecolor=INK2, hatch="///"),
               plt.Rectangle((0, 0), 1, 1, facecolor=INK2, edgecolor=INK2)]
    fig.legend(handles, ["label name only (v1)", "name + description (v2)"], loc="upper right", ncol=2, fontsize=8,
               bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Figure 2. Effect of candidate descriptions (mean ± sd over 5 seeds)", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(CKPT / "figures" / "fig2_descriptions.png", dpi=200)
    plt.close(fig)


def plot_seen_unseen(metrics: dict) -> None:
    methods = ["cosine", "ordered", "reflex_v2"]
    panels = [("Closed-set accuracy", "closed"), ("Hard-removed accuracy (answer NONE)", "gold_removed_hard")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    width = 0.36
    for ax, (title, kind) in zip(axes, panels):
        for i, method in enumerate(methods):
            for j, g in enumerate(("seen", "unseen")):
                stat = metrics[method]["seen_vs_unseen"][g][kind]
                x = i + (j - 0.5) * (width + 0.02)
                ax.bar(x, stat["value"], width, color=COLOR[method] if g == "seen" else SURFACE,
                       edgecolor=COLOR[method], linewidth=1.4, hatch=None if g == "seen" else "///")
                low, high = stat["ci95"]
                ax.errorbar(x, stat["value"], yerr=[[stat["value"] - low], [high - stat["value"]]],
                            color=INK2, lw=1, capsize=2)
                ax.annotate(f"{100 * stat['value']:.1f}", (x, high), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(methods)), [NAME[m] for m in methods], fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("validation accuracy")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=INK2, edgecolor=INK2),
               plt.Rectangle((0, 0), 1, 1, facecolor=SURFACE, edgecolor=INK2, hatch="///")]
    fig.legend(handles, ["seen labels (trained on)", "unseen labels (held out)"], loc="upper right", ncol=2,
               fontsize=8, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Figure 3. Seen versus unseen labels (v2, seed 17, 95% CI)", x=0.01, ha="left", fontsize=11,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(CKPT / "figures" / "fig3_seen_vs_unseen.png", dpi=200)
    plt.close(fig)


def plot_stability(stab: dict) -> None:
    methods = ["cosine", "ordered", "reflex_v2"]
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
    fig.suptitle("Figure 4. Permutation stability (all 24,206 validation sets, v2, seed 17)", x=0.01, ha="left",
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
        "version": "v2",
        "created": date.today().isoformat(),
        "code": "git tag v2",
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
    parser.add_argument("--seeds", type=int, nargs="*", default=[], help="Ablation seeds")
    args = parser.parse_args()

    os.chdir(ROOT)
    config = yaml.safe_load((CKPT / "config.yaml").read_text())
    device = resolve_device(config["device"])
    max_k = max(config["candidate_sizes"])
    val = pd.read_parquet(Path(config["processed_dir"]) / "validation.parquet")
    target, examples = None, val.example_id.to_numpy()

    emb, sets = {}, {}
    for mode in MODES:
        emb[mode] = load_embeddings({**config, "candidate_text": mode}, device)
        sets[mode] = encode_sets(val, emb[mode], max_k)
    target = sets["name_description"].target.numpy()

    # v2 weights (descriptions) and v1 Reflex weights (names), on identical candidate sets
    v2_cal = json.loads((CKPT / "results" / "calibration.json").read_text())
    v2_logits, v2_models, v2_temps = run_logits(CKPT / "weights", v2_cal, sets["name_description"],
                                                emb["name_description"], device)
    v1_cal = json.loads((V1 / "results" / "calibration.json").read_text())
    v1_model = load_head(V1 / "weights" / "set_head.pt", device)
    logits = {"cosine": v2_logits["cosine"], "ordered": v2_logits["ordered"], "reflex_v2": v2_logits["reflex"],
              "reflex_v1": predict_logits(v1_model, sets["name"], emb["name"], device).numpy()}
    temps = {"cosine": v2_temps["cosine"], "ordered": v2_temps["ordered"], "reflex_v2": v2_temps["reflex"],
             "reflex_v1": v1_cal["set_head"]["temperature"]}
    method_sets = {m: sets["name"] if m == "reflex_v1" else sets["name_description"] for m in TABLE}

    metrics = {m: method_metrics(logits[m], temps[m], val, method_sets[m]) for m in TABLE}
    correct = {m: (logits[m].argmax(1) == target).astype(float) for m in TABLE}
    sel = (val.label_group != "unseen").to_numpy()
    subsets = {
        "all": sel,
        "closed_unseen": ((val.label_group == "unseen") & (val.open_set_type == "closed")).to_numpy(),
        "hard_removed_unseen": ((val.label_group == "unseen") & (val.open_set_type == "gold_removed_hard")).to_numpy(),
    }
    paired = {f"reflex_v2_minus_{other}_{name}": boot((correct["reflex_v2"] - correct[other])[mask], examples[mask])
              for other in ("reflex_v1", "ordered", "cosine") for name, mask in subsets.items()}

    stab = {"cosine": stability(v2_models["cosine"], temps["cosine"], sets["name_description"], emb["name_description"], device),
            "ordered": stability(v2_models["ordered"], temps["ordered"], sets["name_description"], emb["name_description"], device),
            "reflex_v2": stability(v2_models["reflex"], temps["reflex_v2"], sets["name_description"], emb["name_description"], device)}
    rejection = {m: rejection_diagnostics(logits[m], temps[m], val, method_sets[m]) for m in TABLE}

    runs = {}
    for mode in MODES if args.seeds else []:
        for seed in args.seeds:
            with tempfile.TemporaryDirectory() as tmp:
                run_config = {**config, "candidate_text": mode, "seed": seed, "artifacts_dir": tmp}
                (Path(tmp) / "config.yaml").write_text(yaml.safe_dump(run_config))
                subprocess.run([sys.executable, "src/train.py", "--config", f"{tmp}/config.yaml", "--force"],
                               check=True, capture_output=True)
                run_cal = json.loads((Path(tmp) / "calibration.json").read_text())
                r_logits, _, r_temps = run_logits(Path(tmp), run_cal, sets[mode], emb[mode], device)
                runs[f"{mode}/{seed}"] = {m: headline(method_metrics(r_logits[m], r_temps[m], val, sets[mode]))
                                          for m in ABLATION}
            print(f"[{mode} seed {seed}] " + "  ".join(f"{m} {runs[f'{mode}/{seed}'][m]['accuracy_all']}" for m in ABLATION))
    summary = {}
    for mode in MODES if runs else []:
        summary[mode] = {}
        for m in ABLATION:
            frame = pd.DataFrame([runs[f"{mode}/{s}"][m] for s in args.seeds])
            summary[mode][m] = {k: {"mean": round(float(frame[k].mean()), 4), "std": round(float(frame[k].std()), 4)}
                                for k in frame.columns}

    results = {
        "version": "v2", "split": "validation", "candidate_text": config["candidate_text"],
        "params": {"ordered": sum(p.numel() for p in v2_models["ordered"].parameters()),
                   "reflex": sum(p.numel() for p in v2_models["reflex"].parameters())},
        "temperatures": temps, "cosine_threshold": v2_cal["cosine"]["threshold"],
        "methods": metrics, "paired_differences": paired, "permutation_stability": stab,
        "rejection_diagnostics": rejection, "ablation_runs": runs, "ablation_summary": summary,
    }
    (CKPT / "results" / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")

    style()
    plot_loss(json.loads((CKPT / "results" / "training_log.json").read_text()))
    if summary:
        plot_ablation(summary)
    plot_seen_unseen(metrics)
    plot_stability(stab)
    write_manifest(config, device)
    print(f"[done] {CKPT / 'results' / 'metrics.json'}, figures, MANIFEST.json")


if __name__ == "__main__":
    main()
