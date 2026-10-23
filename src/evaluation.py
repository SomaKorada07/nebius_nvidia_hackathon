"""Evaluation metrics shared by checkpoint analyses and experiments.

Every function takes a method's raw logits (shape (N, K + 1), last column NONE_OF_THESE) and,
where probabilities are needed, its temperature from calibration.json. Predictions are the argmax.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data import CandidateSets, Embeddings
from train import predict_logits

TYPES = ["closed", "gold_removed_hard", "gold_removed_random", "natural_oos"]
BOOTSTRAP = 1000
SHUFFLES = 10


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


def answer_probability(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrated probability that some listed option is correct, 1 - P(NONE_OF_THESE)."""
    return 1 - F.softmax(torch.from_numpy(logits) / temperature, dim=1)[:, -1].numpy()


def method_metrics(logits: np.ndarray, temperature: float, frame: pd.DataFrame, sets: CandidateSets) -> dict:
    target, mask = sets.target.numpy(), sets.mask.numpy()
    kmax = mask.shape[1]
    kind, group = frame.open_set_type.to_numpy(), frame.label_group.to_numpy()
    examples, source, k = frame.example_id.to_numpy(), frame.source.to_numpy(), frame.num_candidates.to_numpy()
    pred = logits.argmax(1)
    correct = (pred == target).astype(float)
    cand = np.where(mask, logits[:, :kmax], -np.inf)
    present = cand.max(1) - logits[:, kmax]  # threshold-free score: best option versus NONE
    sel = group != "unseen"

    out = {"accuracy": {t: boot(correct[sel & (kind == t)], examples[sel & (kind == t)]) for t in TYPES}}
    out["accuracy"]["all"] = boot(correct[sel], examples[sel])
    out["seen_vs_unseen"] = {
        g: {t: boot(correct[(group == g) & (kind == t)], examples[(group == g) & (kind == t)]) for t in TYPES[:3]}
        for g in ("seen", "unseen")
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
        "accuracy_hard_removed_unseen": m["seen_vs_unseen"]["unseen"]["gold_removed_hard"]["value"],
        "rank1_unseen": m["threshold_free"]["unseen"]["rank1_closed"]["value"],
        "auroc_hard_unseen": m["threshold_free"]["unseen"]["auroc_closed_vs_hard_removed"],
        "none_f1": m["none"]["f1"],
    }


def rejection_diagnostics(logits: np.ndarray, temperature: float, frame: pd.DataFrame, sets: CandidateSets) -> dict:
    """Does the model's belief that an answer is listed depend on whether the gold label was seen in training?"""
    kind, group = frame.open_set_type.to_numpy(), frame.label_group.to_numpy()
    probability = answer_probability(logits, temperature)
    pred, target = logits.argmax(1), sets.target.numpy()
    kmax = sets.mask.shape[1]
    out = {"mean_answer_probability": {
        t: {g: round(float(probability[(kind == t) & (group == g)].mean()), 4) for g in ("seen", "unseen")}
        for t in TYPES[:3]
    }}
    unseen_closed = (kind == "closed") & (group == "unseen")
    wrong = unseen_closed & (pred != target)
    out["unseen_closed_errors"] = {
        "errors": int(wrong.sum()), "sets": int(unseen_closed.sum()),
        "rejected_share": round(float((pred[wrong] == kmax).mean()), 4),
        "wrong_label_share": round(float((pred[wrong] != kmax).mean()), 4),
    }
    return out


def rejection_tradeoff(logits: np.ndarray, frame: pd.DataFrame, sets: CandidateSets, points: int = 201) -> dict:
    """Closed-set accuracy against false accepts on missing-answer sets, as the NONE logit is shifted.

    A constant shift of the NONE logit moves the operating point without changing the candidate
    ranking, so this traces every trade-off a rejection threshold could reach. Per label group:
    seen (missing = hard-removed, random-removed and natural OOS) and unseen (hard- and random-removed).
    """
    kmax = sets.mask.shape[1]
    cand = np.where(sets.mask.numpy(), logits[:, :kmax], -np.inf)
    margin, best, target = cand.max(1) - logits[:, kmax], cand.argmax(1), sets.target.numpy()
    kind, group = frame.open_set_type.to_numpy(), frame.label_group.to_numpy()
    shifts = np.unique(np.r_[np.quantile(margin, np.linspace(0, 1, points)), 0.0])
    out = {}
    for name, sel in {"seen": group != "unseen", "unseen": group == "unseen"}.items():
        closed, missing = sel & (kind == "closed"), sel & (kind != "closed")
        hit = (best == target)[closed]
        curve = [{"shift": float(t), "false_accept": float((margin[missing] > t).mean()),
                  "closed_accuracy": float(((margin[closed] > t) & hit).mean())} for t in shifts]
        curve.sort(key=lambda p: (p["false_accept"], p["closed_accuracy"]))
        fa, acc = np.array([p["false_accept"] for p in curve]), np.array([p["closed_accuracy"] for p in curve])
        own = next(p for p in curve if p["shift"] == 0.0)
        out[name] = {
            "curve": curve,
            "operating_point": {"false_accept": round(own["false_accept"], 4), "closed_accuracy": round(own["closed_accuracy"], 4)},
            "closed_accuracy_at_false_accept": {f"{x:.2f}": round(float(acc[fa <= x].max()), 4) if (fa <= x).any() else 0.0
                                                for x in (0.05, 0.10, 0.20)},
        }
    return out


def shuffled(sets: CandidateSets, rng: np.random.Generator) -> tuple[CandidateSets, np.ndarray]:
    """Shuffle the real candidates of every set (padding stays last); return sets and the slot permutation."""
    mask = sets.mask.numpy()
    keys = np.where(mask, rng.random(mask.shape), np.inf)
    perm = np.argsort(keys, axis=1, kind="stable")  # new slot j holds original slot perm[:, j]
    inverse = np.argsort(perm, axis=1)
    target, kmax = sets.target.numpy(), mask.shape[1]
    rows = np.arange(len(target))
    new_target = np.where(target < kmax, inverse[rows, np.minimum(target, kmax - 1)], target)
    new = CandidateSets(sets.state_idx, torch.from_numpy(np.take_along_axis(sets.cand_idx.numpy(), perm, 1)),
                        sets.mask, torch.from_numpy(new_target))
    return new, perm


def stability(model, temperature: float, sets: CandidateSets, emb: Embeddings, device: str) -> dict:
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
