"""The local option selectors.

Every model maps (state (B, D), candidates (B, K, D), mask (B, K)) to logits (B, K + 1):
one per candidate slot (-inf where padded) followed by one for NONE_OF_THESE.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

NONE_OF_THESE = "NONE_OF_THESE"


def with_none(candidate_logits: torch.Tensor, mask: torch.Tensor, none_logit: torch.Tensor) -> torch.Tensor:
    return torch.cat([candidate_logits.masked_fill(~mask, float("-inf")), none_logit], dim=1)


class CosineBaseline(nn.Module):
    """Pick the most similar candidate; NONE_OF_THESE wins when the best cosine is below the threshold."""

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.register_buffer("threshold", torch.tensor(float(threshold)))

    def forward(self, state, candidates, mask):
        # Embeddings are unit-normalized, so the dot product is the cosine similarity.
        scores = torch.einsum("bd,bkd->bk", state, candidates)
        return with_none(scores, mask, self.threshold.expand(len(state), 1))


class OrderedHead(nn.Module):
    """Laya-style, deliberately order-sensitive baseline.

    Each slot gets the same match features as the Reflex head, [c_i, s*c_i, |s-c_i|], but the
    slots are concatenated in the listed order and fed to one MLP, so every position has its own
    weights. (A plain MLP over [s, c_1, ..., c_K] never learned to match and always chose NONE.)
    """

    def __init__(self, dim: int, hidden_dim: int, max_candidates: int):
        super().__init__()
        self.kwargs = {"dim": dim, "hidden_dim": hidden_dim, "max_candidates": max_candidates}
        self.net = nn.Sequential(
            nn.Linear(dim * (3 * max_candidates + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_candidates + 1),
        )

    def forward(self, state, candidates, mask):
        s = state.unsqueeze(1).expand_as(candidates)
        slots = torch.cat([candidates, s * candidates, (s - candidates).abs()], dim=-1)
        slots = slots * mask.unsqueeze(-1)  # padded slots become zero vectors
        logits = self.net(torch.cat([state, slots.flatten(1)], dim=1))
        return with_none(logits[:, :-1], mask, logits[:, -1:])


class ReflexSetHead(nn.Module):
    """Deep Sets head: the same MLP scores every candidate, and the set summary is an order-free mean."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.kwargs = {"dim": dim, "hidden_dim": hidden_dim}
        self.shared_mlp = nn.Sequential(
            nn.Linear(4 * dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.shared_output = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.state_projection = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU())
        self.none_output = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state, candidates, mask):
        s = state.unsqueeze(1).expand_as(candidates)
        hidden = self.shared_mlp(torch.cat([s, candidates, s * candidates, (s - candidates).abs()], dim=-1))
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        summary = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        candidate_logits = self.shared_output(torch.cat([hidden, summary.unsqueeze(1).expand_as(hidden)], dim=-1))
        none_logit = self.none_output(torch.cat([self.state_projection(state), summary], dim=-1))
        return with_none(candidate_logits.squeeze(-1), mask, none_logit)


class ReflexRankingHead(nn.Module):
    """Reflex v3: the v2 Deep Sets scorer gives candidate logits; NONE_OF_THESE is scored from ranking evidence.

    The NONE logit is a tiny MLP over order-free statistics of the candidate scores, read detached so
    NONE never trains the scorer, and it never sees the state embedding itself. NONE stays the
    (K + 1)-th class of one softmax.
    """

    NONE_FEATURES = ("top1", "top1_minus_top2", "normalized_entropy", "normalized_logsumexp", "max_cosine", "log1p_k")

    def __init__(self, dim: int, hidden_dim: int, none_hidden_dim: int = 16):
        super().__init__()
        self.kwargs = {"dim": dim, "hidden_dim": hidden_dim, "none_hidden_dim": none_hidden_dim}
        self.shared_mlp = nn.Sequential(
            nn.Linear(4 * dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.shared_output = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.none_head = nn.Sequential(
            nn.Linear(len(self.NONE_FEATURES), none_hidden_dim), nn.ReLU(), nn.Linear(none_hidden_dim, 1)
        )

    def candidate_logits(self, state, candidates, mask):
        s = state.unsqueeze(1).expand_as(candidates)
        hidden = self.shared_mlp(torch.cat([s, candidates, s * candidates, (s - candidates).abs()], dim=-1))
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        summary = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        scores = self.shared_output(torch.cat([hidden, summary.unsqueeze(1).expand_as(hidden)], dim=-1))
        return scores.squeeze(-1).masked_fill(~mask, float("-inf"))

    @staticmethod
    def none_features(scores, state, candidates, mask):
        k = mask.sum(dim=1).to(scores.dtype)
        top2 = scores.topk(2, dim=1).values
        log_p = F.log_softmax(scores, dim=1).masked_fill(~mask, 0.0)  # entropy uses a fixed temperature of 1
        entropy = -(log_p.exp() * log_p).masked_fill(~mask, 0.0).sum(dim=1)
        cosine = torch.einsum("bd,bkd->bk", state, candidates).masked_fill(~mask, float("-inf"))
        return torch.stack([
            top2[:, 0],
            top2[:, 0] - top2[:, 1],
            entropy / k.log(),
            torch.logsumexp(scores, dim=1) - k.log(),
            cosine.max(dim=1).values,
            torch.log1p(k),
        ], dim=1)

    def forward(self, state, candidates, mask):
        scores = self.candidate_logits(state, candidates, mask)
        none_logit = self.none_head(self.none_features(scores.detach(), state, candidates, mask))
        return torch.cat([scores, none_logit], dim=1)


HEADS = {cls.__name__: cls for cls in (OrderedHead, ReflexSetHead, ReflexRankingHead)}


def save_head(path: Path, model: nn.Module) -> None:
    state_dict = {key: value.cpu() for key, value in model.state_dict().items()}
    torch.save({"class": type(model).__name__, "kwargs": model.kwargs, "state_dict": state_dict}, path)


def load_head(path: Path, device: str = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = HEADS[checkpoint["class"]](**checkpoint["kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()
