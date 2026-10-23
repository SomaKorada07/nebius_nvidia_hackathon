"""The Reflex head must be order-invariant, and saved heads must reload exactly."""

from pathlib import Path

import pytest
import torch

from models import OrderedHead, ReflexRankingHead, ReflexSetHead, load_head, save_head

ROOT = Path(__file__).resolve().parents[1]
TRAINED_SET_HEAD = ROOT / "artifacts" / "set_head.pt"
K = 10


def random_inputs(dim: int, real_counts=(10, 7, 5, 3)) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(0)
    state = torch.nn.functional.normalize(torch.randn(len(real_counts), dim, generator=generator), dim=-1)
    candidates = torch.nn.functional.normalize(torch.randn(len(real_counts), K, dim, generator=generator), dim=-1)
    mask = torch.zeros(len(real_counts), K, dtype=torch.bool)
    for row, count in enumerate(real_counts):
        mask[row, :count] = True
    return state, candidates, mask


@torch.no_grad()
def max_permutation_change(model, dim: int) -> float:
    """Largest logit change, mapped back to the original candidates, over 10 shuffles."""
    model = model.eval()
    state, candidates, mask = random_inputs(dim)
    base = model(state, candidates, mask)
    finite = torch.isfinite(base)
    worst = 0.0
    for seed in range(10):
        perm = torch.randperm(K, generator=torch.Generator().manual_seed(seed))
        out = model(state, candidates[:, perm], mask[:, perm])
        restored = out.clone()
        restored[:, perm] = out[:, :K]
        assert torch.equal(torch.isfinite(restored), finite)
        worst = max(worst, (restored[finite] - base[finite]).abs().max().item())
    return worst


def test_reflex_logits_are_permutation_invariant():
    torch.manual_seed(0)
    assert max_permutation_change(ReflexSetHead(dim=64, hidden_dim=32), dim=64) < 1e-5


@pytest.mark.skipif(not TRAINED_SET_HEAD.exists(), reason="run src/train.py first")
def test_trained_reflex_head_is_permutation_invariant():
    model = load_head(TRAINED_SET_HEAD)
    assert max_permutation_change(model, dim=model.kwargs["dim"]) < 1e-5


def test_ordered_head_is_order_sensitive():
    torch.manual_seed(0)
    assert max_permutation_change(OrderedHead(dim=64, hidden_dim=32, max_candidates=K), dim=64) > 1e-3


def test_saved_head_reloads_to_same_prediction(tmp_path):
    state, candidates, mask = random_inputs(dim=64)
    torch.manual_seed(0)
    for model in (ReflexSetHead(dim=64, hidden_dim=32), OrderedHead(dim=64, hidden_dim=32, max_candidates=K)):
        path = tmp_path / f"{type(model).__name__}.pt"
        save_head(path, model)
        with torch.no_grad():
            assert torch.equal(model.eval()(state, candidates, mask), load_head(path)(state, candidates, mask))


TRAINED_V3 = ROOT / "artifacts" / "reflex_v3.pt"


def test_reflex_v3_logits_are_permutation_invariant():
    torch.manual_seed(0)
    assert max_permutation_change(ReflexRankingHead(dim=64, hidden_dim=32), dim=64) < 1e-5


@pytest.mark.skipif(not TRAINED_V3.exists(), reason="run src/train.py first")
def test_trained_reflex_v3_is_permutation_invariant():
    model = load_head(TRAINED_V3)
    assert max_permutation_change(model, dim=model.kwargs["dim"]) < 1e-5


def test_reflex_v3_none_logit_does_not_train_the_scorer():
    state, candidates, mask = random_inputs(dim=64)
    torch.manual_seed(0)
    model = ReflexRankingHead(dim=64, hidden_dim=32)
    model(state, candidates, mask)[:, -1].sum().backward()
    for name, parameter in model.named_parameters():
        trained = parameter.grad is not None and parameter.grad.abs().sum() > 0
        assert trained == name.startswith("none_head."), name


def test_saved_reflex_v3_reloads_to_same_prediction(tmp_path):
    state, candidates, mask = random_inputs(dim=64)
    torch.manual_seed(0)
    model = ReflexRankingHead(dim=64, hidden_dim=32)
    save_head(tmp_path / "v3.pt", model)
    with torch.no_grad():
        assert torch.equal(model.eval()(state, candidates, mask), load_head(tmp_path / "v3.pt")(state, candidates, mask))
