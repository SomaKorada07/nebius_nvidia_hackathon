"""The open predictor must reproduce the evaluated model, and keep its answers order-free."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from data import encode_sets, load_embeddings
from models import ReflexRankingHead, save_head
from predict import CHECKPOINT, NONE_OF_THESE, ReflexPredictor, dataset_candidates
from train import predict_logits

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = CHECKPOINT / "weights" / "reflex_v3.pt"
EMBEDDINGS = ROOT / "artifacts" / "embeddings"

OPTIONS = {
    "card_arrival": "Ask when a card that was ordered will arrive.",
    "lost_or_stolen_card": "Report a card as lost or stolen.",
    "card_linking": "Link an existing card to the account.",
    "transfer_limit": "Ask what the maximum amount for a single transfer is.",
    "balance": "Check the current balance of a bank account.",
}


class HashEncoder:
    """Deterministic random unit vectors per text, so tests need no Nemotron model."""

    def __init__(self, dim: int):
        self.dim = dim

    def _vectors(self, texts, **_):
        seeds = [int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) for t in texts]
        vectors = np.stack([np.random.default_rng(s).standard_normal(self.dim) for s in seeds]).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    encode_query = encode_document = _vectors


class CachedEncoder:
    """Returns the cached training-time embeddings, to check the predictor end to end."""

    def __init__(self):
        self.vectors = {}
        for name in ("state", "candidate"):
            index = pd.read_parquet(EMBEDDINGS / f"{name}_index.parquet")
            array = np.load(EMBEDDINGS / f"{name}_embeddings.npy")
            self.vectors[name] = dict(zip(index.text, array[index.row.to_numpy()]))

    def encode_query(self, texts, **_):
        return np.stack([self.vectors["state"][t] for t in texts])

    def encode_document(self, texts, **_):
        return np.stack([self.vectors["candidate"][t] for t in texts])


@pytest.fixture
def random_predictor(tmp_path):
    torch.manual_seed(0)
    save_head(tmp_path / "head.pt", ReflexRankingHead(dim=64, hidden_dim=32))
    (tmp_path / "calibration.json").write_text(json.dumps({"reflex_v3": {"temperature": 0.8}}))
    return ReflexPredictor(tmp_path / "head.pt", tmp_path / "calibration.json", device="cpu", encoder=HashEncoder(64))


def test_probabilities_are_complete_and_order_free(random_predictor):
    base = random_predictor.predict_one("my card has not arrived", OPTIONS)
    assert base["answer"] in [*OPTIONS, NONE_OF_THESE]
    assert sum(base["candidates"].values()) + base["none_probability"] == pytest.approx(1.0, abs=1e-5)
    for seed in range(5):
        order = np.random.default_rng(seed).permutation(list(OPTIONS))
        result = random_predictor.predict_one("my card has not arrived", {name: OPTIONS[name] for name in order})
        assert result["answer"] == base["answer"]
        for name in OPTIONS:
            assert result["candidates"][name] == pytest.approx(base["candidates"][name], abs=1e-6)


def test_none_shift_only_moves_rejection(random_predictor):
    low, high = (random_predictor.predict_one("where is my card", OPTIONS, shift)["none_probability"] for shift in (-5, 5))
    assert low < high
    assert random_predictor.predict_one("where is my card", OPTIONS, 50.0)["answer"] == NONE_OF_THESE


def test_mixed_list_sizes_match_single_scoring(random_predictor):
    questions = ["where is my card", "what's my balance"]
    sets = [dict(list(OPTIONS.items())[:2]), OPTIONS]
    for question, options, result in zip(questions, sets, random_predictor.predict(questions, sets)):
        single = random_predictor.predict_one(question, options)
        assert single["answer"] == result["answer"]
        assert single["none_probability"] == pytest.approx(result["none_probability"], abs=1e-6)


@pytest.mark.parametrize("options", [
    {"balance": "Check the balance."},                           # one option
    {"balance": "Check the balance.", "bill_due": ""},          # missing description
    {"balance": "Check the balance.", NONE_OF_THESE: "None."},  # reserved name
])
def test_invalid_candidates_are_rejected(random_predictor, options):
    with pytest.raises(ValueError):
        random_predictor.predict_one("hi", options)


@pytest.mark.skipif(not (WEIGHTS.exists() and (EMBEDDINGS / "state_index.parquet").exists()),
                    reason="needs checkpoints/v3/weights and artifacts/embeddings")
def test_predictor_reproduces_evaluated_v3_probabilities():
    config = yaml.safe_load((CHECKPOINT / "config.yaml").read_text())
    frame = pd.read_parquet(ROOT / config["processed_dir"] / "validation.parquet").sample(300, random_state=0)
    options = [dataset_candidates(list(ids)) for ids in frame.candidate_ids]
    predictor = ReflexPredictor(device="cpu", encoder=CachedEncoder())
    results = predictor.predict(frame.text.tolist(), options)

    embeddings = load_embeddings(config, "cpu")
    logits = predict_logits(predictor.head, encode_sets(frame, embeddings, 10), embeddings, "cpu")
    expected = torch.softmax(logits / predictor.temperature, dim=1).numpy()
    for result, names, p in zip(results, options, expected):
        assert result["none_probability"] == pytest.approx(p[-1], abs=1e-4)
        assert [result["candidates"][n] for n in names] == pytest.approx(p[: len(names)].tolist(), abs=1e-4)
