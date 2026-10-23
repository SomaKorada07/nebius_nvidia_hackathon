"""Load cached embeddings and turn candidate-set rows into index tensors."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass
class CandidateSets:
    """Candidate sets as row indexes into the embedding matrices."""

    state_idx: torch.Tensor  # (N,) row in the state matrix
    cand_idx: torch.Tensor  # (N, max_candidates) rows in the candidate matrix, 0 where padded
    mask: torch.Tensor  # (N, max_candidates) True for real candidates
    target: torch.Tensor  # (N,) gold candidate position, or max_candidates for NONE_OF_THESE

    def __len__(self) -> int:
        return len(self.target)


@dataclass
class Embeddings:
    states: torch.Tensor  # (num_texts, dim)
    candidates: torch.Tensor  # (num_label_texts, dim)
    state_row: dict[str, int]  # example text -> row
    label_row: dict[str, int]  # label ID -> row

    def gather(self, sets: CandidateSets, index: torch.Tensor, device: str) -> tuple[torch.Tensor, ...]:
        """(state, candidates, mask) tensors for the selected rows."""
        state = self.states[sets.state_idx[index].to(device)]
        candidates = self.candidates[sets.cand_idx[index].to(device)]
        return state, candidates, sets.mask[index].to(device)


def candidate_texts(config: dict) -> dict[str, str]:
    """The text embedded for each label: its raw name (v1) or "name: description" (v2)."""
    labels = json.loads((Path(config["standardized_dir"]) / "labels.json").read_text())
    if config.get("candidate_text", "name") == "name":
        return {label_id: info["label"] for label_id, info in labels.items()}
    descriptions = json.loads(Path(config["label_descriptions"]).read_text())
    return {label_id: f"{info['label']}: {descriptions[label_id]}" for label_id, info in labels.items()}


def load_embeddings(config: dict, device: str) -> Embeddings:
    emb_dir = Path(config["embeddings_dir"])
    state_index = pd.read_parquet(emb_dir / "state_index.parquet")
    candidate_index = pd.read_parquet(emb_dir / "candidate_index.parquet")
    text_row = dict(zip(candidate_index.text, candidate_index.row))
    return Embeddings(
        states=torch.from_numpy(np.load(emb_dir / "state_embeddings.npy")).to(device),
        candidates=torch.from_numpy(np.load(emb_dir / "candidate_embeddings.npy")).to(device),
        state_row=dict(zip(state_index.text, state_index.row)),
        label_row={label_id: text_row[text] for label_id, text in candidate_texts(config).items()},
    )


def encode_sets(frame: pd.DataFrame, embeddings: Embeddings, max_candidates: int) -> CandidateSets:
    n = len(frame)
    cand_idx = np.zeros((n, max_candidates), np.int64)
    mask = np.zeros((n, max_candidates), bool)
    target = np.full(n, max_candidates, np.int64)
    for i, (ids, target_id) in enumerate(zip(frame.candidate_ids, frame.target_id)):
        ids = list(ids)
        cand_idx[i, : len(ids)] = [embeddings.label_row[label_id] for label_id in ids]
        mask[i, : len(ids)] = True
        if isinstance(target_id, str):
            target[i] = ids.index(target_id)
    state_idx = np.array([embeddings.state_row[text] for text in frame.text])
    return CandidateSets(*(torch.from_numpy(a) for a in (state_idx, cand_idx, mask, target)))
