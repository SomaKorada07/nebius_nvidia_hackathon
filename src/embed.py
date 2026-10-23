#!/usr/bin/env python3
"""Embed every unique example text and label with the local Nemotron model.

Usage:
    python src/embed.py --config config.yaml [--force]

Example texts are encoded with encode_query and candidate texts with encode_document. Writes
float32 arrays plus indexes (row, text_hash, text) to artifacts/embeddings/:
    state_embeddings.npy, state_index.parquet          one row per unique example text
    candidate_embeddings.npy, candidate_index.parquet  one row per unique candidate text: every
                                                       raw label name and every "name: description"

Texts already in the cache are skipped, so an interrupted run resumes where it stopped
and a finished run does nothing. --force re-embeds everything.
"""

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data import candidate_texts
from device import resolve_device

DIMENSION = 2048
CHUNK_SIZE = 4096  # texts embedded between cache saves
SPLITS = ("train", "validation", "test")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def empty_cache() -> tuple[pd.DataFrame, np.ndarray]:
    index = pd.DataFrame({"row": pd.Series(dtype="int64"), "text_hash": pd.Series(dtype=str), "text": pd.Series(dtype=str)})
    return index, np.empty((0, DIMENSION), np.float32)


def load_cache(out_dir: Path, name: str) -> tuple[pd.DataFrame, np.ndarray]:
    index_path, array_path = out_dir / f"{name}_index.parquet", out_dir / f"{name}_embeddings.npy"
    if not (index_path.exists() and array_path.exists()):
        return empty_cache()
    index, array = pd.read_parquet(index_path), np.load(array_path)
    # The array is saved before the index, so a crash between the two leaves extra array rows.
    if len(array) < len(index):
        raise RuntimeError(f"{array_path} has fewer rows than {index_path}; rerun with --force")
    return index, array[: len(index)]


def save_cache(out_dir: Path, name: str, index: pd.DataFrame, array: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    array_path, index_path = out_dir / f"{name}_embeddings.npy", out_dir / f"{name}_index.parquet"
    with open(f"{array_path}.tmp", "wb") as file:
        np.save(file, array)
    os.replace(f"{array_path}.tmp", array_path)
    index.to_parquet(f"{index_path}.tmp", index=False)
    os.replace(f"{index_path}.tmp", index_path)


def check_vectors(vectors: np.ndarray) -> None:
    if vectors.ndim != 2 or vectors.shape[1] != DIMENSION:
        raise RuntimeError(f"Expected {DIMENSION}-dimensional vectors, got shape {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise RuntimeError("Embeddings contain NaN or inf")
    norms = np.linalg.norm(vectors, axis=1)
    if np.abs(norms - 1).max() > 1e-2:
        raise RuntimeError(f"Embeddings are not unit-normalized (norms {norms.min():.4f}..{norms.max():.4f})")


def normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def load_model(config: dict):
    from sentence_transformers import SentenceTransformer

    model_dir = Path(config["model_dir"])
    if not model_dir.exists():
        raise RuntimeError(f"{model_dir} not found; run download_model.py first")
    device = resolve_device(config["device"])
    # BF16 matmuls are slow on most CPUs, so the CPU path uses float32.
    dtype = torch.float32 if device == "cpu" else getattr(torch, config["dtype"])
    model = SentenceTransformer(
        str(model_dir),
        device=device,
        model_kwargs={"dtype": dtype, "attn_implementation": "sdpa"},
        local_files_only=True,
    )
    model.max_seq_length = config["max_seq_length"]
    model.eval()
    print(f"[model] {model_dir} on {device} ({dtype})")
    return model, config["batch_size"][device]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--force", action="store_true", help="Re-embed everything, ignoring the cache")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    data_dir, out_dir = Path(config["standardized_dir"]), Path(config["embeddings_dir"])
    states = pd.concat([pd.read_parquet(data_dir / f"{split}.parquet") for split in SPLITS]).text
    # Both candidate-text variants are embedded so either can be selected in config.yaml.
    candidates = {
        text
        for mode in ("name", "name_description")
        for text in candidate_texts({**config, "candidate_text": mode}).values()
    }
    jobs = {
        "state": ("encode_query", sorted(set(states))),
        "candidate": ("encode_document", sorted(candidates)),
    }

    model = batch_size = None
    for name, (method, texts) in jobs.items():
        index, array = empty_cache() if args.force else load_cache(out_dir, name)
        done = set(index.text_hash)
        todo = [text for text in texts if text_hash(text) not in done]
        if not todo:
            print(f"[skip] {name}: all {len(texts)} texts already embedded")
            continue
        if model is None:
            model, batch_size = load_model(config)

        encode = getattr(model, method)
        print(f"[embed] {name}: {len(todo)} new of {len(texts)} texts with {method}")
        for start in range(0, len(todo), CHUNK_SIZE):
            chunk = todo[start : start + CHUNK_SIZE]
            vectors = encode(chunk, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
            vectors = vectors.astype(np.float32)
            check_vectors(vectors)
            vectors = normalize(vectors)  # BF16 output is only unit-length to ~1e-3
            rows = pd.DataFrame(
                {
                    "row": range(len(index), len(index) + len(chunk)),
                    "text_hash": [text_hash(text) for text in chunk],
                    "text": chunk,
                }
            )
            index = rows if index.empty else pd.concat([index, rows], ignore_index=True)
            array = np.concatenate([array, vectors])
            save_cache(out_dir, name, index, array)

        # The same batch must encode identically. The stored vectors were encoded in different
        # batches, so BF16 rounding allows small drift against them (min cosine ~0.995 on MPS).
        sample = todo[:8]
        first, second = (encode(sample, batch_size=batch_size, convert_to_numpy=True) for _ in range(2))
        if not np.array_equal(first, second):
            raise RuntimeError("Encoding the same batch twice gave different vectors")
        stored = array[index.set_index("text").loc[sample, "row"].to_numpy()]
        similarity = (normalize(first.astype(np.float32)) * stored).sum(axis=1).min()
        if similarity < 0.99:
            raise RuntimeError(f"Re-encoded vectors drifted from the stored ones (min cosine {similarity:.5f})")
        print(f"[saved] {name}: {len(index)} vectors in {out_dir} (reproducibility cosine >= {similarity:.5f})")


if __name__ == "__main__":
    main()
