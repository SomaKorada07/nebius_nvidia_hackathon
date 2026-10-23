"""Pick the torch device so the same code runs on a Mac (MPS) and on NVIDIA GPUs (CUDA)."""

import torch


def resolve_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
