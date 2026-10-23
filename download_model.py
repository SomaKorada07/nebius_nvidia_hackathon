#!/usr/bin/env python3
"""Download the Nemotron embedding model into models/ (gitignored).

Usage:
    python download_model.py [--config config.yaml]

The model is pinned to the revision in config.yaml. After downloading, every file is
checked against the size and SHA-256 published on Hugging Face, then a marker file is
written; later runs see the marker and skip without touching the network, so this can
be run once on a login node before embedding on an offline GPU node.
"""

import argparse
import hashlib
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

MARKER = ".verified_revision"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    model_id, revision = config["model_id"], config["model_revision"]
    model_dir = Path(config["model_dir"])
    marker = model_dir / MARKER
    if marker.exists() and marker.read_text().strip() == revision:
        print(f"[skip] {model_dir} already holds verified revision {revision}")
        return

    print(f"[download] {model_id}@{revision} -> {model_dir}")
    snapshot_download(model_id, revision=revision, local_dir=model_dir)

    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    for file in info.siblings:
        path = model_dir / file.rfilename
        if not path.is_file() or path.stat().st_size != file.size:
            raise RuntimeError(f"{path} is missing or the wrong size; rerun to resume the download")
        if file.lfs and sha256(path) != file.lfs.sha256:
            raise RuntimeError(f"Checksum mismatch for {path}; delete it and rerun")

    marker.write_text(revision + "\n")
    print(f"[saved] {model_dir} (verified {len(info.siblings)} files)")


if __name__ == "__main__":
    main()
