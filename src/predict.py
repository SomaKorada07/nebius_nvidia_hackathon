#!/usr/bin/env python3
"""Score a question against candidate options with a trained Reflex head (default: the frozen v3 checkpoint).

Usage:
    python src/predict.py --question "my card hasn't arrived yet" \\
        --candidate card_arrival "Ask when a card that was ordered will arrive." \\
        --candidate lost_or_stolen_card "Report a card as lost or stolen."
    python src/predict.py --input requests.jsonl --output scored.jsonl

The predictor is open: options are any {name: description} pairs, not limited to the training
labels. Each is embedded as "name: description", the format the head was trained on. Input rows
are {"question": ..., "candidates": {name: description, ...}}.

Each result holds the answer (a candidate name, or NONE_OF_THESE), its probability, the
probability of NONE_OF_THESE and every candidate's probability, all temperature-calibrated.
--none-shift moves the operating point: a positive shift answers NONE_OF_THESE more often
(see the rejection trade-off in checkpoints/v3/REPORT.md, section 4.8).

Heads were trained on lists of 5 and 10 candidates; lists of 2 or more are accepted.
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from device import resolve_device
from embed import load_model, normalize
from models import load_head

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "v3"
NONE_OF_THESE = "NONE_OF_THESE"
HEAD_BATCH_SIZE = 1024


def candidate_text(name: str, description: str) -> str:
    """The text embedded for an option; matches data.candidate_texts for name_description."""
    return f"{name}: {description}"


def dataset_candidates(label_ids: list[str], config: Path = CHECKPOINT / "config.yaml") -> dict[str, str]:
    """{name: description} for training-data label IDs such as "banking77:card_arrival"."""
    config = yaml.safe_load(Path(config).read_text())
    labels = json.loads((ROOT / config["standardized_dir"] / "labels.json").read_text())
    descriptions = json.loads((ROOT / config["label_descriptions"]).read_text())
    options = {labels[i]["label"]: descriptions[i] for i in label_ids}
    if len(options) != len(label_ids):
        raise ValueError(f"label names collide (e.g. exchange_rate exists in both datasets): {label_ids}")
    return options


class ReflexPredictor:
    """Embeds questions and options with the frozen Nemotron encoder and scores them with a Reflex head.

    `encoder` may be any object with encode_query / encode_document (sentence-transformers API);
    by default the local Nemotron model from the config is loaded on first use.
    """

    def __init__(self, weights: Path = CHECKPOINT / "weights" / "reflex_v3.pt",
                 calibration: Path = CHECKPOINT / "results" / "calibration.json", calibration_key: str = "reflex_v3",
                 config: Path = CHECKPOINT / "config.yaml", device: str | None = None, encoder=None):
        config = yaml.safe_load(Path(config).read_text())
        config["model_dir"] = str(ROOT / config["model_dir"])  # relative to the repository root
        self.config = config
        self.device = resolve_device(device or config["device"])
        self.head = load_head(Path(weights), self.device)
        self.temperature = json.loads(Path(calibration).read_text())[calibration_key]["temperature"]
        self._encoder, self._batch_size = encoder, config["batch_size"][self.device]

    def encode(self, method: str, texts: list[str]) -> np.ndarray:
        """Unit-length float32 vectors; method is "encode_query" (questions) or "encode_document" (options)."""
        if self._encoder is None:
            with contextlib.redirect_stdout(sys.stderr):  # keep stdout clean for JSONL output
                self._encoder, self._batch_size = load_model(self.config)
        vectors = getattr(self._encoder, method)(texts, batch_size=self._batch_size, convert_to_numpy=True)
        return normalize(np.asarray(vectors, dtype=np.float32))  # BF16 output is only unit-length to ~1e-3

    @torch.no_grad()
    def predict(self, questions: list[str], candidates: list[dict[str, str]], none_shift: float = 0.0) -> list[dict]:
        """Score each question against its own {name: description} options."""
        if len(questions) != len(candidates):
            raise ValueError(f"{len(questions)} questions but {len(candidates)} candidate sets")
        for options in candidates:
            if len(options) < 2:
                raise ValueError(f"each question needs at least 2 candidates, got {list(options)}")
            if NONE_OF_THESE in options:
                raise ValueError(f"{NONE_OF_THESE} is added by the model; do not pass it as a candidate")
            if not all(name.strip() and description.strip() for name, description in options.items()):
                raise ValueError(f"every candidate needs a name and a description: {options}")
        if not questions:
            return []

        texts = [[candidate_text(n, d) for n, d in options.items()] for options in candidates]
        unique = sorted({t for row in texts for t in row})
        row_of = {t: i for i, t in enumerate(unique)}
        states = torch.from_numpy(self.encode("encode_query", list(questions))).to(self.device)
        bank = torch.from_numpy(self.encode("encode_document", unique)).to(self.device)

        kmax = max(len(row) for row in texts)
        cand_idx = torch.zeros(len(questions), kmax, dtype=torch.long)
        mask = torch.zeros(len(questions), kmax, dtype=torch.bool)
        for i, row in enumerate(texts):
            cand_idx[i, : len(row)] = torch.tensor([row_of[t] for t in row])
            mask[i, : len(row)] = True
        cand_idx, mask = cand_idx.to(self.device), mask.to(self.device)

        self.head.eval()
        logits = torch.cat([self.head(states[index], bank[cand_idx[index]], mask[index]).cpu()
                            for index in torch.arange(len(questions), device=self.device).split(HEAD_BATCH_SIZE)])
        logits[:, -1] += none_shift
        probs = torch.softmax(logits / self.temperature, dim=1).numpy()

        results = []
        for question, options, p, z in zip(questions, candidates, probs, logits.numpy()):
            names = list(options)
            scores = np.r_[p[: len(names)], p[-1]]
            best = int(np.argmax(scores))
            results.append({
                "question": question,
                "answer": names[best] if best < len(names) else NONE_OF_THESE,
                "probability": float(scores[best]),
                "none_probability": float(p[-1]),
                "none_logit": float(z[-1]),
                "candidates": {name: float(p[j]) for j, name in enumerate(names)},
                "logits": {name: float(z[j]) for j, name in enumerate(names)},
            })
        return results

    def predict_one(self, question: str, candidates: dict[str, str], none_shift: float = 0.0) -> dict:
        return self.predict([question], [candidates], none_shift)[0]


def format_result(result: dict, number: int) -> str:
    """A readable block per question: options ranked by probability, logits and probabilities to 2 decimals."""
    rows = sorted(((name, result["logits"][name], prob) for name, prob in result["candidates"].items()),
                  key=lambda row: -row[2])
    rows.append((NONE_OF_THESE, result["none_logit"], result["none_probability"]))
    width = max(len(name) for name, _, _ in rows)
    lines = [f"[{number}] {result['question']}", f"    {'option':<{width}}  {'logit':>7}  {'prob':>5}"]
    for name, logit, prob in rows:
        marker = "  <- answer" if name == result["answer"] else ""
        lines.append(f"    {name:<{width}}  {logit:>7.2f}  {prob:>5.2f}{marker}")
    lines.append(f"    answer: {result['answer']} (p = {result['probability']:.2f})")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", help="One question to score")
    parser.add_argument("--candidate", nargs=2, action="append", metavar=("NAME", "DESCRIPTION"),
                        help="An option for --question; repeat for each option")
    parser.add_argument("--input", type=Path, help='JSONL with {"question": ..., "candidates": {name: description}}')
    parser.add_argument("--output", type=Path, help="JSONL output for --input (default: stdout)")
    parser.add_argument("--weights", type=Path, default=CHECKPOINT / "weights" / "reflex_v3.pt")
    parser.add_argument("--calibration", type=Path, default=CHECKPOINT / "results" / "calibration.json")
    parser.add_argument("--calibration-key", default="reflex_v3", help="Entry in the calibration file")
    parser.add_argument("--config", type=Path, default=CHECKPOINT / "config.yaml")
    parser.add_argument("--device", help="cuda, mps or cpu (default: from the config)")
    parser.add_argument("--none-shift", type=float, default=0.0, help="Added to the NONE logit; > 0 rejects more")
    parser.add_argument("--pretty", action="store_true", help="Readable table per question instead of JSONL")
    args = parser.parse_args()
    if (args.question is None) == (args.input is None) or (args.question is not None and not args.candidate):
        parser.error("give either --question with --candidate NAME DESCRIPTION (repeated), or --input")

    if args.question is not None:
        if len({name for name, _ in args.candidate}) != len(args.candidate):
            parser.error("candidate names must be unique")
        questions, candidates = [args.question], [dict(args.candidate)]
    else:
        rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
        questions, candidates = [r["question"] for r in rows], [r["candidates"] for r in rows]

    predictor = ReflexPredictor(args.weights, args.calibration, args.calibration_key, args.config, args.device)
    results = predictor.predict(questions, candidates, args.none_shift)
    if args.pretty:
        lines = "\n".join(format_result(r, i) for i, r in enumerate(results, 1))
    else:
        lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    if args.output:
        args.output.write_text(lines)
        print(f"[saved] {len(results)} results to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(lines)


if __name__ == "__main__":
    main()
