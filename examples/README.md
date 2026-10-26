# Predicting with Reflex v3

`src/predict.py` scores a question against a set of options and returns the best option, or
`NONE_OF_THESE` when no option fits. The options are any `name: description` pairs; they do not
have to be labels the model was trained on.

This folder holds two example files:

| File | Domain | Cases |
|---|---|---|
| `predict_requests.jsonl` | Banking, assistant and restaurant requests | 7 |
| `customer_service_tools.jsonl` | Choosing a customer-service tool (transaction status, refunds, invoices, …) | 11 |

## Before you start

Run everything from the repository root. You need:

1. **The Python environment:**
   ```bash
   python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```
2. **The Nemotron embedding model** in `models/` (2.9 GB, downloaded once):
   ```bash
   .venv/bin/python download_model.py
   ```
3. **The trained v3 weights** are included in git: `checkpoints/v3/weights/reflex_v3.pt` (9 MB).
   To retrain them, follow `checkpoints/v3/REPORT.md` §8.

The predictor uses the GPU automatically if there is one: CUDA, then Apple MPS, then CPU.

## Run the examples

Readable output:

```bash
PYTHONPATH=src .venv/bin/python src/predict.py --input examples/customer_service_tools.jsonl --pretty 2>/dev/null
PYTHONPATH=src .venv/bin/python src/predict.py --input examples/predict_requests.jsonl --pretty 2>/dev/null
```

`2>/dev/null` hides the model's loading messages. Loading takes a few seconds; scoring is fast.

For each question, every option is listed with its logit and probability, highest probability
first, then `NONE_OF_THESE` and the answer:

```text
[1] Did my payment for order 4521 go through?
    option                     logit   prob
    transaction_status          0.85   0.70  <- answer
    update_payment_method      -8.51   0.00
    ...
    talk_to_agent             -16.69   0.00
    NONE_OF_THESE               0.15   0.30
    answer: transaction_status (p = 0.70)
```

- **logit:** the model's raw score.
- **prob:** the calibrated probability. All probabilities for one question, including
  `NONE_OF_THESE`, add up to 1.

To get machine-readable output instead, leave out `--pretty`. You get one JSON line per question,
printed or written to a file with `--output`:

```bash
PYTHONPATH=src .venv/bin/python src/predict.py --input examples/customer_service_tools.jsonl --output scored.jsonl
```

```json
{"question": "...", "answer": "transaction_status", "probability": 0.70, "none_probability": 0.30,
 "none_logit": 0.15, "candidates": {"transaction_status": 0.70, ...}, "logits": {"transaction_status": 0.85, ...}}
```

## Write your own requests

One JSON object per line: the question, and the available options as `name: description` pairs.

```json
{"question": "Where is my package?", "candidates": {"order_tracking": "Track where a shipped order is right now.", "cancel_order": "Cancel an order that has not shipped yet.", "refund_status": "Check the status of a refund that was already requested."}}
```

Rules:
- **At least 2 options** per question, each with a non-empty name and description. Names must be
  unique, and `NONE_OF_THESE` is reserved.
- **Option lists can differ in length** between lines. The model was trained on lists of 5 and 10
  options, and works best near those sizes.
- **Order doesn't matter:** shuffling the options gives the same answer and probabilities.
- **Descriptions matter.** Each option is embedded as `"name: description"`. A one-line
  description saying what the user wants ("Ask when a card that was ordered will arrive.")
  works best, and options whose descriptions sound alike are easy to confuse.

To score a single question without a file:

```bash
PYTHONPATH=src .venv/bin/python src/predict.py --pretty 2>/dev/null \
  --question "what's the most I can send in one transfer?" \
  --candidate transfer_limit "Ask what the maximum amount for a single transfer is." \
  --candidate transfer_fee "Ask why a fee was charged for a transfer." \
  --candidate cancel_transfer "Cancel a transfer that was just made."
```

## Options

| Option | Meaning |
|---|---|
| `--pretty` | Readable table per question instead of JSON lines |
| `--output FILE` | Write the results to a file instead of the screen |
| `--none-shift X` | Added to the `NONE_OF_THESE` logit. Positive values answer `NONE_OF_THESE` more often, negative values less often. Default 0 |
| `--device cuda\|mps\|cpu` | Force a device |
| `--weights`, `--calibration`, `--config` | Use a different checkpoint (default: `checkpoints/v3`) |

## From Python

```python
import sys; sys.path.insert(0, "src")
from predict import ReflexPredictor

predictor = ReflexPredictor()  # loads the v3 checkpoint; the encoder loads on first use
result = predictor.predict_one("Did my payment for order 4521 go through?", {
    "transaction_status": "Check the status of a payment or transaction.",
    "refund_status": "Check the status of a refund that was already requested.",
    "order_tracking": "Track where a shipped order is right now.",
    "generate_invoice": "Create an invoice or receipt for an order.",
    "talk_to_agent": "Connect the customer with a human support agent.",
})
print(result["answer"], round(result["probability"], 2))  # transaction_status 0.58

results = predictor.predict(questions, candidate_dicts)  # many questions at once
```

## What to expect

- **Strong:** rejecting option lists that don't contain the answer, and picking among options
  that are clearly different.
- **Weaker:** options that sound like a training label but mean something else, and close
  calls between the right option and `NONE_OF_THESE`. On new labels the model leans towards
  answering `NONE_OF_THESE` when unsure.

Results can change in the second or third decimal between runs, because the GPU encoder runs in
reduced precision. Validation results and known limitations are in `checkpoints/v3/REPORT.md`.
