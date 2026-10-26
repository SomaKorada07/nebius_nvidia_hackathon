# Reflex demo

A chat page that routes each question to one of a domain's tools. Reflex (the v3 head on Nemotron Embed 1B) answers on its own when it is confident. When it answers `NONE_OF_THESE`, or when its top probability is below the threshold slider, the question goes to a Nemotron LLM (`nvidia/Nemotron-3_5-Lightning`) on Nebius Token Factory. The LLM either picks a tool or names the missing one.

## Run

```bash
cp .env.example .env                 # then put your Token Factory key in NEBIUS_API_KEY
.venv/bin/python demo/app.py         # from the repository root
```

Open http://localhost:8000. The embedding model loads in about 4 s at startup. The header shows when the embedder is ready, and whether the API key works and Token Factory serves the model. Without a key the page still works: questions that would escalate keep Reflex's answer and are marked "LLM unavailable".

Settings are read from the shell or from `.env` in the repository root. Shell variables take precedence.

| Variable | Default |
|---|---|
| `PORT` | `8000` |
| `NEBIUS_API_KEY` | none; required for the LLM fallback |
| `REFLEX_LLM_MODEL` | `nvidia/Nemotron-3_5-Lightning` |
| `NEBIUS_BASE_URL` | `https://api.tokenfactory.nebius.com/v1` |

## What happens on each question

- **Training coverage.** Each domain is tagged *trained* or *unseen*. The server works this out at startup from the v3 label split, `data/processed/v3_split.json`.
  - *Unseen:* Customer service, whose 10 tools were written for the demo.
  - *Trained:* Banking, Smart assistant and Travel & dining. Within these domains, an option Reflex never saw is marked *held out* (one of the 45 labels kept out of training) or *new* (not a dataset label).
- **No history.** Every question is routed on its own.
- **No cache.** The question and every option of the domain are embedded again on every request.
- **Model thread.** Loading, embedding and scoring all run on one worker thread (`reflex-model_0`), never on the web server's event loop. The page stays responsive while the model loads or embeds.
- **Pipeline view.** Each step streams to the page as it finishes:
  1. the question embedding;
  2. the option embeddings, with their cosine similarities;
  3. the Reflex probabilities, plus the six numbers the NONE score is computed from;
  4. the routing decision;
  5. the LLM call, when there is one.

  Clicking an earlier answer shows its pipeline again.

## Files

| File | Contents |
|---|---|
| `domains.json` | The domains, their fixed options (`name: description`) and example questions. Edit it to add a domain; restart the server. |
| `app.py` | FastAPI server; `POST /api/ask` streams the pipeline steps as NDJSON. |
| `static/index.html` | The page, self-contained, works offline. |
