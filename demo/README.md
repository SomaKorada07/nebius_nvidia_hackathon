# Reflex demo

A chat page that routes each question to one of a domain's tools. Reflex (the v3 head on Nemotron Embed 1B) answers on its own when it is confident. When it answers `NONE_OF_THESE`, or when its top probability is below the threshold slider, the question goes to a Nemotron LLM served by Ollama. The LLM either picks a tool or names the missing one.

## Run

```bash
ollama serve                                         # if it is not already running
ollama pull nemotron-3.5-lightning:30b-a3b-mlx       # once, 19 GB
.venv/bin/python demo/app.py                         # from the repository root
```

Open http://localhost:8000. The embedding model loads in about 4 s at startup, and the header shows when both models are ready. The page works without Ollama: questions that would escalate keep Reflex's answer and are marked "LLM unavailable".

| Variable | Default |
|---|---|
| `PORT` | `8000` |
| `OLLAMA_URL` | `http://localhost:11434` |
| `REFLEX_LLM_MODEL` | `nemotron-3.5-lightning:30b-a3b-mlx` |

## What happens on each question

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
