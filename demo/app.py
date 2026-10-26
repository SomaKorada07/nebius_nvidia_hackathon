#!/usr/bin/env python3
"""Reflex demo: fast local option selection on Nemotron embeddings, with a Nemotron LLM fallback.

Usage (from the repository root):
    .venv/bin/python demo/app.py            # then open http://localhost:8000

Every question is routed on its own (no chat history). Each request embeds the question and all of
the domain's options afresh; nothing is cached. The embedding model loads at startup, and all model
work runs on one dedicated worker thread, so the web server never blocks while embedding.

Routing: if Reflex picks an option with probability >= the threshold, the answer is final. If it
answers NONE_OF_THESE or is less sure than the threshold, the question goes to a Nemotron LLM on
Nebius Token Factory, which either picks an option or says none fits and suggests the missing tool.

Environment, read from the shell or from .env in the repository root (see .env.example):
NEBIUS_API_KEY (required for the LLM), REFLEX_LLM_MODEL (default nvidia/Nemotron-3_5-Lightning),
NEBIUS_BASE_URL (default https://api.tokenfactory.nebius.com/v1), PORT (default 8000).
"""

import asyncio
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import httpx
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from predict import NONE_OF_THESE, ReflexPredictor, candidate_text  # noqa: E402



def load_env_file(path: Path) -> None:
    """Read KEY=VALUE lines into os.environ; variables already set in the shell take precedence."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env_file(HERE.parent / ".env")


def training_status() -> dict[str, str]:
    """Label name -> "trained", or "held_out" for the 45 labels kept out of v3 training entirely."""
    data = HERE.parent / "data"
    split = json.loads((data / "processed" / "v3_split.json").read_text())
    trained = set(split["base_labels"]) | set(split["pseudo_unseen_labels"])
    status: dict[str, str] = {}
    for label_id in json.loads((data / "standardized" / "labels.json").read_text()):
        name = label_id.split(":", 1)[1]  # e.g. "banking77:card_arrival"; a name can occur in both datasets
        if status.get(name) != "trained":
            status[name] = "trained" if label_id in trained else "held_out"
    return status


DOMAINS = json.loads((HERE / "domains.json").read_text())
LABEL_STATUS = training_status()
for _domain in DOMAINS.values():  # options that are not dataset labels at all are "new"
    _domain["option_training"] = {name: LABEL_STATUS.get(name, "new") for name in _domain["options"]}
    _domain["trained_domain"] = "trained" in _domain["option_training"].values()
NEBIUS_BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1").rstrip("/")
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "").strip()
LLM_MODEL = os.environ.get("REFLEX_LLM_MODEL", "nvidia/Nemotron-3_5-Lightning")
PREVIEW_DIMS = 128  # embedding dimensions sent to the UI for the vector strip

LLM_SYSTEM = """You are the fallback router of an assistant. A fast local classifier could not confidently \
match the user's request to one of the available tools, so you decide.
Rules:
- Pick the tool whose purpose is what the user wants done, even when the words differ \
(for example "shoot Bob a quick note" fits send_message).
- Answer none only when no tool can do it. A tool that does a different action, or acts on a different \
kind of thing, does not count (for example book_flight cannot book a hotel, and delete_email cannot \
unsubscribe from a newsletter).
- When you answer none, suggest the single missing tool. It must not be one of the available tools: \
give a new snake_case name and a one-sentence description that starts with a verb.
Reply with one JSON object and nothing else, either
{"decision": "tool", "tool": "<one of the tool names>", "reason": "<one short sentence>"}
or
{"decision": "none", "suggested_tool": {"name": "<snake_case>", "description": "<one sentence>"}, \
"reason": "<one short sentence>"}"""

# All model work (loading, embedding, scoring) runs on this one thread, never on the event loop.
MODEL_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reflex-model")
status = {"model": "loading", "device": None, "load_seconds": None, "error": None,
          "llm": "checking", "llm_model": LLM_MODEL}
predictor: ReflexPredictor | None = None
model_ready: asyncio.Future | None = None


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


# ---- model thread ----------------------------------------------------------------------------

def load_model() -> None:
    global predictor
    start = time.perf_counter()
    try:
        loaded = ReflexPredictor()
        warm = loaded.encode("encode_query", ["warm up"])  # loads the encoder and compiles the GPU kernels
        loaded.encode("encode_document", ["warm_up: Warm up the encoder."])
        predictor = loaded
        for size in sorted({len(domain["options"]) for domain in DOMAINS.values()}):  # MPS compiles per list size
            score(warm, np.repeat(warm, size, axis=0))
    except Exception as error:  # surfaced in the UI status
        status.update(model="error", error=str(error))
        raise
    status.update(model="ready", device=loaded.device, load_seconds=round(time.perf_counter() - start, 1))


def embed(method: str, texts: list[str]) -> tuple[np.ndarray, float, str]:
    start = time.perf_counter()
    vectors = predictor.encode(method, texts)
    return vectors, elapsed_ms(start), threading.current_thread().name


@torch.no_grad()
def score(question: np.ndarray, options: np.ndarray) -> dict:
    start = time.perf_counter()
    head, device = predictor.head, predictor.device
    state = torch.from_numpy(question).to(device)  # (1, D)
    candidates = torch.from_numpy(options).unsqueeze(0).to(device)  # (1, K, D)
    mask = torch.ones(1, len(options), dtype=torch.bool, device=device)
    logits = head(state, candidates, mask)
    features = None
    if hasattr(head, "none_features"):  # the v3 head decides NONE from these six numbers
        scores = head.candidate_logits(state, candidates, mask)
        features = head.none_features(scores, state, candidates, mask)[0].cpu().tolist()
    logits = logits[0].cpu()
    probs = torch.softmax(logits / predictor.temperature, dim=0)
    return {"logits": logits.tolist(), "probs": probs.tolist(), "features": features,
            "ms": elapsed_ms(start), "thread": threading.current_thread().name}


async def on_model_thread(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(MODEL_THREAD, fn, *args)


# ---- LLM (Nebius Token Factory) --------------------------------------------------------------

class LLMUnavailable(Exception):
    pass


def token_factory() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=NEBIUS_BASE_URL, timeout=60,
                             headers={"Authorization": f"Bearer {NEBIUS_API_KEY}"})


async def check_llm() -> None:
    """Check that the API key works and that Token Factory serves the model."""
    if not NEBIUS_API_KEY:
        status["llm"] = "no API key"
        return
    try:
        async with token_factory() as client:
            response = await client.get("/models")
    except httpx.HTTPError:
        status["llm"] = "unreachable"
        return
    if response.status_code in (401, 403):
        status["llm"] = "API key rejected"
    elif response.is_error:
        status["llm"] = f"error {response.status_code}"
    else:
        served = {model["id"] for model in response.json().get("data", [])}
        status["llm"] = "ready" if LLM_MODEL in served else "model not found"


def parse_reply(reply: str, options: dict[str, str]) -> dict:
    try:
        parsed = json.loads(reply[reply.index("{"): reply.rindex("}") + 1])
    except ValueError:
        return {"decision": "invalid", "reason": "the reply was not valid JSON"}
    suggested = parsed.get("suggested_tool") or {}
    if parsed.get("decision") == "none" and suggested.get("name") in options:  # "missing" tool is offered
        parsed = {"decision": "tool", "tool": suggested["name"], "reason": parsed.get("reason")}
    if parsed.get("decision") == "tool" and parsed.get("tool") not in options:
        return {"decision": "invalid", "reason": f"picked {parsed.get('tool')!r}, which is not one of the options"}
    if parsed.get("decision") not in ("tool", "none"):
        return {"decision": "invalid", "reason": "unknown decision"}
    return parsed


async def ask_llm(question: str, options: dict[str, str]) -> dict:
    tools = "\n".join(f"- {name}: {description}" for name, description in options.items())
    body = {"model": LLM_MODEL, "temperature": 0, "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},  # otherwise Nemotron reasons first: 3-7 s
            "messages": [{"role": "system", "content": LLM_SYSTEM},
                         {"role": "user", "content": f"Request: {question}\nAvailable tools:\n{tools}"}]}
    if not NEBIUS_API_KEY:
        raise LLMUnavailable("NEBIUS_API_KEY is not set (add it to .env and restart)")
    start = time.perf_counter()
    try:
        async with token_factory() as client:
            response = await client.post("/chat/completions", json=body)
    except httpx.HTTPError as error:
        raise LLMUnavailable(f"Token Factory request failed: {error}") from error
    if response.is_error:
        raise LLMUnavailable(f"Token Factory returned {response.status_code}: {response.text[:300]}")
    data = response.json()
    reply = (data["choices"][0]["message"].get("content") or "").strip()
    usage = data.get("usage") or {}
    return {"ms": elapsed_ms(start), "reply": reply, "parsed": parse_reply(reply, options),
            "tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens")}


# ---- request pipeline ------------------------------------------------------------------------

def event(step: str, **data) -> str:
    return json.dumps({"step": step, **data}) + "\n"


async def pipeline(domain: str, question: str, threshold: float):
    options = DOMAINS[domain]["options"]
    names = list(options)
    texts = [candidate_text(name, description) for name, description in options.items()]
    start = time.perf_counter()
    yield event("start", question=question, domain=domain, options=names, threshold=threshold)
    try:
        if status["model"] != "ready":
            yield event("wait_model", status="running")
            await model_ready
            yield event("wait_model", status="done")

        yield event("embed_question", status="running", text=question)
        question_vec, ms, thread = await on_model_thread(embed, "encode_query", [question])
        yield event("embed_question", status="done", ms=ms, thread=thread, dim=int(question_vec.shape[1]),
                    preview=question_vec[0, :PREVIEW_DIMS].round(4).tolist())

        yield event("embed_options", status="running", count=len(texts))
        option_vecs, ms, thread = await on_model_thread(embed, "encode_document", texts)
        cosines = (option_vecs @ question_vec[0]).tolist()
        yield event("embed_options", status="done", ms=ms, thread=thread, items=[
            {"name": n, "text": t, "cosine": round(c, 4), "preview": v[:PREVIEW_DIMS].round(4).tolist()}
            for n, t, c, v in zip(names, texts, cosines, option_vecs)])

        yield event("score", status="running")
        result = await on_model_thread(score, question_vec, option_vecs)
        probs, logits = result["probs"], result["logits"]
        best = int(np.argmax(probs))
        answer = names[best] if best < len(names) else NONE_OF_THESE
        ranking = sorted(({"name": n, "logit": logits[i], "prob": probs[i]} for i, n in enumerate(names)),
                         key=lambda row: -row["prob"])
        ranking.append({"name": NONE_OF_THESE, "logit": logits[-1], "prob": probs[-1]})
        yield event("score", status="done", ms=result["ms"], thread=result["thread"], ranking=ranking,
                    features=result["features"], answer=answer, probability=probs[best])

        if answer != NONE_OF_THESE and probs[best] >= threshold:
            yield event("route", status="done", target="reflex",
                        reason=f"{answer} with p = {probs[best]:.2f} ≥ threshold {threshold:.2f}")
            yield event("final", answer=answer, source="reflex", total_ms=elapsed_ms(start))
            return

        reason = (f"Reflex answered NONE_OF_THESE (p = {probs[-1]:.2f})" if answer == NONE_OF_THESE else
                  f"{answer} only has p = {probs[best]:.2f} < threshold {threshold:.2f}")
        yield event("route", status="done", target="llm", reason=reason)
        yield event("llm", status="running", model=LLM_MODEL)
        try:
            llm = await ask_llm(question, options)
        except LLMUnavailable as error:
            yield event("llm", status="error", message=str(error))
            yield event("final", answer=answer, source="reflex (LLM unavailable)", total_ms=elapsed_ms(start))
            return
        yield event("llm", status="done", model=LLM_MODEL, **llm)
        parsed = llm["parsed"]
        if parsed["decision"] == "tool":
            yield event("final", answer=parsed["tool"], source="llm", reason=parsed.get("reason"),
                        total_ms=elapsed_ms(start))
        elif parsed["decision"] == "none":
            yield event("final", answer=NONE_OF_THESE, source="llm", reason=parsed.get("reason"),
                        suggested_tool=parsed.get("suggested_tool"), total_ms=elapsed_ms(start))
        else:
            yield event("final", answer=answer, source="reflex (LLM reply unusable)", reason=parsed["reason"],
                        total_ms=elapsed_ms(start))
    except Exception as error:
        yield event("error", message=f"{type(error).__name__}: {error}")


# ---- web app ---------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model_ready
    model_ready = asyncio.get_running_loop().run_in_executor(MODEL_THREAD, load_model)
    llm_check = asyncio.create_task(check_llm())
    yield
    llm_check.cancel()
    MODEL_THREAD.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Reflex demo", lifespan=lifespan)


class Ask(BaseModel):
    domain: str
    question: str = Field(min_length=1, max_length=500)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/status")
def get_status() -> dict:
    return status


@app.get("/api/domains")
def get_domains() -> dict:
    return DOMAINS


@app.post("/api/ask")
async def ask(request: Ask) -> StreamingResponse:
    if request.domain not in DOMAINS:
        raise HTTPException(404, f"unknown domain {request.domain!r}")
    return StreamingResponse(pipeline(request.domain, request.question.strip(), request.threshold),
                             media_type="application/x-ndjson")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
