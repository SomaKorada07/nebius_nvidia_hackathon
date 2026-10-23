#!/usr/bin/env python3
"""Reflex demo: fast local option selection on Nemotron embeddings, with a Nemotron LLM fallback.

Usage (from the repository root):
    .venv/bin/python demo/app.py            # then open http://localhost:8000

Every question is routed on its own (no chat history). Each request embeds the question and all of
the domain's options afresh; nothing is cached. The embedding model loads at startup, and all model
work runs on one dedicated worker thread, so the web server never blocks while embedding.

Routing: if Reflex picks an option with probability >= the threshold, the answer is final. If it
answers NONE_OF_THESE or is less sure than the threshold, the question goes to a Nemotron LLM served
by Ollama, which either picks an option or says none fits and suggests the missing tool.

Environment: OLLAMA_URL (default http://localhost:11434), REFLEX_LLM_MODEL
(default nemotron-3.5-lightning:30b-a3b-mlx), PORT (default 8000).
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

DOMAINS = json.loads((HERE / "domains.json").read_text())
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("REFLEX_LLM_MODEL", "nemotron-3.5-lightning:30b-a3b-mlx")
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


# ---- LLM (Ollama) ----------------------------------------------------------------------------

async def check_llm() -> None:
    """Find the model in Ollama and load it, so the first escalation is fast."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            names = [m["name"] for m in (await client.get(f"{OLLAMA_URL}/api/tags")).json().get("models", [])]
        if LLM_MODEL not in names:
            status["llm"] = "model missing"
            return
        status["llm"] = "loading"
        async with httpx.AsyncClient(timeout=300) as client:
            await client.post(f"{OLLAMA_URL}/api/generate", json={"model": LLM_MODEL, "prompt": "", "keep_alive": "30m"})
        status["llm"] = "ready"
    except httpx.HTTPError:
        status["llm"] = "not running"


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
    body = {"model": LLM_MODEL, "stream": False, "think": False, "keep_alive": "30m",
            "options": {"temperature": 0, "num_ctx": 4096},
            "messages": [{"role": "system", "content": LLM_SYSTEM},
                         {"role": "user", "content": f"Request: {question}\nAvailable tools:\n{tools}"}]}
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=120) as client:
        response = (await client.post(f"{OLLAMA_URL}/api/chat", json=body)).raise_for_status().json()
    reply = response["message"]["content"].strip()
    return {"ms": elapsed_ms(start), "reply": reply, "parsed": parse_reply(reply, options),
            "tokens": response.get("eval_count"), "prompt_tokens": response.get("prompt_eval_count")}


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
        except httpx.HTTPError as error:
            yield event("llm", status="error", message=f"Ollama request failed: {error}")
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
