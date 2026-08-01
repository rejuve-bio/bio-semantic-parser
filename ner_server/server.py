"""
ner_server/server.py
────────────────────
Lightweight FastAPI server that loads the HunFlair NER model ONCE at startup
and keeps it in memory.

The main pipeline calls this server via HTTP instead of loading the
1.4 GB+ model into the same process, which caused OOM crashes.

Usage:
    # Terminal 1 — start the model server (loads models once, stays running)
    ./start-ner-server.sh

    # Terminal 2 — run your pipeline as normal
    # (set NER_SERVER_URL=http://localhost:8001 in .env)

Endpoints:
    GET  /health         → {"status": "ok", "models": [...]}
    POST /ner/hunflair   → run HunFlair NER on text
    POST /ner            → run HunFlair and return results
"""
from __future__ import annotations

try:
    import transformers as _tr
    if not hasattr(_tr, "set_seed"):
        from transformers.trainer_utils import set_seed as _set_seed
        _tr.set_seed = _set_seed
except Exception:
    pass

import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    _env = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(_env):
        load_dotenv(_env)
except ImportError:
    pass

_startup_errors: list[str] = []
_loaded_models: list[str] = []


HUNFLAIR_LABEL_MAP: dict[str, str] = {
    "Disease":  "DISEASE",
    "Chemical": "CHEMICAL",
    "Gene":     "GENE_OR_GENE_PRODUCT",
    "Species":  "ORGANISM",
    "CellLine": "CELL_LINE",
}

_hunflair_tagger: Optional[object] = None
_hunflair_tagger_lock = threading.Lock()


def _load_hunflair() -> None:
    global _hunflair_tagger
    print("[NER-Server] Loading HunFlair2 model (~1.4 GB) …", flush=True)
    try:
        from flair.nn import Classifier
        tagger = Classifier.load("hunflair2")
        with _hunflair_tagger_lock:
            _hunflair_tagger = tagger
        _loaded_models.append("hunflair")
        print("[NER-Server] ✅ HunFlair2 ready", flush=True)
    except Exception as e:
        msg = f"HunFlair2 failed to load: {e}"
        _startup_errors.append(msg)
        print(f"[NER-Server] ⚠️  {msg}", file=sys.stderr, flush=True)


def _run_hunflair(text: str, threshold: float = 0.72) -> list[dict]:
    if _hunflair_tagger is None:
        return []
    try:
        from flair.data import Sentence
        sentence = Sentence(text[:10000])
        with _hunflair_tagger_lock:
            _hunflair_tagger.predict(sentence)
    except Exception as e:
        print(f"[NER-Server] HunFlair predict error: {e}", file=sys.stderr)
        return []

    entities: list[dict] = []
    seen: set[str] = set()
    for span in sentence.get_spans("ner"):
        word = span.text.strip()
        lbl = span.get_label("ner")
        label = lbl.value
        score = lbl.score

        if not word or len(word) < 2 or score < threshold:
            continue
        entity_type = HUNFLAIR_LABEL_MAP.get(label, "OTHER")
        if entity_type == "OTHER":
            continue
        if word.lower() in {"the", "a", "an", "of", "in", "to", "is", "are", "was"}:
            continue

        key = word.lower()
        if key in seen:
            continue
        seen.add(key)

        entities.append({
            "text":       word,
            "normalized": key,
            "label":      entity_type,
            "start":      span.start_position,
            "end":        span.end_position,
            "negated":    False,
            "assertion":  "PRESENT",
            "confidence": round(score, 3),
            "source":     "hunflair",
        })
    return entities

@asynccontextmanager
async def _lifespan(app: FastAPI):
    threads: list[threading.Thread] = []

    if os.getenv("HUNFLAIR_ENABLED", "true").lower() == "true":
        t = threading.Thread(target=_load_hunflair, daemon=True)
        t.start()
        threads.append(t)

    if not threads:
        print("[NER-Server] HunFlair not enabled. Set HUNFLAIR_ENABLED=true.", flush=True)
    else:
        # Wait until model finishes loading (success or failure) — max 3 min
        deadline = time.time() + 180
        while time.time() < deadline:
            if len(_loaded_models) + len(_startup_errors) >= len(threads):
                break
            time.sleep(1)

    yield


app = FastAPI(
    title="Bio-Semantic NER Server",
    description="Heavy-model NER server (HunFlair). Keeps model warm so the main pipeline stays lightweight.",
    version="1.0.0",
    lifespan=_lifespan,
)

class NERRequest(BaseModel):
    text: str
    threshold: float = 0.75

@app.get("/health")
def health():
    return {
        "status":         "ok" if _loaded_models else "loading",
        "models_ready":   _loaded_models,
        "startup_errors": _startup_errors,
    }


@app.post("/ner/hunflair")
def ner_hunflair(req: NERRequest):
    if _hunflair_tagger is None:
        raise HTTPException(503, "HunFlair model not ready yet — check /health")
    return {"entities": _run_hunflair(req.text, req.threshold)}


@app.post("/ner")
def ner_combined(req: NERRequest):
    """Run HunFlair and return entity list."""
    if _hunflair_tagger is None:
        raise HTTPException(503, "HunFlair model not ready yet — check /health")

    entities = _run_hunflair(req.text, req.threshold)
    return {"entities": entities, "models_used": _loaded_models}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("NER_SERVER_PORT", "8001"))
    print(f"[NER-Server] Starting on http://localhost:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
