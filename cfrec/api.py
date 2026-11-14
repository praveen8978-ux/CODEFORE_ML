"""FastAPI service: GET /recommend/{handle}?k=10

    CFREC_ARTIFACT=artifacts/model.joblib uvicorn cfrec.api:app --port 8000

The response reports `fetch_ms` (time spent waiting on the Codeforces API, which enforces a
request every ~2 s) separately from `inference_ms` (candidate generation + re-ranking), so
the model's own latency is measured honestly.
"""
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from .fetch import CFError, HandleNotFound, get_live_state

STATIC = Path(__file__).resolve().parent.parent / "static"
BUNDLE = {}
CACHE = {}  # (handle, k) -> (timestamp, payload)
CACHE_TTL = 600
_fetch_lock = threading.Lock()  # one Codeforces API call sequence at a time (rate limit)


@asynccontextmanager
async def lifespan(app):
    BUNDLE.update(joblib.load(os.environ.get("CFREC_ARTIFACT", "artifacts/model.joblib")))
    yield
    BUNDLE.clear()


app = FastAPI(title="Codeforces problem recommender", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "trained_at": BUNDLE.get("trained_at"), "data_end": BUNDLE.get("data_end")}


@app.get("/recommend/{handle}")
def recommend(handle: str, k: int = Query(10, ge=1, le=50)):
    key = (handle.lower(), k)
    hit = CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return {**hit[1], "cached": True}

    rec, ranker = BUNDLE["rec"], BUNDLE["ranker"]
    t0 = time.perf_counter()
    try:
        with _fetch_lock:
            state = get_live_state(handle, {p: i for i, p in enumerate(rec.problems.pid)}, rec.rating)
    except HandleNotFound:
        raise HTTPException(404, f"Codeforces handle '{handle}' not found")
    except CFError as e:
        raise HTTPException(502, f"Codeforces API unavailable: {e}")
    fetch_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    recs = rec.recommend(state, ranker, k=k)
    inference_ms = (time.perf_counter() - t1) * 1000

    payload = {
        "handle": state.handle,
        "rating": state.rating,
        "solved": int(len(state.cols)),
        "fetch_ms": round(fetch_ms, 1),
        "inference_ms": round(inference_ms, 1),
        "recommendations": [
            {
                "pid": r.pid,
                "name": r.name,
                "rating": r.rating,
                "tags": list(r.tags),
                "url": f"https://codeforces.com/contest/{r.contest_id}/problem/{r.index}",
                "score": round(float(r.score), 4),
            }
            for r in recs.itertuples()
        ],
        "cached": False,
    }
    CACHE[key] = (time.time(), payload)
    return payload


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
