"""Measure model-only latency (candidate generation + features + re-ranking), without any
Codeforces API calls, over many real users.

    python -m cfrec.benchmark --data data/processed --model artifacts/model.joblib --n 200
"""
import argparse
import os
import platform
import time

import joblib
import numpy as np

from .data import Dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--model", default="artifacts/model.joblib")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    ds = Dataset.load(args.data)
    bundle = joblib.load(args.model)
    rec, ranker = bundle["rec"], bundle["ranker"]
    states = [
        s for s in ds.user_states(ds.end_time) if np.isfinite(s.rating) and len(s.cols) >= rec.cfg.min_history
    ]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(states), size=min(args.n, len(states)), replace=False)

    for i in pick[:5]:  # warm-up, not timed
        rec.recommend(states[i], ranker, k=args.k)

    ms = []
    for i in pick:
        t = time.perf_counter()
        rec.recommend(states[i], ranker, k=args.k)
        ms.append((time.perf_counter() - t) * 1000)

    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    print(f"requests: {len(ms)} (one user per request, k={args.k}, single process)")
    print(f"latency ms: p50={p50:.1f}  p95={p95:.1f}  p99={p99:.1f}  mean={np.mean(ms):.1f}")
    print(f"machine: {platform.platform()}, {os.cpu_count()} logical CPUs, Python {platform.python_version()}")


if __name__ == "__main__":
    main()
