"""Synthetic Codeforces-like data, ONLY for testing the pipeline.

Metrics computed on this data say nothing about real-world quality and must never be
reported on a resume. The generator plants structure (skill-matched difficulty, tag
preferences, problem quality) so that the code can be checked end to end.

    python -m cfrec.synth --out data/synthetic
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86400
TAGS = [f"tag{i}" for i in range(20)]


def generate(out_dir, n_users=250, n_contests=300, solves_range=(60, 220), seed=0):
    rng = np.random.default_rng(seed)
    t_end = 1_700_000_000
    starts = t_end - (n_contests - np.arange(n_contests)) * 3 * DAY

    rows = []
    base = np.array([900, 1200, 1500, 1800, 2200, 2600])
    for c in range(n_contests):
        for j, letter in enumerate("ABCDEF"):
            rating = int(np.clip(round((base[j] + rng.normal(0, 150)) / 100) * 100, 800, 3500))
            n_t = rng.integers(1, 4)
            # tag choice tilted by difficulty so tags carry some signal about rating
            w = np.exp(-0.5 * ((np.arange(20) - (rating - 800) / 150) / 4.0) ** 2) + 0.05
            tags = rng.choice(20, size=n_t, replace=False, p=w / w.sum())
            rows.append(
                {
                    "pid": f"{1000 + c}{letter}",
                    "contest_id": 1000 + c,
                    "index": letter,
                    "name": f"Problem {1000 + c}{letter}",
                    "rating": rating,
                    "tags": "|".join(TAGS[t] for t in tags),
                    "start": int(starts[c]),
                    "_tags": tags,
                }
            )
    problems = pd.DataFrame(rows)
    P = len(problems)
    quality = rng.lognormal(0, 0.7, P)
    p_rating = problems.rating.to_numpy(float)
    p_start = problems.start.to_numpy()
    tag_matrix = np.zeros((P, 20))
    for i, t in enumerate(problems._tags):
        tag_matrix[i, t] = 1

    solves, rating_rows = [], []
    for u in range(n_users):
        handle = f"user{u:04d}"
        skill0 = float(np.clip(rng.normal(1400, 300), 800, 2600))
        growth = rng.uniform(0.1, 0.5)
        join = int(starts[0] + rng.uniform(0, 0.6) * (t_end - starts[0]))
        pref = rng.dirichlet(np.full(20, 0.3))
        n_solves = int(rng.integers(*solves_range))
        times = np.sort(rng.uniform(join, t_end - 1, n_solves)).astype(int)
        solved = np.zeros(P, bool)

        def skill_at(t):
            return min(skill0 + growth * (t - join) / DAY, 3000)

        for t in times:
            n_pub = np.searchsorted(p_start, t, side="left")
            if n_pub == 0:
                continue
            s = skill_at(t)
            w = (
                quality[:n_pub]
                * np.exp(-0.5 * ((p_rating[:n_pub] - (s + 120)) / 220) ** 2)
                * (1 + 6 * (tag_matrix[:n_pub] @ pref))
            )
            w = w * ~solved[:n_pub]
            if w.sum() <= 0:
                continue
            i = rng.choice(n_pub, p=w / w.sum())
            solved[i] = True
            solves.append((handle, problems.pid.iloc[i], int(t)))
        for ci in range(0, n_contests, 4):
            if starts[ci] > join:
                rating_rows.append((handle, int(starts[ci] + 7200), int(skill_at(starts[ci]) + rng.normal(0, 40))))

    solves_df = pd.DataFrame(solves, columns=["handle", "pid", "time"])
    problems["solved_count_api"] = problems.pid.map(solves_df.pid.value_counts()).fillna(0).astype(int)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    problems.drop(columns="_tags").to_csv(out / "problems.csv", index=False)
    solves_df.to_csv(out / "solves.csv", index=False)
    pd.DataFrame(rating_rows, columns=["handle", "time", "rating"]).to_csv(out / "rating_history.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--users", type=int, default=250)
    args = ap.parse_args()
    print("wrote", generate(args.out, n_users=args.users))
