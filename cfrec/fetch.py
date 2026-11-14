"""Download public Codeforces data through the official API and cache it on disk.

The download is resumable: every user is stored in its own JSON file, and existing
files are skipped. The API allows roughly one request every two seconds, so
fetching N users takes about N * 2 requests * 2.1 s (about 70 min for 1000 users).

Run:
    python -m cfrec.fetch --users 600 --raw data/raw --out data/processed
"""
import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://codeforces.com/api"
MIN_INTERVAL = 2.1
_last_call = 0.0


class CFError(RuntimeError):
    pass


class HandleNotFound(CFError):
    pass


def api_get(method, params=None, retries=4, timeout=90):
    """Call a Codeforces API method, respecting the rate limit, retrying transient errors."""
    global _last_call
    for attempt in range(retries):
        wait = MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            resp = requests.get(f"{BASE}/{method}", params=params, timeout=timeout)
            data = resp.json()
        except (requests.RequestException, ValueError):
            time.sleep(3 * 2**attempt)
            continue
        if data.get("status") == "OK":
            return data["result"]
        comment = data.get("comment", "")
        if "not found" in comment:
            raise HandleNotFound(comment)
        if "limit" in comment.lower():
            time.sleep(5 * (attempt + 1))
            continue
        time.sleep(3 * 2**attempt)
    raise CFError(f"{method} failed after {retries} attempts")


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def fetch_problem_catalog(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_json(raw_dir / "problemset.json", api_get("problemset.problems"))
    _write_json(raw_dir / "contests.json", api_get("contest.list", {"gym": "false"}))


RATING_BINS = [0, 1200, 1400, 1600, 1900, 2200, 5000]


def sample_handles(n, seed=42):
    """Stratified sample of active rated users so every skill level is represented."""
    users = api_get("user.ratedList", {"activeOnly": "true", "includeRetired": "false"}, timeout=300)
    rng = random.Random(seed)
    bins = [[] for _ in range(len(RATING_BINS) - 1)]
    for u in users:
        r = u.get("rating", 0)
        for i in range(len(RATING_BINS) - 1):
            if RATING_BINS[i] <= r < RATING_BINS[i + 1]:
                bins[i].append(u["handle"])
                break
    per_bin = n // len(bins)
    chosen = []
    for b in bins:
        rng.shuffle(b)
        chosen.extend(b[:per_bin])
    return chosen


def first_accepted(subs) -> dict:
    """pid -> time of the user's first accepted submission for that problem."""
    first_ok = {}
    for s in subs:
        if s.get("verdict") != "OK":
            continue
        p = s["problem"]
        if p.get("contestId") is None:
            continue
        pid = f"{p['contestId']}{p['index']}"
        t = s["creationTimeSeconds"]
        if pid not in first_ok or t < first_ok[pid]:
            first_ok[pid] = t
    return first_ok


def get_live_state(handle, pid_to_col, problem_ratings):
    """Build a UserState for a handle from the live API (used by the web service)."""
    import numpy as np

    from .data import UserState

    info = api_get("user.info", {"handles": handle})[0]
    solved = {pid: t for pid, t in first_accepted(api_get("user.status", {"handle": handle, "from": 1, "count": 100000})).items() if pid in pid_to_col}
    cols = np.array([pid_to_col[p] for p in solved], dtype=int)
    times = np.array(list(solved.values()), dtype=float)
    rating = info.get("rating")
    if rating is None:  # unrated user: estimate from what they solved
        rs = problem_ratings[cols]
        rs = rs[np.isfinite(rs)]
        rating = float(np.median(rs) + 100) if len(rs) else 1200.0
    return UserState(handle, cols, times, float(rating))


def fetch_user(handle, raw_dir: Path):
    path = raw_dir / "users" / f"{handle}.json"
    if path.exists():
        return
    subs = api_get("user.status", {"handle": handle, "from": 1, "count": 100000})
    first_ok = first_accepted(subs)
    ratings = api_get("user.rating", {"handle": handle})
    _write_json(
        path,
        {
            "handle": handle,
            "solves": [[pid, t] for pid, t in first_ok.items()],
            "ratings": [[r["ratingUpdateTimeSeconds"], r["newRating"]] for r in ratings],
        },
    )


def fetch_all(n_users, raw_dir: Path, seed=42):
    handles_path = raw_dir / "handles.json"
    if handles_path.exists():
        handles = json.loads(handles_path.read_text())
    else:
        fetch_problem_catalog(raw_dir)
        handles = sample_handles(n_users, seed)
        _write_json(handles_path, handles)
    handles = handles[:n_users] if len(handles) > n_users else handles
    for i, h in enumerate(handles, 1):
        try:
            fetch_user(h, raw_dir)
        except HandleNotFound:
            print(f"skip {h}: not found")
        except CFError as e:
            print(f"skip {h}: {e}")
        if i % 25 == 0:
            print(f"{i}/{len(handles)} users fetched", flush=True)


def build_dataset(raw_dir: Path, out_dir: Path):
    """Turn the raw JSON cache into the CSV files that cfrec.data.Dataset loads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ps = json.loads((raw_dir / "problemset.json").read_text())
    contests = json.loads((raw_dir / "contests.json").read_text())
    start = {c["id"]: c.get("startTimeSeconds") for c in contests}
    solved_count = {(s["contestId"], s["index"]): s.get("solvedCount", 0) for s in ps["problemStatistics"]}
    rows = []
    for p in ps["problems"]:
        cid = p.get("contestId")
        if cid is None or start.get(cid) is None:
            continue
        rows.append(
            {
                "pid": f"{cid}{p['index']}",
                "contest_id": cid,
                "index": p["index"],
                "name": p["name"],
                "rating": p.get("rating"),
                "tags": "|".join(p.get("tags", [])),
                "solved_count_api": solved_count.get((cid, p["index"]), 0),
                "start": start[cid],
            }
        )
    problems = pd.DataFrame(rows)
    known = set(problems.pid)

    solves, ratings = [], []
    for f in sorted((raw_dir / "users").glob("*.json")):
        u = json.loads(f.read_text())
        for pid, t in u["solves"]:
            if pid in known:
                solves.append((u["handle"], pid, t))
        for t, r in u["ratings"]:
            ratings.append((u["handle"], t, r))
    pd.DataFrame(solves, columns=["handle", "pid", "time"]).to_csv(out_dir / "solves.csv", index=False)
    pd.DataFrame(ratings, columns=["handle", "time", "rating"]).to_csv(out_dir / "rating_history.csv", index=False)
    problems.to_csv(out_dir / "problems.csv", index=False)
    print(f"problems={len(problems)} solves={len(solves)} users={len({s[0] for s in solves})}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--users", type=int, default=600)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--build-only", action="store_true", help="skip downloading, just rebuild the CSVs")
    args = ap.parse_args()
    raw = Path(args.raw)
    if not args.build_only:
        fetch_all(args.users, raw, args.seed)
    build_dataset(raw, Path(args.out))


if __name__ == "__main__":
    main()
