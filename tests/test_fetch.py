"""The live API is unreachable in CI, so these tests feed fake API-shaped JSON through the parsers."""
import json

import pytest

import cfrec.fetch as fetch
from cfrec.data import Dataset


def _sub(cid, idx, t, verdict="OK"):
    return {"verdict": verdict, "creationTimeSeconds": t, "problem": {"contestId": cid, "index": idx}}


def test_first_accepted_keeps_earliest_ok_and_skips_failures():
    subs = [_sub(1, "A", 500), _sub(1, "A", 300), _sub(1, "B", 100, "WRONG_ANSWER"), {"verdict": "OK", "creationTimeSeconds": 1, "problem": {"index": "A"}}]
    assert fetch.first_accepted(subs) == {"1A": 300}


def test_build_dataset_roundtrip(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    (raw / "users").mkdir(parents=True)
    (raw / "problemset.json").write_text(
        json.dumps(
            {
                "problems": [
                    {"contestId": 1, "index": "A", "name": "P1", "rating": 800, "tags": ["math", "greedy"]},
                    {"contestId": 1, "index": "B", "name": "P2", "tags": []},
                    {"contestId": 999, "index": "A", "name": "no start time", "rating": 900, "tags": []},
                ],
                "problemStatistics": [{"contestId": 1, "index": "A", "solvedCount": 42}],
            }
        )
    )
    (raw / "contests.json").write_text(json.dumps([{"id": 1, "startTimeSeconds": 1000}]))
    (raw / "users" / "alice.json").write_text(
        json.dumps({"handle": "alice", "solves": [["1A", 2000], ["1B", 2100], ["777Z", 5]], "ratings": [[1500, 1200], [3000, 1300]]})
    )
    fetch.build_dataset(raw, out)
    ds = Dataset.load(out)
    assert set(ds.problems.pid) == {"1A", "1B"}  # problem without contest start time dropped
    assert ds.problems.set_index("pid").loc["1A", "tags"] == ["math", "greedy"]
    assert len(ds.solves) == 2  # solve of unknown problem dropped
    assert ds.rating_at("alice", 2000) == 1200
    assert ds.rating_at("alice", 1000) != ds.rating_at("alice", 1000)  # NaN before first contest
    state = ds.user_states(2050)[0]
    assert state.cols.tolist() == [ds.pid_to_col["1A"]]


def test_api_get_retries_then_raises(monkeypatch):
    calls = []

    class R:
        def json(self):
            return {"status": "FAILED", "comment": "Call limit exceeded"}

    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: calls.append(1) or R())
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    with pytest.raises(fetch.CFError):
        fetch.api_get("x", retries=3)
    assert len(calls) == 3


def test_api_get_handle_not_found_is_not_retried(monkeypatch):
    calls = []

    class R:
        def json(self):
            return {"status": "FAILED", "comment": "handles: User with handle zzz not found"}

    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: calls.append(1) or R())
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    with pytest.raises(fetch.HandleNotFound):
        fetch.api_get("user.info", {"handles": "zzz"})
    assert len(calls) == 1


def test_sample_handles_is_stratified(monkeypatch):
    users = [{"handle": f"u{i}", "rating": 800 + i * 5} for i in range(600)]
    monkeypatch.setattr(fetch, "api_get", lambda *a, **k: users)
    chosen = fetch.sample_handles(30, seed=1)
    ratings = [next(u["rating"] for u in users if u["handle"] == h) for h in chosen]
    bins = [sum(lo <= r < hi for r in ratings) for lo, hi in zip(fetch.RATING_BINS, fetch.RATING_BINS[1:])]
    assert all(b <= 5 for b in bins)
    assert len(chosen) <= 30
