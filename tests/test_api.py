import pytest
from fastapi.testclient import TestClient

import cfrec.api as api
from cfrec.config import Config
from cfrec.fetch import HandleNotFound
from cfrec.train import train
import joblib


@pytest.fixture()
def client(small_ds, tmp_path, monkeypatch):
    artifact = tmp_path / "model.joblib"
    joblib.dump(train(small_ds, Config(min_history=10)), artifact)
    monkeypatch.setenv("CFREC_ARTIFACT", str(artifact))

    states = {s.handle: s for s in small_ds.user_states(small_ds.end_time)}

    def fake_live(handle, pid_to_col, ratings):
        if handle not in states:
            raise HandleNotFound("not found")
        return states[handle]

    monkeypatch.setattr(api, "get_live_state", fake_live)
    api.CACHE.clear()
    with TestClient(api.app) as c:
        yield c, next(iter(states))


def test_recommend_returns_unsolved_problems(client, small_ds):
    c, handle = client
    r = c.get(f"/recommend/{handle}?k=5")
    assert r.status_code == 200
    body = r.json()
    assert len(body["recommendations"]) == 5
    solved = {small_ds.problems.pid[i] for i in next(s for s in small_ds.user_states(small_ds.end_time) if s.handle == handle).cols}
    assert not ({p["pid"] for p in body["recommendations"]} & solved)
    assert body["inference_ms"] >= 0


def test_unknown_handle_is_404(client):
    c, _ = client
    assert c.get("/recommend/nobody_here").status_code == 404


def test_health_and_index(client):
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/").status_code == 200
