import numpy as np
import pandas as pd

from cfrec.config import Config
from cfrec.data import Dataset
from cfrec.evaluate import evaluate, split_times
from cfrec.recommender import Recommender, positives_between


def test_itemcf_matches_bruteforce(small_ds):
    cfg = Config()
    t1, t2, t_end = split_times(small_ds, cfg)
    rec = Recommender(cfg).fit(small_ds, t2)
    states = small_ds.user_states(t2)[:5]
    Xq = rec.state_matrix(states)
    fast = rec.itemcf_scores(Xq)

    X = rec.X.toarray()
    n = X.sum(0)
    C = X.T @ X
    sim = C / np.sqrt(np.outer(np.maximum(n, 1), np.maximum(n, 1)))
    slow = Xq.toarray() @ sim
    unsolved = Xq.toarray() == 0
    np.testing.assert_allclose(fast[unsolved], slow[unsolved], rtol=1e-4, atol=1e-5)


def test_no_future_leakage(small_ds):
    """Adding solves from after the cutoff must not change any feature at the cutoff."""
    cfg = Config()
    t1, t2, t_end = split_times(small_ds, cfg)
    base = Recommender(cfg).fit(small_ds, t2)
    frame_a = base.build_frame(small_ds.user_states(t2))

    # inject NEW solves (problems the user has not solved) dated after the cutoff
    rng = np.random.default_rng(0)
    all_pids = small_ds.problems.pid.to_numpy()
    rows = []
    for handle, g in small_ds.solves.groupby("handle"):
        unsolved = np.setdiff1d(all_pids, g.pid.to_numpy())
        for pid in rng.choice(unsolved, size=30, replace=False):
            rows.append((handle, pid, t2 + 100))
    future = pd.DataFrame(rows, columns=["handle", "pid", "time"])
    ds2 = Dataset(
        small_ds.problems,
        pd.concat([small_ds.solves[["handle", "pid", "time"]], future]),
        pd.DataFrame({"handle": [], "time": [], "rating": []}),
    )
    ds2._rh = small_ds._rh  # rating history is not what this test is about
    assert len(ds2.solves) == len(small_ds.solves) + len(future)

    frame_b = Recommender(cfg).fit(ds2, t2).build_frame(ds2.user_states(t2))
    pd.testing.assert_frame_equal(frame_a.reset_index(drop=True), frame_b.reset_index(drop=True))

    # control: the same injection DOES change features if the cutoff moves past it
    later = t2 + 200
    f_old = Recommender(cfg).fit(small_ds, later).build_frame(small_ds.user_states(later))
    f_new = Recommender(cfg).fit(ds2, later).build_frame(ds2.user_states(later))
    assert not f_old.reset_index(drop=True).equals(f_new.reset_index(drop=True))


def test_positives_exclude_history(small_ds):
    cfg = Config()
    t1, t2, t_end = split_times(small_ds, cfg)
    rec = Recommender(cfg).fit(small_ds, t2)
    states = small_ds.user_states(t2)
    pos = positives_between(small_ds, states, rec, t2, t_end)
    for s, p in zip(states, pos):
        assert not (set(s.cols.tolist()) & p)
        assert all(rec.eligible_base[c] for c in p)


def test_evaluate_runs_and_metrics_valid(small_ds):
    res = evaluate(small_ds, Config(min_history=10), ablate=False)
    k = res["meta"]["k"]
    assert res["meta"]["eval_users"] > 0
    for name, m in res["methods"].items():
        for key in (f"hit@{k}", f"recall@{k}", f"ndcg@{k}", f"right_difficulty@{k}"):
            mean, lo, hi = m[key]
            assert 0.0 <= lo <= mean <= hi <= 1.0, (name, key)
