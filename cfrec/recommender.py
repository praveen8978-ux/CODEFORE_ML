"""Two-stage recommender: candidate generation (item-CF + rating-band popularity), then LightGBM re-ranking.

Everything is fitted "as of" a cutoff time and only reads solves strictly before it,
so evaluation can replay the past without leaking the future.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse

from .config import Config
from .data import Dataset, UserState

DAY = 86400

FEATURES = [
    "f_rating",
    "f_user_rating",
    "f_rating_minus_user",
    "f_abs_diff",
    "f_rating_minus_recent",
    "f_rating_minus_p90",
    "f_log_pop",
    "f_age_days",
    "f_cf_score",
    "f_cf_rank",
    "f_cf_rel",
    "f_tag_aff",
    "f_user_log_n",
    "f_user_recent30",
]

# Used for ablations: drop one group at a time and see what the model loses.
FEATURE_GROUPS = {
    "itemcf": ["f_cf_score", "f_cf_rank", "f_cf_rel"],
    "tags": ["f_tag_aff"],
    "difficulty_fit": ["f_rating_minus_user", "f_abs_diff", "f_rating_minus_recent", "f_rating_minus_p90"],
    "popularity": ["f_log_pop", "f_age_days"],
}


def top_k(scores: np.ndarray, eligible: np.ndarray, k: int) -> list:
    """Per-row indices of the k highest-scoring eligible columns, best first."""
    s = np.where(eligible, scores, -np.inf)
    k_eff = min(k, s.shape[1])
    idx = np.argpartition(-s, k_eff - 1, axis=1)[:, :k_eff]
    out = []
    for u in range(s.shape[0]):
        row = idx[u]
        row = row[np.argsort(-s[u, row], kind="stable")]
        out.append(row[np.isfinite(s[u, row])])
    return out


class Recommender:
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()

    # ------------------------------------------------------------------ fitting
    def fit(self, ds: Dataset, cutoff: int) -> "Recommender":
        self.cutoff = int(cutoff)
        pr = ds.problems
        self.problems = pr[["pid", "name", "rating", "tags", "start", "contest_id", "index"]].reset_index(drop=True)
        n_problems = len(pr)
        hist = ds.solves[ds.solves.time < cutoff]
        user_codes = pd.factorize(hist.handle)[0]
        self.X = sparse.csr_matrix(
            (np.ones(len(hist), dtype=np.float32), (user_codes, hist.col.to_numpy())),
            shape=(int(user_codes.max()) + 1 if len(hist) else 0, n_problems),
        )
        self.pop = np.asarray(self.X.sum(axis=0)).ravel().astype(np.float32)
        self.inv_sqrt = (1.0 / np.sqrt(np.maximum(self.pop, 1.0))).astype(np.float32)
        self.rating = pr.rating.to_numpy(float)
        self.start = pr.start.fillna(0).to_numpy(float)
        self.eligible_base = (self.start < cutoff) & ~np.isnan(self.rating)
        vocab = sorted({t for tags in pr.tags for t in tags})
        tag_idx = {t: i for i, t in enumerate(vocab)}
        r, c = [], []
        for i, tags in enumerate(pr.tags):
            for t in tags:
                r.append(i)
                c.append(tag_idx[t])
        self.T = sparse.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(n_problems, len(vocab)))
        self.n_tags = np.asarray(self.T.sum(axis=1)).ravel()
        return self

    # --------------------------------------------------------------- primitives
    def state_matrix(self, states) -> sparse.csr_matrix:
        lens = [len(s.cols) for s in states]
        rows = np.repeat(np.arange(len(states)), lens)
        cols = np.concatenate([s.cols for s in states]) if states else np.array([], dtype=int)
        return sparse.csr_matrix(
            (np.ones(len(cols), dtype=np.float32), (rows, cols)), shape=(len(states), len(self.pop))
        )

    def eligibility(self, Xq) -> np.ndarray:
        """Published, rated and not yet solved by the user."""
        return (Xq.toarray() == 0) & self.eligible_base[None, :]

    def itemcf_scores(self, Xq) -> np.ndarray:
        """score(u, i) = sum over j solved by u of cosine(i, j).

        Computed without building the P x P similarity matrix:
        cosine(i, j) = C[i, j] / sqrt(n_i n_j) with C = X^T X, so
        score = ((Xq D) X^T) X D  where D = diag(1/sqrt(n)).
        """
        D = sparse.diags(self.inv_sqrt)
        W = (Xq @ D) @ self.X.T  # (q x users)
        S = W @ self.X  # (q x problems)
        S = S.toarray() if sparse.issparse(S) else np.asarray(S)
        return (S * self.inv_sqrt[None, :]).astype(np.float32)

    def band_scores(self, ratings: np.ndarray) -> np.ndarray:
        lo, hi = self.cfg.window_low, self.cfg.window_high
        r = ratings[:, None]
        in_window = (self.rating[None, :] >= r + lo) & (self.rating[None, :] <= r + hi)
        return self.pop[None, :] + 1e6 * in_window

    # ---------------------------------------------------------------- baselines
    def baseline_recs(self, states, k=None) -> dict:
        k = k or self.cfg.k
        Xq = self.state_matrix(states)
        elig = self.eligibility(Xq)
        ratings = np.array([s.rating for s in states], float)
        pop = np.broadcast_to(self.pop, elig.shape)
        return {
            "popularity": top_k(pop, elig, k),
            "rating_band_popularity": top_k(self.band_scores(ratings), elig, k),
            "item_cf": top_k(self.itemcf_scores(Xq), elig, k),
        }

    # ----------------------------------------------------- candidates + features
    def build_frame(self, states, now=None) -> pd.DataFrame:
        """Candidate generation + feature computation. One row per (user, candidate problem)."""
        now = self.cutoff if now is None else now
        Xq = self.state_matrix(states)
        elig = self.eligibility(Xq)
        ratings = np.array([s.rating for s in states], float)
        cf = self.itemcf_scores(Xq)
        n = self.cfg.n_candidates
        cand_cf = top_k(cf, elig, n)
        cand_band = top_k(self.band_scores(ratings), elig, n)

        # user-level statistics
        n_solved = np.array([len(s.cols) for s in states], float)
        recent30 = np.array([(s.times >= now - 30 * DAY).sum() for s in states], float)
        recent_med = np.full(len(states), np.nan)
        p90 = np.full(len(states), np.nan)
        for u, s in enumerate(states):
            solved_ratings = self.rating[s.cols]
            order = np.argsort(s.times)
            last20 = solved_ratings[order][-20:]
            if np.isfinite(last20).any():
                recent_med[u] = np.nanmedian(last20)
            if np.isfinite(solved_ratings).any():
                p90[u] = np.nanpercentile(solved_ratings, 90)

        share = (Xq @ self.T).toarray() / np.maximum(n_solved, 1)[:, None]
        tag_aff = np.asarray((self.T @ share.T).T) / np.maximum(self.n_tags, 1)[None, :]

        u_idx, c_idx, cf_rank = [], [], []
        cf_max = np.ones(len(states))
        for u in range(len(states)):
            cand = np.union1d(cand_cf[u], cand_band[u])
            if len(cand) == 0:
                continue
            e = np.sort(cf[u][elig[u]])
            cf_max[u] = max(float(e[-1]) if len(e) else 0.0, 1e-9)
            rank = len(e) - np.searchsorted(e, cf[u, cand], side="right")
            u_idx.append(np.full(len(cand), u))
            c_idx.append(cand)
            cf_rank.append(rank)
        if not u_idx:
            return pd.DataFrame(columns=["user_idx", "col"] + FEATURES)
        u_idx = np.concatenate(u_idx)
        c_idx = np.concatenate(c_idx)
        cf_rank = np.concatenate(cf_rank)

        f_rating = self.rating[c_idx]
        f_user_rating = ratings[u_idx]
        frame = pd.DataFrame(
            {
                "user_idx": u_idx,
                "col": c_idx,
                "f_rating": f_rating,
                "f_user_rating": f_user_rating,
                "f_rating_minus_user": f_rating - f_user_rating,
                "f_abs_diff": np.abs(f_rating - f_user_rating),
                "f_rating_minus_recent": f_rating - recent_med[u_idx],
                "f_rating_minus_p90": f_rating - p90[u_idx],
                "f_log_pop": np.log1p(self.pop[c_idx]),
                "f_age_days": np.log1p(np.maximum(now - self.start[c_idx], 0) / DAY),
                "f_cf_score": cf[u_idx, c_idx],
                "f_cf_rank": cf_rank,
                "f_cf_rel": cf[u_idx, c_idx] / cf_max[u_idx],
                "f_tag_aff": tag_aff[u_idx, c_idx],
                "f_user_log_n": np.log1p(n_solved[u_idx]),
                "f_user_recent30": recent30[u_idx],
            }
        )
        return frame

    # ------------------------------------------------------------------ serving
    def recommend(self, state: UserState, ranker: "Ranker", k=10, now=None) -> pd.DataFrame:
        now = time.time() if now is None else now
        frame = self.build_frame([state], now=now)
        if frame.empty:
            return pd.DataFrame(columns=["pid", "name", "rating", "tags", "score"])
        frame["score"] = ranker.predict(frame)
        best = frame.sort_values("score", ascending=False).head(k)
        out = self.problems.iloc[best.col.to_numpy()][["pid", "name", "rating", "tags", "contest_id", "index"]].copy()
        out["score"] = best.score.to_numpy()
        return out.reset_index(drop=True)


class Ranker:
    """LightGBM LambdaRank re-ranker over the generated candidates."""

    def __init__(self, features=None, cfg: Config = None):
        self.features = list(features or FEATURES)
        self.cfg = cfg or Config()

    def fit(self, frame: pd.DataFrame) -> "Ranker":
        frame = frame.sort_values("user_idx", kind="stable")
        has_pos = frame.groupby("user_idx").label.transform("max") > 0
        frame = frame[has_pos]
        groups = frame.groupby("user_idx", sort=True).size().to_numpy()
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=self.cfg.seed,
            verbose=-1,
        )
        self.model.fit(frame[self.features], frame.label.astype(int), group=groups)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(frame[self.features])


def positives_between(ds: Dataset, states, rec: Recommender, start, end) -> list:
    """Problems each user newly solved in [start, end) that the recommender could have suggested."""
    w = ds.solves[(ds.solves.time >= start) & (ds.solves.time < end)]
    by_handle = {h: g.col.to_numpy() for h, g in w.groupby("handle")}
    out = []
    for s in states:
        c = np.unique(by_handle.get(s.handle, np.array([], dtype=int)))
        c = c[rec.eligible_base[c]]
        c = np.setdiff1d(c, s.cols)
        out.append(set(c.tolist()))
    return out


def label_frame(frame: pd.DataFrame, positives: list) -> pd.DataFrame:
    frame = frame.copy()
    frame["label"] = [
        int(c in positives[u]) for u, c in zip(frame.user_idx.to_numpy(), frame.col.to_numpy())
    ]
    return frame
