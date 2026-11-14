"""Ranking metrics and bootstrap confidence intervals (resampling users)."""
import numpy as np


def hit_at_k(ranked, positives, k):
    return float(any(c in positives for c in ranked[:k]))


def recall_at_k(ranked, positives, k):
    if not positives:
        return 0.0
    return sum(c in positives for c in ranked[:k]) / len(positives)


def ndcg_at_k(ranked, positives, k):
    if not positives:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, c in enumerate(ranked[:k]) if c in positives)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(positives), k)))
    return dcg / idcg


def right_difficulty_at_k(ranked, ratings, user_rating, k, low, high):
    """Share of the top-k whose rating is within [user_rating + low, user_rating + high]."""
    top = ranked[:k]
    if len(top) == 0:
        return 0.0
    r = ratings[np.asarray(top)]
    return float(np.mean((r >= user_rating + low) & (r <= user_rating + high)))


def bootstrap_ci(values, n_boot=1000, seed=0, alpha=0.05):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a, b, n_boot=1000, seed=0, alpha=0.05):
    """Mean(a - b) with a bootstrap CI over users; both arrays are per-user metric values."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), n_boot, seed, alpha)
