import numpy as np
import pytest

from cfrec import metrics as M


def test_hit_and_recall():
    assert M.hit_at_k([5, 6, 7], {7}, 3) == 1.0
    assert M.hit_at_k([5, 6, 7], {7}, 2) == 0.0
    assert M.recall_at_k([5, 6, 7], {6, 7, 9, 10}, 3) == pytest.approx(0.5)


def test_ndcg_hand_computed():
    # relevant item at rank 2 (index 1), one relevant item in total
    assert M.ndcg_at_k([1, 2, 3], {2}, 3) == pytest.approx(1 / np.log2(3))
    # perfect ranking
    assert M.ndcg_at_k([1, 2], {1, 2}, 2) == pytest.approx(1.0)
    assert M.ndcg_at_k([9, 8], {1}, 2) == 0.0


def test_right_difficulty():
    ratings = np.array([800, 1500, 1700, 2500])
    # user rated 1500, window [-100, +300] -> [1400, 1800]; items 1 and 2 are inside
    assert M.right_difficulty_at_k([0, 1, 2, 3], ratings, 1500, 4, -100, 300) == pytest.approx(0.5)


def test_bootstrap_ci_contains_mean():
    v = np.random.default_rng(0).random(200)
    m, lo, hi = M.bootstrap_ci(v)
    assert lo <= m <= hi
