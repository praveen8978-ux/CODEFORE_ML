"""Offline evaluation with a temporal split.

Timeline (T_end = just after the newest solve in the data):

    history ......... T1 ......... T2 ......... T_end
                       |  ranker    |  evaluation
                       |  labels    |  labels

* The re-ranker is trained at T1: features use only solves before T1, labels are the
  problems each user newly solved in [T1, T2).
* Every method is evaluated at T2: features use only solves before T2, and the target
  is the set of problems each user newly solved in [T2, T_end).

Nothing from the future reaches a feature, and the ranker never sees evaluation labels.

    python -m cfrec.evaluate --data data/processed --out results
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .config import Config
from .data import Dataset
from .recommender import (
    FEATURE_GROUPS,
    FEATURES,
    Ranker,
    Recommender,
    label_frame,
    positives_between,
)

DAY = 86400


def split_times(ds: Dataset, cfg: Config):
    t_end = ds.end_time
    t2 = t_end - cfg.eval_days * DAY
    t1 = t2 - cfg.ranker_label_days * DAY
    return t1, t2, t_end


def usable_with_positives(states, positives, cfg: Config):
    return [
        i
        for i, s in enumerate(states)
        if len(s.cols) >= cfg.min_history and np.isfinite(s.rating) and positives[i]
    ]


def top_from_frame(frame: pd.DataFrame, scores, n_users: int, k: int) -> list:
    df = (
        frame[["user_idx", "col"]]
        .assign(score=scores)
        .sort_values(["user_idx", "score"], ascending=[True, False], kind="stable")
    )
    out = [np.array([], dtype=int) for _ in range(n_users)]
    for u, g in df.groupby("user_idx", sort=False):
        out[u] = g.col.to_numpy()[:k]
    return out


def training_frame(ds: Dataset, cfg: Config, t1, t2):
    rec = Recommender(cfg).fit(ds, t1)
    states = ds.user_states(t1)
    pos = positives_between(ds, states, rec, t1, t2)
    keep = usable_with_positives(states, pos, cfg)
    states_k = [states[i] for i in keep]
    frame = label_frame(rec.build_frame(states_k), [pos[i] for i in keep])
    return frame, len(keep)


def _iso(t):
    return datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d")


def evaluate(ds: Dataset, cfg: Config = None, ablate: bool = True) -> dict:
    cfg = cfg or Config()
    t1, t2, t_end = split_times(ds, cfg)

    train_frame, n_train_users = training_frame(ds, cfg, t1, t2)
    variants = {"lightgbm_rerank": FEATURES}
    if ablate:
        for group, cols in FEATURE_GROUPS.items():
            variants[f"lgbm_without_{group}"] = [f for f in FEATURES if f not in cols]
    rankers = {name: Ranker(feats, cfg).fit(train_frame) for name, feats in variants.items()}

    rec = Recommender(cfg).fit(ds, t2)
    all_states = ds.user_states(t2)
    all_pos = positives_between(ds, all_states, rec, t2, t_end)
    keep = usable_with_positives(all_states, all_pos, cfg)
    states = [all_states[i] for i in keep]
    pos = [all_pos[i] for i in keep]

    recs = rec.baseline_recs(states, cfg.k)
    frame = rec.build_frame(states)
    for name, ranker in rankers.items():
        recs[name] = top_from_frame(frame, ranker.predict(frame), len(states), cfg.k)

    cand_sets = frame.groupby("user_idx").col.apply(set).to_dict()
    cand_recall = np.mean([len(cand_sets.get(u, set()) & pos[u]) / len(pos[u]) for u in range(len(states))])
    n_eligible_problems = int(rec.eligible_base.sum())

    per_user = {}
    summary = {}
    k = cfg.k
    for name, lists in recs.items():
        hit = [M.hit_at_k(lists[u], pos[u], k) for u in range(len(states))]
        rec_k = [M.recall_at_k(lists[u], pos[u], k) for u in range(len(states))]
        ndcg = [M.ndcg_at_k(lists[u], pos[u], k) for u in range(len(states))]
        right = [
            M.right_difficulty_at_k(lists[u], rec.rating, states[u].rating, k, cfg.window_low, cfg.window_high)
            for u in range(len(states))
        ]
        per_user[name] = {"hit": hit, "ndcg": ndcg}
        flat = np.concatenate([np.asarray(x, dtype=int) for x in lists]) if lists else np.array([], dtype=int)
        summary[name] = {
            f"hit@{k}": M.bootstrap_ci(hit),
            f"recall@{k}": M.bootstrap_ci(rec_k),
            f"ndcg@{k}": M.bootstrap_ci(ndcg),
            f"right_difficulty@{k}": M.bootstrap_ci(right),
            "catalog_coverage": float(len(set(flat.tolist())) / max(n_eligible_problems, 1)),
        }

    comparisons = {}
    if "lightgbm_rerank" in per_user:
        for other in ("item_cf", "rating_band_popularity", "popularity"):
            for metric in ("ndcg", "hit"):
                comparisons[f"lightgbm_rerank_minus_{other}__{metric}@{k}"] = M.paired_bootstrap_diff(
                    per_user["lightgbm_rerank"][metric], per_user[other][metric]
                )

    meta = {
        "ranker_training_window": f"{_iso(t1)} to {_iso(t2)}",
        "evaluation_window": f"{_iso(t2)} to {_iso(t_end)}",
        "eval_users": len(states),
        "ranker_training_users": n_train_users,
        "ranker_training_rows": int(len(train_frame)),
        "mean_candidates_per_user": float(frame.groupby("user_idx").size().mean()),
        "candidate_recall": float(cand_recall),
        "eligible_problems": n_eligible_problems,
        "k": k,
        "right_difficulty_window": [cfg.window_low, cfg.window_high],
    }
    return {"meta": meta, "methods": summary, "comparisons": comparisons}


def _fmt(ci):
    m, lo, hi = ci
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def to_markdown(res: dict) -> str:
    k = res["meta"]["k"]
    lines = [
        f"| Method | Hit@{k} | Recall@{k} | nDCG@{k} | Right-difficulty@{k} | Coverage |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in res["methods"].items():
        lines.append(
            f"| {name} | {_fmt(m[f'hit@{k}'])} | {_fmt(m[f'recall@{k}'])} | {_fmt(m[f'ndcg@{k}'])} "
            f"| {_fmt(m[f'right_difficulty@{k}'])} | {m['catalog_coverage']:.3f} |"
        )
    lines += ["", "Paired differences (mean over users, 95% bootstrap CI):", ""]
    for name, ci in res["comparisons"].items():
        lines.append(f"- {name}: {_fmt(ci)}")
    lines += ["", "Setup:", ""] + [f"- {key}: {val}" for key, val in res["meta"].items()]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--eval-days", type=int, default=Config.eval_days)
    ap.add_argument("--min-history", type=int, default=Config.min_history)
    args = ap.parse_args()
    cfg = Config(eval_days=args.eval_days, min_history=args.min_history)
    res = evaluate(Dataset.load(args.data), cfg, ablate=not args.no_ablation)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    md = to_markdown(res)
    (out / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
