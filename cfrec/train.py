"""Train the serving model on ALL available data and save it for the API.

    python -m cfrec.train --data data/processed --out artifacts/model.joblib

The ranker learns from the newest labelled window (the last `ranker_label_days` of data);
the recommender itself is fitted on the full history.
"""
import argparse
import time
from pathlib import Path

import joblib

from .config import Config
from .data import Dataset
from .evaluate import DAY, training_frame
from .recommender import FEATURES, Ranker, Recommender


def train(ds: Dataset, cfg: Config = None) -> dict:
    cfg = cfg or Config()
    t_end = ds.end_time
    t1 = t_end - cfg.ranker_label_days * DAY
    frame, n_users = training_frame(ds, cfg, t1, t_end)
    ranker = Ranker(FEATURES, cfg).fit(frame)
    rec = Recommender(cfg).fit(ds, t_end)
    return {
        "rec": rec,
        "ranker": ranker,
        "trained_at": int(time.time()),
        "data_end": int(t_end),
        "ranker_training_users": n_users,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="artifacts/model.joblib")
    args = ap.parse_args()
    bundle = train(Dataset.load(args.data))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB), ranker trained on {bundle['ranker_training_users']} users")


if __name__ == "__main__":
    main()
