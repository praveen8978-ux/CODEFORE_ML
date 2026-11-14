# Codeforces problem recommender

Recommends the next Codeforces problems for a user: right difficulty, matching topics, and what similar users went on to solve. The project is built around an honest offline evaluation: a temporal split, strong baselines, ablations and confidence intervals.

## How it works

Two stages:

1. **Candidate generation.** Two generators each propose 100 problems the user has not solved: item-based collaborative filtering (cosine similarity over co-solves, computed without a P x P matrix) and popularity within the user's rating window.
2. **Re-ranking.** A LightGBM LambdaRank model scores each candidate with 14 features: difficulty fit against the user's rating, recent and 90th-percentile solved rating, CF score and rank, tag affinity, popularity, problem age, user activity.

## Evaluation protocol

```
history ......... T1 ......... T2 ......... T_end
                   |  ranker    |  evaluation
                   |  labels    |  labels
```

* The ranker is trained at T1 using only solves before T1; its labels are what users newly solved in [T1, T2).
* All methods are evaluated at T2 using only solves before T2; the target is what each user newly solved in [T2, T_end) (default: 60 days).
* `tests/test_recommender.py::test_no_future_leakage` checks that adding solves after the cutoff cannot change any feature.
* Metrics: Hit@10, Recall@10, nDCG@10, right-difficulty@10 (share of recommendations rated within [rating - 100, rating + 300]), catalog coverage. Confidence intervals are 95% bootstrap over users, and the paired difference between LightGBM and each baseline is reported too.
* Baselines: global popularity, rating-band popularity, item-CF. Ablations drop one feature group at a time.

## Results

Fill this in from your own run (`results/results.md`). Do not report numbers from the synthetic data.

| Method | Hit@10 | Recall@10 | nDCG@10 | Right-difficulty@10 |
|---|---|---|---|---|
| popularity | | | | |
| rating_band_popularity | | | | |
| item_cf | | | | |
| lightgbm_rerank | | | | |

Setup: __ users sampled (stratified by rating), data downloaded on __, evaluation window __.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Download public data (resumable; about 70 min for 1000 users because the API allows 1 request / 2 s)
python -m cfrec.fetch --users 600 --raw data/raw --out data/processed

# 2. Evaluate (prints a results table, writes results/results.md and results.json)
python -m cfrec.evaluate --data data/processed --out results

# 3. Train the serving model and start the web app
python -m cfrec.train --data data/processed --out artifacts/model.joblib
CFREC_ARTIFACT=artifacts/model.joblib uvicorn cfrec.api:app --port 8000
# open http://localhost:8000

# Tests (use synthetic data, no network needed)
python -m pytest
```

To try the pipeline without downloading anything: `python -m cfrec.synth --out data/synthetic`, then pass `--data data/synthetic`. Synthetic results only prove the code runs.

## Design notes and limitations

* **Task.** The target is *practice picks*: problems published before the cutoff that the user newly solves later (mostly upsolving). It does not predict what happens in a future live contest.
* **User sample.** Active rated users, stratified by rating bin. The sample is not the whole population.
* **Rating at the cutoff** comes from each user's rating history; users without a rating before the cutoff are excluded.
* **Unrated problems** are not recommended (no difficulty signal).
* **Popularity** in features and baselines is computed from the sampled users' training history, not from the API's `solvedCount`, which includes the future.
* **API latency.** The `/recommend` endpoint reports `fetch_ms` (waiting on the Codeforces API) and `inference_ms` (model only) separately. Quote `inference_ms` as the model's latency.
* **Data.** `data/` is git-ignored. Don't commit raw dumps; the data is per-user public activity, so share aggregate results only.
* **Deploy.** The Dockerfile runs one service (API plus the static page). Free tiers can sleep when idle, so the first request after a pause is slow.

## Layout

```
cfrec/fetch.py        download and cache Codeforces data, build CSVs
cfrec/data.py         Dataset and point-in-time user state
cfrec/recommender.py  candidate generation, features, LightGBM ranker
cfrec/evaluate.py     temporal-split evaluation, ablations, bootstrap CIs
cfrec/metrics.py      Hit, Recall, nDCG, right-difficulty, bootstrap
cfrec/train.py        train the serving artifact
cfrec/api.py          FastAPI service
cfrec/synth.py        synthetic data for tests only
static/index.html     minimal web UI
tests/                metrics, leakage, parsers, API
```
