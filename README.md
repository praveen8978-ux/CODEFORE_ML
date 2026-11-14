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

Real Codeforces data, downloaded 2026-11-14: 596 active rated users sampled (stratified by rating bin), 11,520 problems, 288,275 solves. Evaluation window 2026-09-15 to 2026-11-14 (60 days), ranker trained on labels from 2026-07-17 to 2026-09-15, 341 evaluated users. Values are means with 95% bootstrap CIs over users.

| Method | Hit@10 | Recall@10 | nDCG@10 | Right-difficulty@10 | Coverage |
|---|---|---|---|---|---|
| popularity | 0.111 [0.079, 0.147] | 0.007 [0.003, 0.012] | 0.025 [0.015, 0.036] | 0.058 [0.042, 0.075] | 0.008 |
| rating_band_popularity | 0.147 [0.111, 0.185] | 0.007 [0.005, 0.009] | 0.027 [0.018, 0.037] | 0.991 [0.982, 1.000] | 0.027 |
| item_cf | 0.126 [0.091, 0.161] | 0.012 [0.007, 0.019] | 0.046 [0.031, 0.062] | 0.133 [0.111, 0.153] | 0.071 |
| **lightgbm_rerank** | **0.305 [0.258, 0.358]** | **0.028 [0.020, 0.038]** | **0.082 [0.064, 0.101]** | 0.719 [0.691, 0.749] | 0.079 |
| lgbm_without_itemcf | 0.299 [0.252, 0.349] | 0.027 [0.020, 0.037] | 0.079 [0.062, 0.098] | 0.728 [0.700, 0.758] | 0.075 |
| lgbm_without_tags | 0.308 [0.258, 0.358] | 0.030 [0.021, 0.040] | 0.083 [0.064, 0.102] | 0.694 [0.665, 0.725] | 0.084 |
| lgbm_without_difficulty_fit | 0.290 [0.243, 0.340] | 0.021 [0.016, 0.027] | 0.068 [0.052, 0.086] | 0.718 [0.691, 0.745] | 0.072 |
| lgbm_without_popularity | 0.246 [0.199, 0.293] | 0.018 [0.012, 0.024] | 0.064 [0.047, 0.081] | 0.719 [0.689, 0.750] | 0.103 |

Paired differences, LightGBM minus baseline (mean over users, 95% bootstrap CI):

| Baseline | Hit@10 | nDCG@10 |
|---|---|---|
| item_cf | +0.179 [0.129, 0.232] | +0.036 [0.019, 0.052] |
| rating_band_popularity | +0.158 [0.106, 0.205] | +0.055 [0.036, 0.073] |
| popularity | +0.194 [0.144, 0.243] | +0.057 [0.036, 0.077] |

What this shows, and what it doesn't:

* The re-ranker beats all three baselines on Hit@10 and nDCG@10, and every paired interval excludes zero. Against the strongest baseline, Hit@10 roughly doubles (0.147 to 0.305) and nDCG@10 rises from 0.046 (item-CF) to 0.082.
* Absolute numbers are low. The candidate generators contain only 13.9% of the problems users went on to solve, which caps what any re-ranker can reach, and the task (picking a user's next practice problems out of about 11,000) is hard.
* Paired ablations (full model minus the model without that feature group, mean over users with 95% bootstrap CI):

  | Feature group removed | nDCG@10 drop | Hit@10 drop |
  |---|---|---|
  | popularity | +0.018 [0.005, 0.030] | +0.059 [0.012, 0.103] |
  | difficulty fit | +0.014 [0.004, 0.024] | +0.015 [-0.021, 0.050] |
  | item-CF features | +0.003 [-0.007, 0.012] | +0.006 [-0.023, 0.035] |
  | tag affinity | -0.001 [-0.009, 0.007] | -0.003 [-0.041, 0.032] |

  Popularity and difficulty-fit features each contribute measurably to nDCG@10 (intervals exclude zero; for difficulty fit, Hit@10 does not). The item-CF and tag features add nothing measurable once the others are present. This only removes the features: item-CF still generates candidates, and no test here shows the generator is unnecessary.
* Right-difficulty is 0.991 for the rating-band baseline by construction. The re-ranker's 0.719 means it also recommends problems outside the window, which is learned from what users actually solve.
* Popularity is computed from the 596 sampled users, not from the whole site, so the popularity baseline is weaker than a site-wide count would be.

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
pip install -r requirements-dev.txt
python -m pytest
```

## Deploy (free, Render)

1. Train on your real data and commit the serving model (about a few MB; it contains problem metadata and an anonymous solve matrix, no handles):
   ```bash
   python -m cfrec.train --data data/processed --out artifacts/model.joblib
   git add artifacts/model.joblib && git commit -m "Add trained serving model" && git push
   ```
2. On render.com: New, Blueprint, pick this repo. `render.yaml` sets up a free Docker web service with a `/health` check.
3. Open the service URL, enter a handle, and check the result.

The service fetches the user's submissions from the Codeforces API on each request, so it depends on Codeforces accepting requests from the host's IP. Free instances sleep when idle, so the first request after a pause is slow. To refresh the model with new problems and solves, re-run the fetch, train and commit steps.

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
