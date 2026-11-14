from dataclasses import dataclass


@dataclass
class Config:
    # Recommendation list size used by every metric
    k: int = 10
    # Evaluation window: "what will the user newly solve in the next N days?"
    eval_days: int = 60
    # Window used to build training labels for the ranker (strictly before the eval window)
    ranker_label_days: int = 60
    # Users need at least this many solved problems before the cutoff to be evaluated
    min_history: int = 20
    # "Right difficulty" window relative to the user's rating at the cutoff
    window_low: int = -100
    window_high: int = 300
    # Candidates taken from each generator (item-CF, rating-band popularity) before re-ranking
    n_candidates: int = 100
    seed: int = 42
