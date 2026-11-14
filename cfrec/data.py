"""Dataset loading and per-user state at a point in time."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class UserState:
    """What we know about a user at one moment: solved problem columns, their times, rating."""

    handle: str
    cols: np.ndarray  # column index in Dataset.problems of every solved problem (unique)
    times: np.ndarray  # solve time (unix seconds), same order as cols
    rating: float  # rating at that moment, NaN if unknown
    rating_estimated: bool = False  # True when the rating was guessed (unrated live user)


class Dataset:
    def __init__(self, problems: pd.DataFrame, solves: pd.DataFrame, rating_hist: pd.DataFrame):
        self.problems = problems.reset_index(drop=True)
        self.pid_to_col = {pid: i for i, pid in enumerate(self.problems.pid)}
        s = solves[solves.pid.isin(self.pid_to_col)].copy()
        s["col"] = s.pid.map(self.pid_to_col).astype(int)
        s = s.drop_duplicates(["handle", "pid"]).sort_values("time", kind="stable").reset_index(drop=True)
        self.solves = s
        self._rh = {}
        for h, g in rating_hist.sort_values("time").groupby("handle"):
            self._rh[h] = (g.time.to_numpy(), g.rating.to_numpy(float))

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        problems = pd.read_csv(d / "problems.csv")
        problems["tags"] = problems.tags.fillna("").map(lambda s: [t for t in s.split("|") if t])
        problems["rating"] = problems.rating.astype(float)
        return cls(problems, pd.read_csv(d / "solves.csv"), pd.read_csv(d / "rating_history.csv"))

    @property
    def end_time(self) -> int:
        return int(self.solves.time.max()) + 1

    def rating_at(self, handle, t) -> float:
        """Rating after the last contest strictly before t (NaN if the user had none yet)."""
        if handle not in self._rh:
            return float("nan")
        times, ratings = self._rh[handle]
        i = np.searchsorted(times, t, side="left")
        return float(ratings[i - 1]) if i > 0 else float("nan")

    def user_states(self, cutoff) -> list:
        """One UserState per user, using only solves strictly before cutoff."""
        hist = self.solves[self.solves.time < cutoff]
        states = []
        for handle, g in hist.groupby("handle", sort=True):
            states.append(
                UserState(handle, g.col.to_numpy(), g.time.to_numpy(), self.rating_at(handle, cutoff))
            )
        return states
