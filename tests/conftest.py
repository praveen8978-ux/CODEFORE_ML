import pytest

from cfrec.data import Dataset
from cfrec.synth import generate


@pytest.fixture(scope="session")
def small_ds(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    generate(d, n_users=80, n_contests=150, solves_range=(40, 120), seed=1)
    return Dataset.load(d)
