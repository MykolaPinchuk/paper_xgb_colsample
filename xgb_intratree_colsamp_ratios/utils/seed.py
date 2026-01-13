from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Seeds:
    master: int
    scenario: int
    rep: int


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def cpu_count() -> int:
    return max(os.cpu_count() or 1, 1)


def default_xgb_threads(reserve_cpus: int) -> int:
    return max(cpu_count() - int(reserve_cpus), 1)

