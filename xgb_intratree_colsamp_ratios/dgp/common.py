from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


DistVariant = Literal["normal", "student_t", "mixture"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def _scale_to_sigma(x: np.ndarray, sigma: float) -> np.ndarray:
    std = float(np.std(x))
    if std <= 0:
        return np.zeros_like(x)
    return x * (float(sigma) / std)


def sample_symmetric(
    rng: np.random.Generator,
    size: int,
    sigma: float,
    dist_variant: DistVariant,
    *,
    student_t_df: float = 5.0,
    mixture_mean: float = 1.0,
) -> np.ndarray:
    if dist_variant == "normal":
        return rng.normal(loc=0.0, scale=float(sigma), size=int(size))
    if dist_variant == "student_t":
        x = rng.standard_t(df=float(student_t_df), size=int(size))
        return _scale_to_sigma(x, sigma)
    if dist_variant == "mixture":
        signs = rng.choice([-1.0, 1.0], size=int(size))
        x = rng.normal(loc=signs * float(mixture_mean), scale=1.0, size=int(size))
        x = x - float(np.mean(x))
        return _scale_to_sigma(x, sigma)
    raise ValueError(f"Unknown dist_variant={dist_variant!r}")


def calibrate_intercept_for_prevalence(
    linear_score: np.ndarray,
    prevalence_target: float,
    *,
    max_iter: int = 80,
    lo: float = -60.0,
    hi: float = 60.0,
) -> float:
    target = float(prevalence_target)
    if not (0.0 < target < 1.0):
        raise ValueError("prevalence_target must be in (0,1)")

    lo_val = float(np.mean(sigmoid(linear_score + lo)))
    hi_val = float(np.mean(sigmoid(linear_score + hi)))
    if lo_val > target:
        return lo
    if hi_val < target:
        return hi

    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        mid_val = float(np.mean(sigmoid(linear_score + mid)))
        if mid_val < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class SplitData:
    X_base: pd.DataFrame
    X_engineered: pd.DataFrame
    y: np.ndarray
    latents: dict[str, np.ndarray]


@dataclass(frozen=True)
class GeneratedDataset:
    dgp_name: str
    params: dict[str, Any]
    intended_pairs: list[tuple[str, str]]
    splits: dict[str, SplitData]
