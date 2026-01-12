from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class Metrics:
    roc_auc: float
    pr_auc: float


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    return Metrics(
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
    )


def logit(p: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, float(eps), 1.0 - float(eps))
    return np.log(p / (1.0 - p))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("pearson_corr requires x and y to have the same shape")
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.mean(x * x)) * np.sqrt(np.mean(y * y)))
    if denom <= 0.0:
        return 0.0
    return float(np.mean(x * y) / denom)

