from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


@dataclass(frozen=True)
class TrainResult:
    model: XGBClassifier
    seconds: float
    best_iteration: int | None
    n_estimators_used: int


def train_xgb(
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    fixed_params: dict[str, Any],
    arm_params: dict[str, Any],
    n_estimators: int,
    early_stopping: bool,
    early_stopping_rounds: int | None,
    nthread: int,
    seed: int,
) -> TrainResult:
    params = dict(fixed_params)
    params.update(arm_params)
    params.setdefault("verbosity", 0)
    params.setdefault("eval_metric", "aucpr")
    if early_stopping:
        if early_stopping_rounds is None:
            raise ValueError("early_stopping_rounds required when early_stopping=True")
        params.setdefault("early_stopping_rounds", int(early_stopping_rounds))

    model = XGBClassifier(
        **params,
        n_estimators=int(n_estimators),
        n_jobs=int(nthread),
        random_state=int(seed),
    )

    t0 = perf_counter()
    if early_stopping:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, verbose=False)
    seconds = perf_counter() - t0

    best_iteration: int | None = None
    n_estimators_used = int(n_estimators)
    if early_stopping and hasattr(model, "best_iteration") and model.best_iteration is not None:
        best_iteration = int(model.best_iteration)
        n_estimators_used = best_iteration + 1

    return TrainResult(
        model=model,
        seconds=float(seconds),
        best_iteration=best_iteration,
        n_estimators_used=n_estimators_used,
    )
