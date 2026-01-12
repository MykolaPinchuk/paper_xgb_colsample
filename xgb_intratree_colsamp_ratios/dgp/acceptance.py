from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..modeling.metrics import Metrics, compute_binary_metrics
from ..modeling.xgb_train import train_xgb
from .common import GeneratedDataset


@dataclass(frozen=True)
class AcceptanceStats:
    oracle: Metrics
    engineered_ratio: Metrics
    best1d: Metrics
    feasibility_xgb: Metrics
    closes_oracle_gap_frac_rocauc: float
    closes_oracle_gap_frac_prauc: float
    accepted: bool


def _fit_1d_logit_and_score(
    x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(solver="lbfgs", max_iter=200),
    )
    model.fit(x_train.reshape(-1, 1), y_train)
    return model.predict_proba(x_valid.reshape(-1, 1))[:, 1]


def best_1d_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    *,
    max_features: int | None = None,
    force_cols: list[str] | None = None,
) -> Metrics:
    force = set(force_cols or [])

    all_cols = list(X_train.columns)
    if not all_cols:
        raise ValueError("No features available for best_1d_baseline")

    if max_features is None or int(max_features) >= len(all_cols):
        candidate_cols = all_cols
    else:
        k = max(1, int(max_features))
        X = X_train.to_numpy(dtype=np.float32, copy=False)
        y = y_train.astype(np.float32, copy=False)
        y_centered = y - float(y.mean())
        y_std = float(y_centered.std())
        if y_std <= 0:
            candidate_cols = list(force) if force else [all_cols[0]]
        else:
            X_centered = X - X.mean(axis=0, keepdims=True)
            x_std: NDArray[np.float32] = X_centered.std(axis=0)
            denom = (x_std * y_std).astype(np.float32)
            denom = np.where(denom > 0, denom, np.float32(1.0))
            corr = (X_centered * y_centered.reshape(-1, 1)).mean(axis=0) / denom
            score = np.abs(corr)
            score = np.where(np.isfinite(score), score, 0.0)

            col_to_idx = {c: i for i, c in enumerate(all_cols)}
            forced_cols = [c for c in all_cols if c in force]
            remaining = [c for c in all_cols if c not in force]
            if remaining:
                remaining_scores = score[[col_to_idx[c] for c in remaining]]
                k_remaining = max(0, k - len(forced_cols))
                if k_remaining <= 0:
                    top_remaining = []
                elif k_remaining >= len(remaining):
                    top_remaining = remaining
                else:
                    idx = np.argpartition(-remaining_scores, kth=k_remaining - 1)[:k_remaining]
                    top_remaining = [remaining[int(i)] for i in idx]
                candidate_cols = forced_cols + top_remaining
            else:
                candidate_cols = forced_cols or [all_cols[0]]

    best: Metrics | None = None
    for col in candidate_cols:
        score = _fit_1d_logit_and_score(
            X_train[col].to_numpy(dtype=float),
            y_train,
            X_valid[col].to_numpy(dtype=float),
        )
        m = compute_binary_metrics(y_valid, score)
        if best is None or m.roc_auc > best.roc_auc:
            best = m
    if best is None:
        raise ValueError("No candidate features available for best_1d_baseline")
    return best


def _gap_frac(oracle: float, best1d: float, feas: float) -> float:
    denom = oracle - best1d
    if denom <= 0:
        return 0.0
    return float((feas - best1d) / denom)


def evaluate_acceptance(
    dataset: GeneratedDataset,
    *,
    acceptance_cfg: dict[str, Any],
    xgb_fixed_cfg: dict[str, Any],
    nthread: int,
    seed: int,
) -> AcceptanceStats:
    tr = dataset.splits["train"]
    va = dataset.splits["valid"]

    if dataset.dgp_name == "continuous_logratio":
        oracle_score = va.latents["V"]
        engineered_score = va.X_engineered["r_logratio"].to_numpy(dtype=float)
    elif dataset.dgp_name == "count_exposure":
        oracle_score = va.latents["V"]
        engineered_score = va.X_engineered["r_logcountdiff"].to_numpy(dtype=float)
    elif dataset.dgp_name == "multicomponent":
        oracle_score = va.latents["V_sum"]
        engineered_score = va.X_engineered["r_logratio_sum"].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown dgp_name={dataset.dgp_name!r}")

    oracle = compute_binary_metrics(va.y, oracle_score)
    engineered_ratio = compute_binary_metrics(va.y, engineered_score)

    forced_cols: list[str] = []
    for a, b in dataset.intended_pairs:
        forced_cols.extend([a, b])
    best1d = best_1d_baseline(
        tr.X_base,
        tr.y,
        va.X_base,
        va.y,
        max_features=acceptance_cfg.get("best1d_max_features", None),
        force_cols=forced_cols,
    )

    feas_cfg = dict(acceptance_cfg.get("feasibility", {}))
    feas_depth = int(feas_cfg.get("max_depth", 7))
    feas_estimators = int(feas_cfg.get("n_estimators", 800))
    feas_esr = int(feas_cfg.get("early_stopping_rounds", 50))
    feas_fixed = dict(xgb_fixed_cfg)
    feas_fixed_overrides = dict(feas_cfg.get("fixed_overrides", {}))
    feas_fixed.update(feas_fixed_overrides)
    feas_fixed.setdefault("min_child_weight", 1)
    feas_fixed.setdefault("subsample", 1.0)

    feas_train = train_xgb(
        X_train=tr.X_base,
        y_train=tr.y,
        X_valid=va.X_base,
        y_valid=va.y,
        fixed_params=feas_fixed,
        arm_params={
            "max_depth": feas_depth,
            "colsample_bylevel": 1.0,
            "colsample_bynode": 1.0,
        },
        n_estimators=feas_estimators,
        early_stopping=True,
        early_stopping_rounds=feas_esr,
        nthread=nthread,
        seed=seed,
    )
    feas_score = feas_train.model.predict_proba(va.X_base)[:, 1]
    feasibility_xgb = compute_binary_metrics(va.y, feas_score)

    gap_frac_roc = _gap_frac(oracle.roc_auc, best1d.roc_auc, feasibility_xgb.roc_auc)
    gap_frac_pr = _gap_frac(oracle.pr_auc, best1d.pr_auc, feasibility_xgb.pr_auc)

    min_gap_frac_roc = float(
        acceptance_cfg.get(
            "min_xgb_closes_oracle_gap_frac_rocauc",
            acceptance_cfg["min_xgb_closes_oracle_gap_frac"],
        )
    )
    min_gap_frac_pr = float(
        acceptance_cfg.get(
            "min_xgb_closes_oracle_gap_frac_prauc",
            acceptance_cfg["min_xgb_closes_oracle_gap_frac"],
        )
    )

    accepted = True
    if (oracle.roc_auc - best1d.roc_auc) < float(acceptance_cfg["min_oracle_minus_best1d_rocauc"]):
        accepted = False
    if (oracle.pr_auc - best1d.pr_auc) < float(acceptance_cfg["min_oracle_minus_best1d_prauc"]):
        accepted = False
    if gap_frac_roc < min_gap_frac_roc:
        accepted = False
    if gap_frac_pr < min_gap_frac_pr:
        accepted = False

    return AcceptanceStats(
        oracle=oracle,
        engineered_ratio=engineered_ratio,
        best1d=best1d,
        feasibility_xgb=feasibility_xgb,
        closes_oracle_gap_frac_rocauc=gap_frac_roc,
        closes_oracle_gap_frac_prauc=gap_frac_pr,
        accepted=accepted,
    )
