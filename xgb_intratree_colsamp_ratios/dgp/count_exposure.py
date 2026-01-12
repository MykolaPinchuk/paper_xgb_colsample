from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import GeneratedDataset, SplitData, calibrate_intercept_for_prevalence, sigmoid
from .distractors import iid_noise_distractors, loge_distractors


def _neg_binomial_from_mean_phi(
    rng: np.random.Generator, mean: np.ndarray, phi: float
) -> np.ndarray:
    phi = float(phi)
    if phi <= 0:
        raise ValueError("phi must be > 0")
    mean = np.asarray(mean, dtype=float)
    mean = np.clip(mean, 0.0, 1e9)
    p = phi / (phi + mean)
    n = phi
    return rng.negative_binomial(n=n, p=p, size=mean.shape).astype(np.int32)


def generate(
    rng: np.random.Generator,
    *,
    n_train: int,
    n_valid: int,
    n_test: int,
    prevalence_target: float,
    mu_loge: float,
    sigma_loge: float,
    sigma_v: float,
    beta: float,
    dispersion_phi: float,
    smoothing_s: float,
    distractors: dict[str, Any] | None = None,
) -> GeneratedDataset:
    mu_loge = float(mu_loge)
    sigma_loge = float(sigma_loge)
    sigma_v = float(sigma_v)
    beta = float(beta)
    phi = float(dispersion_phi)
    smoothing_s = float(smoothing_s)

    def one_split(n: int) -> SplitData:
        loge = rng.normal(loc=mu_loge, scale=sigma_loge, size=int(n))
        e = np.exp(np.clip(loge, -20.0, 20.0))
        v = rng.normal(loc=0.0, scale=sigma_v, size=int(n))

        mean_a = e * np.exp(np.clip(v, -10.0, 10.0))
        mean_b = e * np.exp(np.clip(-v, -10.0, 10.0))
        a = _neg_binomial_from_mean_phi(rng, mean_a, phi=phi)
        b = _neg_binomial_from_mean_phi(rng, mean_b, phi=phi)

        r = np.log(a.astype(float) + smoothing_s) - np.log(b.astype(float) + smoothing_s)

        linear = beta * v
        eta = calibrate_intercept_for_prevalence(linear, prevalence_target)
        p = sigmoid(linear + eta)
        y = rng.binomial(n=1, p=p, size=int(n)).astype(np.int32)

        base = pd.DataFrame({"A": a.astype(float), "B": b.astype(float)})

        d_cfg = distractors or {}
        p_e = int(d_cfg.get("p_e", 0))
        e_noise_scale_mult = float(d_cfg.get("loge_noise_scale_mult", 1.0))
        e_df = loge_distractors(
            rng,
            loge,
            p=p_e,
            noise_scale=sigma_loge * e_noise_scale_mult,
            prefix="e_d",
        )
        p_noise = int(d_cfg.get("p_noise", 0))
        noise_scale_mult = float(d_cfg.get("noise_scale_mult", 1.0))
        noise_df = iid_noise_distractors(
            rng,
            n=int(n),
            p=p_noise,
            noise_scale=sigma_loge * noise_scale_mult,
            prefix="n_d",
        )
        X_base = pd.concat([base, e_df, noise_df], axis=1)
        X_engineered = pd.DataFrame({"r_logcountdiff": r})
        return SplitData(
            X_base=X_base,
            X_engineered=X_engineered,
            y=y,
            latents={"logE": loge, "E": e, "V": v, "r_logcountdiff": r, "eta": np.full(int(n), eta)},
        )

    splits = {
        "train": one_split(int(n_train)),
        "valid": one_split(int(n_valid)),
        "test": one_split(int(n_test)),
    }

    return GeneratedDataset(
        dgp_name="count_exposure",
        params={
            "mu_loge": mu_loge,
            "sigma_loge": sigma_loge,
            "sigma_v": sigma_v,
            "beta": beta,
            "dispersion_phi": phi,
            "smoothing_s": smoothing_s,
            "distractors": distractors or {},
            "prevalence_target": float(prevalence_target),
        },
        intended_pairs=[("A", "B")],
        splits=splits,
    )
