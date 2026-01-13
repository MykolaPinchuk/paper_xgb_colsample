from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from .common import GeneratedDataset, SplitData, calibrate_intercept_for_prevalence, sample_symmetric, sigmoid
from .distractors import iid_noise_distractors, u_distractors


InputRep = Literal["raw", "log"]


def generate(
    rng: np.random.Generator,
    *,
    n_train: int,
    n_valid: int,
    n_test: int,
    prevalence_target: float,
    input_representation: InputRep,
    dist_variant: str,
    sigma_v: float,
    kappa_sigmaU_over_sigmaV: float,
    sigma_eps: float,
    beta: float,
    distractors: dict[str, Any] | None = None,
) -> GeneratedDataset:
    sigma_v = float(sigma_v)
    sigma_u = float(kappa_sigmaU_over_sigmaV) * sigma_v
    sigma_eps = float(sigma_eps)
    beta = float(beta)
    if input_representation not in ("raw", "log"):
        raise ValueError("input_representation must be 'raw' or 'log'")

    intended_pairs = [("a", "b")] if input_representation == "raw" else [("log_a", "log_b")]

    def one_split(n: int, split_name: str) -> SplitData:
        u = sample_symmetric(rng, n, sigma=sigma_u, dist_variant=dist_variant)  # type: ignore[arg-type]
        v = sample_symmetric(rng, n, sigma=sigma_v, dist_variant=dist_variant)  # type: ignore[arg-type]
        eps_a = sample_symmetric(rng, n, sigma=sigma_eps, dist_variant=dist_variant)  # type: ignore[arg-type]
        eps_b = sample_symmetric(rng, n, sigma=sigma_eps, dist_variant=dist_variant)  # type: ignore[arg-type]

        log_a = u + v + eps_a
        log_b = u - v + eps_b
        r = log_a - log_b

        linear = beta * v
        eta = calibrate_intercept_for_prevalence(linear, prevalence_target)
        p = sigmoid(linear + eta)
        y = rng.binomial(n=1, p=p, size=n).astype(np.int32)

        if input_representation == "raw":
            clip = 20.0
            a = np.exp(np.clip(log_a, -clip, clip))
            b = np.exp(np.clip(log_b, -clip, clip))
            base = pd.DataFrame({"a": a, "b": b})
        else:
            base = pd.DataFrame({"log_a": log_a, "log_b": log_b})

        d_cfg = distractors or {}
        p_u = int(d_cfg.get("p_u", 0))
        u_noise_scale_mult = float(d_cfg.get("u_noise_scale_mult", 1.0))
        u_noise_scale = sigma_u * u_noise_scale_mult
        u_df = u_distractors(rng, u, p=p_u, noise_scale=u_noise_scale, prefix="u_d")

        p_noise = int(d_cfg.get("p_noise", 0))
        noise_scale_mult = float(d_cfg.get("noise_scale_mult", 1.0))
        noise_scale = sigma_u * noise_scale_mult
        noise_df = iid_noise_distractors(rng, n=n, p=p_noise, noise_scale=noise_scale, prefix="n_d")

        X_base = pd.concat([base, u_df, noise_df], axis=1)
        X_engineered = pd.DataFrame({"r_logratio": r})
        return SplitData(
            X_base=X_base,
            X_engineered=X_engineered,
            y=y,
            latents={"U": u, "V": v, "log_a": log_a, "log_b": log_b, "r_logratio": r, "eta": np.full(n, eta)},
        )

    splits = {
        "train": one_split(int(n_train), "train"),
        "valid": one_split(int(n_valid), "valid"),
        "test": one_split(int(n_test), "test"),
    }

    return GeneratedDataset(
        dgp_name="continuous_logratio",
        params={
            "input_representation": input_representation,
            "dist_variant": dist_variant,
            "sigma_v": sigma_v,
            "kappa_sigmaU_over_sigmaV": float(kappa_sigmaU_over_sigmaV),
            "sigma_eps": sigma_eps,
            "beta": beta,
            "distractors": distractors or {},
            "prevalence_target": float(prevalence_target),
        },
        intended_pairs=intended_pairs,
        splits=splits,
    )
