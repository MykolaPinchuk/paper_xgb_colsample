from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from .common import GeneratedDataset, SplitData, calibrate_intercept_for_prevalence, sample_symmetric, sigmoid
from .distractors import u_distractors


InputRep = Literal["raw", "log"]


def _sample_matrix(
    rng: np.random.Generator, n: int, k: int, sigma: float, dist_variant: str
) -> np.ndarray:
    flat = sample_symmetric(rng, size=int(n) * int(k), sigma=float(sigma), dist_variant=dist_variant)  # type: ignore[arg-type]
    return flat.reshape(int(n), int(k))


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
    k_components: int,
    distractors: dict[str, Any] | None = None,
) -> GeneratedDataset:
    sigma_v = float(sigma_v)
    sigma_u = float(kappa_sigmaU_over_sigmaV) * sigma_v
    sigma_eps = float(sigma_eps)
    beta = float(beta)
    k = int(k_components)
    if k <= 0:
        raise ValueError("k_components must be >= 1")
    if input_representation not in ("raw", "log"):
        raise ValueError("input_representation must be 'raw' or 'log'")

    intended_pairs = [(f"a{i+1}", f"b{i+1}") for i in range(k)]
    if input_representation == "log":
        intended_pairs = [(f"log_a{i+1}", f"log_b{i+1}") for i in range(k)]

    def one_split(n: int) -> SplitData:
        u = sample_symmetric(rng, n, sigma=sigma_u, dist_variant=dist_variant)  # type: ignore[arg-type]
        v = _sample_matrix(rng, n, k, sigma=sigma_v, dist_variant=dist_variant)
        eps_a = _sample_matrix(rng, n, k, sigma=sigma_eps, dist_variant=dist_variant)
        eps_b = _sample_matrix(rng, n, k, sigma=sigma_eps, dist_variant=dist_variant)

        log_a = u.reshape(-1, 1) + v + eps_a
        log_b = u.reshape(-1, 1) - v + eps_b
        r = (log_a - log_b).sum(axis=1)
        v_sum = v.sum(axis=1)

        linear = beta * v_sum
        eta = calibrate_intercept_for_prevalence(linear, prevalence_target)
        p = sigmoid(linear + eta)
        y = rng.binomial(n=1, p=p, size=int(n)).astype(np.int32)

        cols = []
        data = []
        if input_representation == "raw":
            clip = 20.0
            a = np.exp(np.clip(log_a, -clip, clip))
            b = np.exp(np.clip(log_b, -clip, clip))
            for i in range(k):
                cols.extend([f"a{i+1}", f"b{i+1}"])
                data.append(a[:, i])
                data.append(b[:, i])
        else:
            for i in range(k):
                cols.extend([f"log_a{i+1}", f"log_b{i+1}"])
                data.append(log_a[:, i])
                data.append(log_b[:, i])
        base = pd.DataFrame(np.column_stack(data), columns=cols)

        d_cfg = distractors or {}
        p_u = int(d_cfg.get("p_u", 0))
        u_noise_scale_mult = float(d_cfg.get("u_noise_scale_mult", 1.0))
        u_noise_scale = sigma_u * u_noise_scale_mult
        u_df = u_distractors(rng, u, p=p_u, noise_scale=u_noise_scale, prefix="u_d")

        X_base = pd.concat([base, u_df], axis=1)
        X_engineered = pd.DataFrame({"r_logratio_sum": r})
        return SplitData(
            X_base=X_base,
            X_engineered=X_engineered,
            y=y,
            latents={
                "U": u,
                "V_sum": v_sum,
                "r_logratio_sum": r,
                "eta": np.full(int(n), eta),
            },
        )

    splits = {
        "train": one_split(int(n_train)),
        "valid": one_split(int(n_valid)),
        "test": one_split(int(n_test)),
    }

    return GeneratedDataset(
        dgp_name="multicomponent",
        params={
            "input_representation": input_representation,
            "dist_variant": dist_variant,
            "sigma_v": sigma_v,
            "kappa_sigmaU_over_sigmaV": float(kappa_sigmaU_over_sigmaV),
            "sigma_eps": sigma_eps,
            "beta": beta,
            "k_components": k,
            "distractors": distractors or {},
            "prevalence_target": float(prevalence_target),
        },
        intended_pairs=intended_pairs,
        splits=splits,
    )

