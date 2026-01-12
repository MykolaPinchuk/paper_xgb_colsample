from __future__ import annotations

import numpy as np
import pandas as pd


def u_distractors(
    rng: np.random.Generator,
    u: np.ndarray,
    *,
    p: int,
    noise_scale: float,
    prefix: str = "u_d",
) -> pd.DataFrame:
    n = int(u.shape[0])
    p_int = int(p)
    if p_int <= 0:
        return pd.DataFrame(index=np.arange(n))
    noise = rng.normal(loc=0.0, scale=float(noise_scale), size=(n, p_int))
    x = u.reshape(-1, 1) + noise
    cols = [f"{prefix}{j}" for j in range(p_int)]
    return pd.DataFrame(x, columns=cols)


def loge_distractors(
    rng: np.random.Generator,
    loge: np.ndarray,
    *,
    p: int,
    noise_scale: float,
    prefix: str = "e_d",
) -> pd.DataFrame:
    n = int(loge.shape[0])
    p_int = int(p)
    if p_int <= 0:
        return pd.DataFrame(index=np.arange(n))
    noise = rng.normal(loc=0.0, scale=float(noise_scale), size=(n, p_int))
    x = loge.reshape(-1, 1) + noise
    cols = [f"{prefix}{j}" for j in range(p_int)]
    return pd.DataFrame(x, columns=cols)


def iid_noise_distractors(
    rng: np.random.Generator,
    *,
    n: int,
    p: int,
    noise_scale: float,
    prefix: str = "n_d",
) -> pd.DataFrame:
    n_int = int(n)
    p_int = int(p)
    if p_int <= 0:
        return pd.DataFrame(index=np.arange(n_int))
    x = rng.normal(loc=0.0, scale=float(noise_scale), size=(n_int, p_int))
    cols = [f"{prefix}{j}" for j in range(p_int)]
    return pd.DataFrame(x, columns=cols)
