from __future__ import annotations

from typing import Any

import numpy as np


def sample_from_spec(rng: np.random.Generator, spec: Any) -> Any:
    if isinstance(spec, dict):
        return {k: sample_from_spec(rng, v) for k, v in spec.items()}
    if isinstance(spec, list):
        if not spec:
            raise ValueError("Cannot sample from empty list spec")
        idx = int(rng.integers(low=0, high=len(spec)))
        return sample_from_spec(rng, spec[idx])
    return spec

