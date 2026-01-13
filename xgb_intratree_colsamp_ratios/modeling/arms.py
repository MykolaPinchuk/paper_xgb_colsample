from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    feature_set: str  # F0|F1
    colsamp_arm: str  # C0|C1|C2|C3
    s: float
    colsample_bylevel: float
    colsample_bynode: float
    max_depth: int
    regime: str
    n_estimators: int
    early_stopping: bool
    early_stopping_rounds: int | None


def build_arms(
    *,
    depths: list[int],
    colsamp_s: list[float],
    regimes: list[dict],
    allowed_colsamp_arms: list[str] | None = None,
) -> list[Arm]:
    arms: list[Arm] = []
    allowed = set(allowed_colsamp_arms) if allowed_colsamp_arms is not None else None
    for regime in regimes:
        name = str(regime["name"])
        n_estimators = int(regime["n_estimators"])
        early_stopping = bool(regime["early_stopping"])
        early_stopping_rounds = regime.get("early_stopping_rounds")
        for max_depth in depths:
            for s in colsamp_s:
                s = float(s)
                if s == 1.0:
                    colsamp_defs = [("C0", 1.0, 1.0)]
                else:
                    colsamp_defs = [
                        ("C1", s, 1.0),
                        ("C2", 1.0, s),
                        ("C3", s, s),
                    ]
                for feature_set in ("F0", "F1"):
                    for colsamp_arm, bylevel, bynode in colsamp_defs:
                        if allowed is not None and colsamp_arm not in allowed:
                            continue
                        arms.append(
                            Arm(
                                feature_set=feature_set,
                                colsamp_arm=colsamp_arm,
                                s=s,
                                colsample_bylevel=float(bylevel),
                                colsample_bynode=float(bynode),
                                max_depth=int(max_depth),
                                regime=name,
                                n_estimators=n_estimators,
                                early_stopping=early_stopping,
                                early_stopping_rounds=(
                                    int(early_stopping_rounds) if early_stopping_rounds is not None else None
                                ),
                            )
                        )
    return arms
