from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping at {p}, got {type(data).__name__}")
    return data


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    outputs_dir: Path
    results_csv: Path
    config_yaml: Path


def make_run_paths(run_id: str) -> RunPaths:
    outputs_dir = Path("outputs") / "results" / run_id
    return RunPaths(
        run_id=run_id,
        outputs_dir=outputs_dir,
        results_csv=outputs_dir / "results.csv",
        config_yaml=outputs_dir / "config.yaml",
    )

