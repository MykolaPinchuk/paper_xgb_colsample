from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    config: str
    run_id: str
    agg_out_dir: str


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    jobs = [
        Job(
            config="configs/phase0_pipeline_dgpA_strongratio_C1C2C3_s09_s08_s06_s04.yaml",
            run_id="phase0_20251220_223214",
            agg_out_dir="outputs/aggregate/agg_20251220_223214_only",
        ),
        Job(
            config="configs/phase0_pipeline_dgpB_strongratio_C1C2C3_s09_s08_s06_s04.yaml",
            run_id="phase0_20251220_230029",
            agg_out_dir="outputs/aggregate/agg_20251220_230029_only",
        ),
        Job(
            config="configs/phase0_pipeline_dgpA_boundary_strong.yaml",
            run_id="phase0_20251220_221556",
            agg_out_dir="outputs/aggregate/agg_20251220_221556_only",
        ),
        Job(
            config="configs/phase0_pipeline_dgpA_boundary_mid.yaml",
            run_id="phase0_20251220_221807",
            agg_out_dir="outputs/aggregate/agg_20251220_221807_only",
        ),
        Job(
            config="configs/phase0_pipeline_dgpA_boundary_weak.yaml",
            run_id="phase0_20251220_221953",
            agg_out_dir="outputs/aggregate/agg_20251220_221953_only",
        ),
    ]

    for j in jobs:
        _run(
            [
                "python",
                "-m",
                "xgb_intratree_colsamp_ratios.runner.run_phase0",
                "--config",
                str(repo_root / j.config),
                "--run-id",
                j.run_id,
            ]
        )
        _run(
            [
                "python",
                "-m",
                "xgb_intratree_colsamp_ratios.runner.aggregate",
                "--runs-dir",
                str(repo_root / "outputs/results"),
                "--run-ids",
                j.run_id,
                "--out-dir",
                str(repo_root / j.agg_out_dir),
            ]
        )

    print("Done. Next: python scripts/build_paper_assets.py && (cd paper && pdflatex main_v3.tex && pdflatex main_v3.tex)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
