from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _find_run_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted([p for p in runs_dir.iterdir() if p.is_dir()])


def _parse_run_ids_arg(run_ids: str | None) -> set[str] | None:
    if run_ids is None:
        return None
    parts = [p.strip() for p in str(run_ids).split(",")]
    out = {p for p in parts if p}
    return out or None


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _add_run_id(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    out = df.copy()
    out["run_id"] = run_id
    return out


def _compute_deltas(results: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["run_id", "scenario_id", "rep_id", "feature_set", "max_depth", "regime"]
    metric_cols = [
        c
        for c in [
            "valid_prauc",
            "valid_rocauc",
            "valid_latent_corr",
            "test_prauc",
            "test_rocauc",
            "test_latent_corr",
            "cooc_mean",
            "cooc_path_mean",
            "cooc_allfeat_tree",
            "cooc_allfeat_path",
        ]
        if c in results.columns
    ]

    baseline = results[results["colsamp_arm"] == "C0"][key_cols + metric_cols].copy()
    baseline = baseline.rename(columns={c: f"{c}_base" for c in metric_cols})
    merged = results.merge(baseline, on=key_cols, how="left")
    for c in metric_cols:
        merged[f"delta_{c}"] = merged[c] - merged[f"{c}_base"]
    return merged


def _aggregate_summary(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.to_flat_index()]
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="outputs/results", help="Directory containing run subdirs.")
    ap.add_argument("--out-dir", default=None, help="Directory to write aggregate outputs.")
    ap.add_argument(
        "--run-ids",
        default=None,
        help="Optional comma-separated subset of run directory names to include (e.g. phase0_...,...).",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / "aggregate" / f"agg_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    include_run_ids = _parse_run_ids_arg(args.run_ids)
    run_dirs = _find_run_dirs(runs_dir)
    if include_run_ids is not None:
        run_dirs = [p for p in run_dirs if p.name in include_run_ids]
    if not run_dirs:
        print(f"No run directories found under {runs_dir}")
        return 1

    results_list: list[pd.DataFrame] = []
    acceptance_list: list[pd.DataFrame] = []

    for run_dir in run_dirs:
        run_id = run_dir.name
        res = _read_csv_if_exists(run_dir / "results.csv")
        if res is not None and not res.empty:
            results_list.append(_add_run_id(res, run_id))
        acc = _read_csv_if_exists(run_dir / "acceptance.csv")
        if acc is not None and not acc.empty:
            acceptance_list.append(_add_run_id(acc, run_id))

    results = pd.concat(results_list, ignore_index=True) if results_list else pd.DataFrame()
    acceptance = pd.concat(acceptance_list, ignore_index=True) if acceptance_list else pd.DataFrame()

    if not results.empty:
        results_with_delta = _compute_deltas(results)
        results_with_delta.to_csv(out_dir / "results_with_deltas.csv", index=False)

        group_cols = ["dgp_name", "feature_set", "colsamp_arm", "s", "max_depth", "regime"]
        metric_cols = [c for c in results_with_delta.columns if c.startswith("delta_") or c in {"test_prauc", "test_rocauc", "cooc_mean"}]
        summary = _aggregate_summary(results_with_delta, group_cols, metric_cols)
        summary.to_csv(out_dir / "summary.csv", index=False)

    if not acceptance.empty:
        acc_group_cols = ["run_id", "dgp_name"]
        acc_metrics = ["accepted"]
        acc_summary = acceptance.groupby(acc_group_cols)[acc_metrics].agg(["mean", "count"]).reset_index()
        acc_summary.columns = ["_".join([c for c in col if c]) for col in acc_summary.columns.to_flat_index()]
        acc_summary.to_csv(out_dir / "acceptance_summary.csv", index=False)

    report_lines: list[str] = []
    report_lines.append("# Aggregate report")
    report_lines.append("")
    report_lines.append(f"- runs_dir: {runs_dir}")
    report_lines.append(f"- out_dir: {out_dir}")
    report_lines.append(f"- run_ids_filter: {sorted(include_run_ids) if include_run_ids is not None else None}")
    report_lines.append(f"- runs_found: {len(run_dirs)}")
    report_lines.append(f"- results_rows: {len(results)}")
    report_lines.append(f"- acceptance_rows: {len(acceptance)}")
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote aggregate outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
