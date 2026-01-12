from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _plot_lines(df: pd.DataFrame, x_col: str, y_col: str, line_col: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(6.5, 4.0))
    for line_value, sub in df.groupby(line_col):
        sub = sub.sort_values(x_col)
        plt.plot(sub[x_col], sub[y_col], marker="o", label=str(line_value))
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True, help="Path to summary.csv (from aggregate).")
    ap.add_argument("--out-dir", default=None, help="Directory for plots (default: <summary parent>/figs).")
    args = ap.parse_args()

    summary_path = Path(args.summary_csv)
    df = pd.read_csv(summary_path)
    if df.empty:
        print("summary.csv is empty; nothing to plot.")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_metrics = [
        "delta_test_prauc",
        "delta_test_rocauc",
        "delta_test_latent_corr",
        "delta_cooc_mean",
        "delta_cooc_path_mean",
        "test_prauc",
        "test_rocauc",
        "test_latent_corr",
        "cooc_mean",
    ]
    metric_cols = [m for m in candidate_metrics if f"{m}_mean" in df.columns]
    if not metric_cols:
        print("No recognized metric columns found in summary.csv.")
        return 1

    group_cols = ["dgp_name", "feature_set", "colsamp_arm", "s"]
    for metric in metric_cols:
        metric_col = f"{metric}_mean"
        df_metric = df[group_cols + [metric_col]].dropna()
        if df_metric.empty:
            continue
        for (dgp_name, feature_set), sub in df_metric.groupby(["dgp_name", "feature_set"]):
            title = f"{dgp_name} | {feature_set} | {metric}"
            out_path = out_dir / f"{dgp_name}__{feature_set}__{metric}.png"
            _plot_lines(sub, x_col="s", y_col=metric_col, line_col="colsamp_arm", title=title, out_path=out_path)

    report_path = out_dir / "report.md"
    lines = [
        "# Plot report",
        "",
        f"- generated_utc: {datetime.now(timezone.utc).isoformat()}",
        f"- summary_csv: {summary_path}",
        f"- out_dir: {out_dir}",
        f"- metrics: {metric_cols}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
