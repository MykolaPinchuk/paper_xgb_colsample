from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _classify_effect(ci_lo: float, ci_hi: float) -> str:
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return "n/a"
    if ci_hi < 0:
        return "hurts"
    if ci_lo > 0:
        return "helps"
    return "inconclusive"


def _mean_ci(series: pd.Series) -> tuple[float, float, float, int, float]:
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = int(x.shape[0])
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0, float("nan")
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    ci_lo = mean - 1.96 * se
    ci_hi = mean + 1.96 * se
    frac_neg = float((x < 0).mean())
    return mean, ci_lo, ci_hi, n, frac_neg


def _format_float(x: float, digits: int = 4) -> str:
    if np.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def _write_effect_report(results_with_deltas: pd.DataFrame, out_path: Path) -> None:
    df = results_with_deltas.copy()
    for col in ["dgp_name", "feature_set", "colsamp_arm", "regime"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Focus on arms with a defined delta (baseline present)
    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    if not delta_cols:
        raise ValueError("No delta_* columns found; run aggregate first to produce results_with_deltas.csv")
    df = df.dropna(subset=delta_cols, how="all")

    # Prefer PR-AUC (primary), also report ROC-AUC, latent alignment, and co-occurrence (mechanism).
    delta_prauc = "delta_test_prauc" if "delta_test_prauc" in df.columns else None
    delta_rocauc = "delta_test_rocauc" if "delta_test_rocauc" in df.columns else None
    delta_latent = "delta_test_latent_corr" if "delta_test_latent_corr" in df.columns else None
    delta_cooc = "delta_cooc_mean" if "delta_cooc_mean" in df.columns else None
    delta_cooc_path = "delta_cooc_path_mean" if "delta_cooc_path_mean" in df.columns else None
    delta_cooc_allfeat_tree = "delta_cooc_allfeat_tree" if "delta_cooc_allfeat_tree" in df.columns else None
    delta_cooc_allfeat_path = "delta_cooc_allfeat_path" if "delta_cooc_allfeat_path" in df.columns else None
    if (
        delta_prauc is None
        and delta_rocauc is None
        and delta_latent is None
        and delta_cooc is None
        and delta_cooc_path is None
        and delta_cooc_allfeat_tree is None
        and delta_cooc_allfeat_path is None
    ):
        raise ValueError(
            "No recognized delta metrics present (expected delta_test_prauc, delta_test_rocauc, delta_cooc_mean, delta_cooc_path_mean, delta_cooc_allfeat_*)"
        )
    metrics = [
        c
        for c in [
            delta_prauc,
            delta_rocauc,
            delta_latent,
            delta_cooc,
            delta_cooc_path,
            delta_cooc_allfeat_tree,
            delta_cooc_allfeat_path,
        ]
        if c is not None
    ]

    lines: list[str] = []
    lines.append("# Effect report (C? vs C0 deltas)")
    lines.append("")
    lines.append(f"- generated_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- rows: {len(df)}")
    lines.append("")
    lines.append("## Legend")
    lines.append("- `F0`: primitives only")
    lines.append("- `F1`: primitives + engineered ratio feature")
    lines.append("- `C0`: bylevel=1.0, bynode=1.0 (baseline)")
    lines.append("- Deltas are computed within (run_id, scenario_id, rep_id, feature_set, depth, regime) relative to `C0`.")
    lines.append("- `Δlatent corr`: change in Pearson correlation between model logit(pred) and the true latent signal (V or V_sum).")
    if delta_cooc_path is not None:
        lines.append("- `Δcooc path mean`: change in mean leaf-path co-usage fraction (weighted by XGB JSON `cover`).")
    if delta_cooc_allfeat_path is not None:
        lines.append("- `Δcooc allfeat path`: change in mean leaf-path fraction containing all intended primitive features.")
    lines.append("")

    group_cols = [c for c in ["dgp_name", "feature_set", "colsamp_arm", "s"] if c in df.columns]
    if "colsamp_arm" in group_cols:
        # Exclude baseline rows from "effect" summaries (their delta is 0 by construction when present).
        df_eff = df[df["colsamp_arm"] != "C0"].copy()
    else:
        df_eff = df.copy()

    if df_eff.empty:
        lines.append("## Summary")
        lines.append("- No non-baseline arms found; nothing to summarize.")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    # Aggregate effects.
    summary_rows = []
    for keys, sub in df_eff.groupby(group_cols):
        key_dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row = dict(key_dict)
        for metric in metrics:
            mean, ci_lo, ci_hi, n, frac_neg = _mean_ci(sub[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lo"] = ci_lo
            row[f"{metric}_ci_hi"] = ci_hi
            row[f"{metric}_n"] = n
            row[f"{metric}_frac_neg"] = frac_neg
            row[f"{metric}_effect"] = _classify_effect(ci_lo, ci_hi)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(group_cols).reset_index(drop=True)

    # Write a compact markdown table per DGP/feature_set.
    for (dgp_name, feature_set), sub in summary.groupby(["dgp_name", "feature_set"]):
        lines.append(f"## {dgp_name} / {feature_set}")
        colsamp_arms = sorted(sub["colsamp_arm"].unique()) if "colsamp_arm" in sub.columns else []
        lines.append(f"- colsamp_arms: {colsamp_arms}")
        lines.append("")
        has_cooc_path = delta_cooc_path is not None
        has_allfeat_path = delta_cooc_allfeat_path is not None

        if not has_cooc_path and not has_allfeat_path:
            lines.append(
                "| colsamp_arm | s | ΔPR-AUC mean (95% CI) | PR effect | ΔROC-AUC mean (95% CI) | ROC effect | Δlatent corr mean (95% CI) | latent effect | Δcooc mean (95% CI) | cooc effect | n |"
            )
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        else:
            extra_headers = ""
            extra_dividers = ""
            if has_cooc_path:
                extra_headers += " Δcooc path mean (95% CI) | cooc path effect |"
                extra_dividers += "---:|---:|"
            if has_allfeat_path:
                extra_headers += " Δcooc allfeat path (95% CI) | allfeat effect |"
                extra_dividers += "---:|---:|"
            lines.append(
                "| colsamp_arm | s | ΔPR-AUC mean (95% CI) | PR effect | ΔROC-AUC mean (95% CI) | ROC effect | Δlatent corr mean (95% CI) | latent effect | Δcooc mean (95% CI) | cooc effect |"
                f"{extra_headers} n |"
            )
            lines.append(
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
                f"{extra_dividers}---:|"
            )
        for _, r in sub.iterrows():
            pr_mean = _format_float(r.get("delta_test_prauc_mean", float("nan")))
            pr_lo = _format_float(r.get("delta_test_prauc_ci_lo", float("nan")))
            pr_hi = _format_float(r.get("delta_test_prauc_ci_hi", float("nan")))
            pr_eff = str(r.get("delta_test_prauc_effect", "n/a"))

            roc_mean = _format_float(r.get("delta_test_rocauc_mean", float("nan")))
            roc_lo = _format_float(r.get("delta_test_rocauc_ci_lo", float("nan")))
            roc_hi = _format_float(r.get("delta_test_rocauc_ci_hi", float("nan")))
            roc_eff = str(r.get("delta_test_rocauc_effect", "n/a"))

            lat_mean = _format_float(r.get("delta_test_latent_corr_mean", float("nan")))
            lat_lo = _format_float(r.get("delta_test_latent_corr_ci_lo", float("nan")))
            lat_hi = _format_float(r.get("delta_test_latent_corr_ci_hi", float("nan")))
            lat_eff = str(r.get("delta_test_latent_corr_effect", "n/a"))

            co_mean = _format_float(r.get("delta_cooc_mean_mean", float("nan")))
            co_lo = _format_float(r.get("delta_cooc_mean_ci_lo", float("nan")))
            co_hi = _format_float(r.get("delta_cooc_mean_ci_hi", float("nan")))
            co_eff = str(r.get("delta_cooc_mean_effect", "n/a"))

            n = int(
                max(
                    int(r.get("delta_test_prauc_n", 0)),
                    int(r.get("delta_test_rocauc_n", 0)),
                    int(r.get("delta_test_latent_corr_n", 0)),
                    int(r.get("delta_cooc_mean_n", 0)),
                    int(r.get("delta_cooc_path_mean_n", 0)),
                    int(r.get("delta_cooc_allfeat_tree_n", 0)),
                    int(r.get("delta_cooc_allfeat_path_n", 0)),
                )
            )

            if not has_cooc_path and not has_allfeat_path:
                lines.append(
                    f"| {r.get('colsamp_arm')} | {r.get('s')} | {pr_mean} [{pr_lo}, {pr_hi}] | {pr_eff} | {roc_mean} [{roc_lo}, {roc_hi}] | {roc_eff} | {lat_mean} [{lat_lo}, {lat_hi}] | {lat_eff} | {co_mean} [{co_lo}, {co_hi}] | {co_eff} | {n} |"
                )
            else:
                extras = ""
                if has_cooc_path:
                    cp_mean = _format_float(r.get("delta_cooc_path_mean_mean", float("nan")))
                    cp_lo = _format_float(r.get("delta_cooc_path_mean_ci_lo", float("nan")))
                    cp_hi = _format_float(r.get("delta_cooc_path_mean_ci_hi", float("nan")))
                    cp_eff = str(r.get("delta_cooc_path_mean_effect", "n/a"))
                    extras += f" {cp_mean} [{cp_lo}, {cp_hi}] | {cp_eff} |"
                if has_allfeat_path:
                    afp_mean = _format_float(r.get("delta_cooc_allfeat_path_mean", float("nan")))
                    afp_lo = _format_float(r.get("delta_cooc_allfeat_path_ci_lo", float("nan")))
                    afp_hi = _format_float(r.get("delta_cooc_allfeat_path_ci_hi", float("nan")))
                    afp_eff = str(r.get("delta_cooc_allfeat_path_effect", "n/a"))
                    extras += f" {afp_mean} [{afp_lo}, {afp_hi}] | {afp_eff} |"

                lines.append(
                    f"| {r.get('colsamp_arm')} | {r.get('s')} | {pr_mean} [{pr_lo}, {pr_hi}] | {pr_eff} | {roc_mean} [{roc_lo}, {roc_hi}] | {roc_eff} | {lat_mean} [{lat_lo}, {lat_hi}] | {lat_eff} | {co_mean} [{co_lo}, {co_hi}] | {co_eff} |{extras} {n} |"
                )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-with-deltas", required=True, help="Path to results_with_deltas.csv (from aggregate).")
    ap.add_argument("--out", default=None, help="Output markdown path (default: <parent>/effect_report.md).")
    args = ap.parse_args()

    in_path = Path(args.results_with_deltas)
    df = pd.read_csv(in_path)
    out_path = Path(args.out) if args.out else in_path.parent / "effect_report.md"
    _write_effect_report(df, out_path)
    print(f"Wrote effect report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
