from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _mean_ci(x: np.ndarray) -> tuple[float, float, float, int]:
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(x.mean())
    if n == 1:
        return mean, mean, mean, 1
    std = float(x.std(ddof=1))
    se = std / np.sqrt(n)
    z = 1.96
    return mean, mean - z * se, mean + z * se, n


def _fmt_ci(mean: float, lo: float, hi: float, digits: int = 4) -> str:
    if not (np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi)):
        return ""
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _load_results_with_deltas(path: Path, *, dgp_name: str, feature_set: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[(df["dgp_name"] == dgp_name) & (df["feature_set"] == feature_set)].copy()


def _dual_axis_summary(
    results_with_deltas_csv: Path,
    *,
    dgp_name: str,
    feature_set: str,
    delta_metric: str,
    base_metric: str,
    right_metric: str,
) -> pd.DataFrame:
    df = _load_results_with_deltas(results_with_deltas_csv, dgp_name=dgp_name, feature_set=feature_set)
    df = df[df["colsamp_arm"].isin(["C1", "C2", "C3"])].copy()

    rows = []
    for (arm, s), sub in df.groupby(["colsamp_arm", "s"], dropna=False):
        delta = sub[delta_metric].to_numpy(dtype=float, copy=False)
        base = sub[base_metric].to_numpy(dtype=float, copy=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = delta / base
        rel_mean, _, _, n = _mean_ci(rel)

        right = sub[right_metric].to_numpy(dtype=float, copy=False)
        right_mean, _, _, _ = _mean_ci(right)

        rows.append(
            {
                "colsamp_arm": str(arm),
                "s": float(s),
                "rel_delta_pct": 100.0 * rel_mean,
                "right_mean": right_mean,
                "n": int(n),
            }
        )
    return pd.DataFrame(rows).sort_values(["colsamp_arm", "s"])


def _plot_dual_axis_3panel(
    df: pd.DataFrame,
    *,
    title: str,
    left_label: str,
    right_label: str,
    out_path: Path,
) -> None:
    arms = ["C1", "C2", "C3"]
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12.0, 3.6), sharex=True)
    color_left = "#1f77b4"
    color_right = "#ff7f0e"

    for ax, arm in zip(axes, arms):
        sub = df[df["colsamp_arm"] == arm].sort_values("s")
        ax.plot(sub["s"], sub["rel_delta_pct"], marker="o", color=color_left, linewidth=2, label=left_label)
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        ax.set_xlabel("s")
        ax.set_ylabel(left_label)
        ax.set_title(f"{arm} Arm")
        ax.grid(True, alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(sub["s"], sub["right_mean"], marker="s", color=color_right, linewidth=2, label=right_label)
        ax2.set_ylabel(right_label, labelpad=8)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, frameon=True)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_s_sweep_grid(
    summary_csv: Path,
    *,
    out_path: Path,
    title_prefix: str,
) -> None:
    df = pd.read_csv(summary_csv)
    df = df[df["colsamp_arm"].isin(["C1", "C2", "C3"])].copy()
    df = df[df["s"].isin([0.4, 0.6, 0.8, 0.9])].copy()

    metrics = [
        ("delta_test_prauc_mean", "ΔPR-AUC (test)"),
        ("delta_test_rocauc_mean", "ΔROC-AUC (test)"),
        ("delta_test_latent_corr_mean", "Δlatent corr (test)"),
        ("delta_cooc_mean_mean", "Δcooc_mean"),
    ]
    for col, _ in metrics:
        if col not in df.columns:
            raise ValueError(f"{summary_csv}: missing column {col}")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12.0, 6.8), sharex=True)
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, metrics):
        for arm, sub in df.groupby("colsamp_arm"):
            sub = sub.sort_values("s")
            ax.plot(sub["s"], sub[col], marker="o", linewidth=2, label=arm)
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        ax.set_title(f"{title_prefix} | {title}")
        ax.set_xlabel("s")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@dataclass(frozen=True)
class BoundaryPoint:
    label: str
    feature_set: str
    s: float
    mean: float
    lo: float
    hi: float


def _load_boundary_points(results_with_deltas_csv: Path, *, label: str) -> list[BoundaryPoint]:
    df = pd.read_csv(results_with_deltas_csv)
    df = df[(df["dgp_name"] == "continuous_logratio") & (df["colsamp_arm"] == "C3")].copy()
    out: list[BoundaryPoint] = []
    for (feature_set, s), sub in df.groupby(["feature_set", "s"], dropna=False):
        mean, lo, hi, _ = _mean_ci(sub["delta_test_prauc"].to_numpy(dtype=float, copy=False))
        out.append(BoundaryPoint(label=label, feature_set=str(feature_set), s=float(s), mean=mean, lo=lo, hi=hi))
    return out


def _plot_boundary_delta_prauc(
    boundary_points: list[BoundaryPoint],
    *,
    out_path: Path,
) -> None:
    df = pd.DataFrame([p.__dict__ for p in boundary_points])
    order = {"strong": 0, "mid": 1, "weak": 2}
    df["x"] = df["label"].map(order)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10.5, 3.6), sharey=True)
    for ax, feature_set in zip(axes, ["F0", "F1"]):
        sub_fs = df[df["feature_set"] == feature_set].copy()
        for s_val, sub in sub_fs.groupby("s"):
            sub = sub.sort_values("x")
            ax.errorbar(
                sub["x"],
                sub["mean"],
                yerr=[sub["mean"] - sub["lo"], sub["hi"] - sub["mean"]],
                marker="o",
                linewidth=2,
                label=f"s={s_val:g}",
            )
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        ax.set_title(f"{feature_set}")
        ax.set_xticks([0, 1, 2], ["strong", "mid", "weak"])
        ax.set_xlabel("ratio necessity setting")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("ΔPR-AUC (test)")
    axes[0].legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _build_tables(
    *,
    out_dir: Path,
    sweep_a_results_with_deltas: Path,
    sweep_b_results_with_deltas: Path,
) -> None:
    def baseline_rows(results_with_deltas_csv: Path, dgp_letter: str) -> list[dict]:
        df = pd.read_csv(results_with_deltas_csv)
        df = df[(df["colsamp_arm"] == "C0") & (df["s"] == 1.0)].copy()
        rows = []
        for feature_set, sub in df.groupby("feature_set", dropna=False):
            pr_mean = float(sub["test_prauc"].mean())
            roc_mean = float(sub["test_rocauc"].mean())
            lat_mean = float(sub["test_latent_corr"].mean())
            cooc_mean = float(sub["cooc_mean"].mean())
            n = int(sub.shape[0])
            rows.append(
                {
                    "DGP": dgp_letter,
                    "Feature set": str(feature_set),
                    "PR-AUC": pr_mean,
                    "ROC-AUC": roc_mean,
                    "latent_corr": lat_mean,
                    "cooc_mean": cooc_mean,
                    "n": n,
                }
            )
        return rows

    baseline = baseline_rows(sweep_a_results_with_deltas, "A") + baseline_rows(sweep_b_results_with_deltas, "B")
    bdf = pd.DataFrame(baseline).sort_values(["DGP", "Feature set"])

    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_tex = out_dir / "tab_app_v3_baseline.tex"
    baseline_tex.write_text(
        "\n".join(
            [
                "\\begin{table}[t]",
                "\\centering",
                "\\small",
                "\\setlength{\\tabcolsep}{4pt}",
                "\\renewcommand{\\arraystretch}{1.1}",
                "\\begin{tabular}{llrrrrr}",
                "\\toprule",
                "DGP & Feature set & PR-AUC & ROC-AUC & \\texttt{latent\\_corr} & \\texttt{cooc\\_mean} & $n$ \\\\",
                "\\midrule",
                *[
                    f"{r['DGP']} & {r['Feature set']} & {r['PR-AUC']:.4f} & {r['ROC-AUC']:.4f} & {r['latent_corr']:.4f} & {r['cooc_mean']:.4f} & {int(r['n'])} \\\\"
                    for _, r in bdf.iterrows()
                ],
                "\\bottomrule",
                "\\end{tabular}",
                "\\caption{Baseline (C0, $s=1.0$) test metrics used to compute relative deltas in Tables~\\ref{tab:app_v3_f0}--\\ref{tab:app_v3_f1}. DGP A is continuous log ratio; DGP B is count plus exposure.}",
                "\\label{tab:app_v3_baseline}",
                "\\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    def full_s_table(results_with_deltas_csv: Path, *, dgp_letter: str, feature_set: str) -> pd.DataFrame:
        df = pd.read_csv(results_with_deltas_csv)
        df = df[(df["feature_set"] == feature_set) & (df["colsamp_arm"].isin(["C1", "C2", "C3"]))].copy()
        rows = []
        for (arm, s), sub in df.groupby(["colsamp_arm", "s"], dropna=False):
            pr = sub["delta_test_prauc"].to_numpy(dtype=float, copy=False)
            roc = sub["delta_test_rocauc"].to_numpy(dtype=float, copy=False)
            cooc = sub["delta_cooc_mean"].to_numpy(dtype=float, copy=False)
            lat = sub["delta_test_latent_corr"].to_numpy(dtype=float, copy=False)
            pr_mean, pr_lo, pr_hi, n = _mean_ci(pr)
            roc_mean, roc_lo, roc_hi, _ = _mean_ci(roc)
            cooc_mean, _, _, _ = _mean_ci(cooc)
            lat_mean, _, _, _ = _mean_ci(lat)

            pr_base = sub["test_prauc_base"].to_numpy(dtype=float, copy=False)
            roc_base = sub["test_rocauc_base"].to_numpy(dtype=float, copy=False)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_pr = pr / pr_base
                rel_roc = roc / roc_base
            rel_pr_mean, _, _, _ = _mean_ci(rel_pr)
            rel_roc_mean, _, _, _ = _mean_ci(rel_roc)

            rows.append(
                {
                    "DGP": dgp_letter,
                    "Arm": str(arm),
                    "s": float(s),
                    "d_pr": _fmt_ci(pr_mean, pr_lo, pr_hi, digits=4),
                    "rel_pr_pct": 100.0 * rel_pr_mean,
                    "d_roc": _fmt_ci(roc_mean, roc_lo, roc_hi, digits=4),
                    "rel_roc_pct": 100.0 * rel_roc_mean,
                    "d_cooc": cooc_mean,
                    "d_lat": lat_mean,
                    "n": int(n),
                }
            )
        return pd.DataFrame(rows).sort_values(["Arm", "s"])

    def write_longtable(df: pd.DataFrame, *, out_path: Path, caption: str, label: str) -> None:
        lines = []
        lines.append("\\begin{landscape}")
        lines.append("\\begin{center}")
        lines.append("\\scriptsize")
        lines.append("\\setlength{\\tabcolsep}{3pt}")
        lines.append("\\renewcommand{\\arraystretch}{1.05}")
        lines.append("\\begin{longtable}{llr P{4.2cm} r P{4.2cm} r r r r}")
        lines.append(f"\\caption{{{caption}}}\\label{{{label}}}\\\\")
        lines.append("\\toprule")
        lines.append(
            "DGP & Arm & $s$ & $\\Delta$PR-AUC (95\\% CI) & rel$\\Delta$PR (\\%) & $\\Delta$ROC-AUC (95\\% CI) & rel$\\Delta$ROC (\\%) & $\\Delta$\\texttt{cooc\\_mean} & $\\Delta$\\texttt{latent\\_corr} & $n$ \\\\"
        )
        lines.append("\\midrule")
        lines.append("\\endfirsthead")
        lines.append("\\multicolumn{10}{l}{\\textit{Table \\thetable\\ continued from previous page}}\\\\")
        lines.append("\\toprule")
        lines.append(
            "DGP & Arm & $s$ & $\\Delta$PR-AUC (95\\% CI) & rel$\\Delta$PR (\\%) & $\\Delta$ROC-AUC (95\\% CI) & rel$\\Delta$ROC (\\%) & $\\Delta$\\texttt{cooc\\_mean} & $\\Delta$\\texttt{latent\\_corr} & $n$ \\\\"
        )
        lines.append("\\midrule")
        lines.append("\\endhead")
        lines.append("\\bottomrule")
        lines.append("\\endfoot")
        lines.append("\\bottomrule")
        lines.append("\\endlastfoot")
        for _, r in df.iterrows():
            lines.append(
                f"{r['DGP']} & {r['Arm']} & {r['s']:.1f} & {r['d_pr']} & {r['rel_pr_pct']:.1f} & {r['d_roc']} & {r['rel_roc_pct']:.1f} & {r['d_cooc']:.3f} & {r['d_lat']:.3f} & {int(r['n'])} \\\\"
            )
        lines.append("\\end{longtable}")
        lines.append("\\end{center}")
        lines.append("\\end{landscape}")
        lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")

    # Full sweep tables (F0/F1), stacked over DGP-A + DGP-B.
    f0 = pd.concat(
        [
            full_s_table(sweep_a_results_with_deltas, dgp_letter="A", feature_set="F0"),
            full_s_table(sweep_b_results_with_deltas, dgp_letter="B", feature_set="F0"),
        ],
        ignore_index=True,
    )
    f1 = pd.concat(
        [
            full_s_table(sweep_a_results_with_deltas, dgp_letter="A", feature_set="F1"),
            full_s_table(sweep_b_results_with_deltas, dgp_letter="B", feature_set="F1"),
        ],
        ignore_index=True,
    )

    write_longtable(
        f0,
        out_path=out_dir / "tab_app_v3_f0.tex",
        caption=(
            "Full $s$ sweep results for feature set F0 (primitives only). DGP A is continuous log ratio; DGP B is count plus exposure. "
            "Deltas are paired against the matched baseline (C0, $s=1.0$). Relative deltas are percent change relative to the corresponding baseline metric."
        ),
        label="tab:app_v3_f0",
    )
    write_longtable(
        f1,
        out_path=out_dir / "tab_app_v3_f1.tex",
        caption="Full $s$ sweep results for feature set F1 (primitives plus engineered ratio). Same conventions as Table~\\ref{tab:app_v3_f0}.",
        label="tab:app_v3_f1",
    )

    # Main-body table (DGP-A, C3 only).
    df = pd.read_csv(sweep_a_results_with_deltas)
    df = df[(df["dgp_name"] == "continuous_logratio") & (df["colsamp_arm"] == "C3")].copy()
    df = df[df["s"].isin([0.4, 0.6, 0.8, 0.9])].copy()

    def panel(feature_set: str) -> list[str]:
        sub_fs = df[df["feature_set"] == feature_set]
        lines = []
        for s in [0.4, 0.6, 0.8, 0.9]:
            sub = sub_fs[sub_fs["s"] == s]
            pr = sub["delta_test_prauc"].to_numpy(dtype=float, copy=False)
            roc = sub["delta_test_rocauc"].to_numpy(dtype=float, copy=False)
            cooc = sub["delta_cooc_mean"].to_numpy(dtype=float, copy=False)
            lat = sub["delta_test_latent_corr"].to_numpy(dtype=float, copy=False)
            pr_mean, pr_lo, pr_hi, n = _mean_ci(pr)
            roc_mean, roc_lo, roc_hi, _ = _mean_ci(roc)
            cooc_mean, _, _, _ = _mean_ci(cooc)
            lat_mean, _, _, _ = _mean_ci(lat)

            pr_base = sub["test_prauc_base"].to_numpy(dtype=float, copy=False)
            roc_base = sub["test_rocauc_base"].to_numpy(dtype=float, copy=False)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_pr = pr / pr_base
                rel_roc = roc / roc_base
            rel_pr_mean, _, _, _ = _mean_ci(rel_pr)
            rel_roc_mean, _, _, _ = _mean_ci(rel_roc)

            lines.append(
                f"C3 & {s:.1f} & {_fmt_ci(pr_mean, pr_lo, pr_hi, digits=4)} & {100.0*rel_pr_mean:.1f} & {_fmt_ci(roc_mean, roc_lo, roc_hi, digits=4)} & {100.0*rel_roc_mean:.1f} & {cooc_mean:.3f} & {lat_mean:.3f} & {int(n)} \\\\"
            )
        return lines

    base = pd.read_csv(sweep_a_results_with_deltas)
    base = base[(base["dgp_name"] == "continuous_logratio") & (base["colsamp_arm"] == "C0") & (base["s"] == 1.0)].copy()
    base_lines = []
    for fs in ["F0", "F1"]:
        sub = base[base["feature_set"] == fs]
        pr = float(sub["test_prauc"].mean())
        roc = float(sub["test_rocauc"].mean())
        lat = float(sub["test_latent_corr"].mean())
        cooc = float(sub["cooc_mean"].mean())
        n = int(sub.shape[0])
        base_lines.append(f"{fs} & {pr:.4f} & {roc:.4f} & {lat:.4f} & {cooc:.4f} & {n} \\\\")

    main_tex = out_dir / "tab_main_body_c3_dgpA.tex"
    main_tex.write_text(
        "\n".join(
            [
                "\\begin{table}[t]",
                "\\centering",
                "\\scriptsize",
                "\\setlength{\\tabcolsep}{3.0pt}",
                "\\renewcommand{\\arraystretch}{1.15}",
                "\\caption{DGP-A (continuous log ratio), arm C3. Panels A and B report paired deltas versus the baseline C0. Panel C reports baseline raw metric values under C0 ($s=1.0$). Relative deltas are percent change relative to the corresponding baseline.}",
                "\\label{tab:main_dgpA_c3}",
                "\\vspace{4pt}",
                "",
                "\\noindent\\textbf{Panel A: F0 (primitives only)}\\\\[-2pt]",
                "\\resizebox{\\textwidth}{!}{%",
                "\\begin{tabular}{llllllrrr}",
                "\\toprule",
                "Arm & $s$ & \\shortstack{$\\Delta$PR-AUC\\\\(95\\% CI)} & \\shortstack{rel$\\Delta$PR-AUC\\\\(\\%)} & \\shortstack{$\\Delta$ROC-AUC\\\\(95\\% CI)} & \\shortstack{rel$\\Delta$ROC-AUC\\\\(\\%)} & $\\Delta$cooc\\_path\\_mean & $\\Delta$latent\\_corr & $n$ \\\\",
                "\\midrule",
                *panel("F0"),
                "\\bottomrule",
                "\\end{tabular}%",
                "}",
                "",
                "\\vspace{6pt}",
                "\\noindent\\textbf{Panel B: F1 (primitives + engineered ratio)}\\\\[-2pt]",
                "\\resizebox{\\textwidth}{!}{%",
                "\\begin{tabular}{llllllrrr}",
                "\\toprule",
                "Arm & $s$ & \\shortstack{$\\Delta$PR-AUC\\\\(95\\% CI)} & \\shortstack{rel$\\Delta$PR-AUC\\\\(\\%)} & \\shortstack{$\\Delta$ROC-AUC\\\\(95\\% CI)} & \\shortstack{rel$\\Delta$ROC-AUC\\\\(\\%)} & $\\Delta$cooc\\_path\\_mean & $\\Delta$latent\\_corr & $n$ \\\\",
                "\\midrule",
                *panel("F1"),
                "\\bottomrule",
                "\\end{tabular}%",
                "}",
                "",
                "\\vspace{6pt}",
                "\\noindent\\textbf{Panel C: Baseline C0 (raw values, $s=1.0$)}\\\\[-2pt]",
                "\\begin{tabular}{lrrrrr}",
                "\\toprule",
                "Feature set & PR-AUC & ROC-AUC & latent\\_corr & cooc\\_path\\_mean & $n$ \\\\",
                "\\midrule",
                *base_lines,
                "\\bottomrule",
                "\\end{tabular}%",
                "\\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    sweep_a = repo_root / "outputs/aggregate/agg_20251220_223214_only"
    sweep_b = repo_root / "outputs/aggregate/agg_20251220_230029_only"
    boundary = {
        "strong": repo_root / "outputs/aggregate/agg_20251220_221556_only",
        "mid": repo_root / "outputs/aggregate/agg_20251220_221807_only",
        "weak": repo_root / "outputs/aggregate/agg_20251220_221953_only",
    }

    out_assets = repo_root / "paper" / "assets"
    out_tables = repo_root / "paper" / "tables"
    out_assets.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    # Main figure (DGP-A, F0): relΔPR-AUC (%) + Δcooc_mean.
    df_main = _dual_axis_summary(
        sweep_a / "results_with_deltas.csv",
        dgp_name="continuous_logratio",
        feature_set="F0",
        delta_metric="delta_test_prauc",
        base_metric="test_prauc_base",
        right_metric="delta_cooc_mean",
    )
    _plot_dual_axis_3panel(
        df_main,
        title="Performance change with respect to s for DGP-A in F0 feature set",
        left_label="relΔ PR-AUC (%)",
        right_label="Δcooc_mean",
        out_path=out_assets / "fig_main_dgpA_f0_dual_axis_c1c2c3.pdf",
    )

    # Appendix dual-axis (ROC-AUC, DGP-A F0).
    df_a_roc = _dual_axis_summary(
        sweep_a / "results_with_deltas.csv",
        dgp_name="continuous_logratio",
        feature_set="F0",
        delta_metric="delta_test_rocauc",
        base_metric="test_rocauc_base",
        right_metric="delta_cooc_mean",
    )
    _plot_dual_axis_3panel(
        df_a_roc,
        title="Performance change with respect to s for continuous_logratio in F0 (relΔ ROC-AUC (%))",
        left_label="relΔ ROC-AUC (%)",
        right_label="Δcooc_mean",
        out_path=out_assets / "figA_dgpA_f0_dual_axis_rel_roc_auc.pdf",
    )

    # Appendix dual-axis (PR-AUC + ROC-AUC, DGP-B F0).
    df_b_pr = _dual_axis_summary(
        sweep_b / "results_with_deltas.csv",
        dgp_name="count_exposure",
        feature_set="F0",
        delta_metric="delta_test_prauc",
        base_metric="test_prauc_base",
        right_metric="delta_cooc_mean",
    )
    _plot_dual_axis_3panel(
        df_b_pr,
        title="Performance change with respect to s for count_exposure in F0 (relΔ PR-AUC (%))",
        left_label="relΔ PR-AUC (%)",
        right_label="Δcooc_mean",
        out_path=out_assets / "figA_dgpB_f0_dual_axis_rel_pr_auc.pdf",
    )

    df_b_roc = _dual_axis_summary(
        sweep_b / "results_with_deltas.csv",
        dgp_name="count_exposure",
        feature_set="F0",
        delta_metric="delta_test_rocauc",
        base_metric="test_rocauc_base",
        right_metric="delta_cooc_mean",
    )
    _plot_dual_axis_3panel(
        df_b_roc,
        title="Performance change with respect to s for count_exposure in F0 (relΔ ROC-AUC (%))",
        left_label="relΔ ROC-AUC (%)",
        right_label="Δcooc_mean",
        out_path=out_assets / "figA_dgpB_f0_dual_axis_rel_roc_auc.pdf",
    )

    # Low-s sweep grids.
    _plot_s_sweep_grid(
        sweep_a / "summary.csv",
        out_path=out_assets / "figA1_dgpA_low_s_sweep.png",
        title_prefix="DGP-A sweep (continuous_logratio)",
    )
    _plot_s_sweep_grid(
        sweep_b / "summary.csv",
        out_path=out_assets / "figA2_dgpB_low_s_sweep.png",
        title_prefix="DGP-B sweep (count_exposure)",
    )

    # Boundary sweep (ΔPR-AUC, C3).
    points: list[BoundaryPoint] = []
    for label, d in boundary.items():
        points.extend(_load_boundary_points(d / "results_with_deltas.csv", label=label))
    _plot_boundary_delta_prauc(points, out_path=out_assets / "figA3_dgpA_boundary_sweep_delta_prauc.png")

    # Tables.
    _build_tables(
        out_dir=out_tables,
        sweep_a_results_with_deltas=sweep_a / "results_with_deltas.csv",
        sweep_b_results_with_deltas=sweep_b / "results_with_deltas.csv",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
