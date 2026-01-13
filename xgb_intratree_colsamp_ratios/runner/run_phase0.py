from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..dgp.acceptance import evaluate_acceptance
from ..dgp.continuous_logratio import generate as gen_continuous_logratio
from ..dgp.count_exposure import generate as gen_count_exposure
from ..dgp.multicomponent import generate as gen_multicomponent
from ..modeling.arms import build_arms
from ..modeling.diagnostics_paths import (
    cooccurrence_for_pairs,
    path_fraction_for_pairs,
    tree_level_all_features_cooccurrence_fraction,
    tree_level_all_features_path_fraction,
)
from ..modeling.metrics import compute_binary_metrics, logit, pearson_corr
from ..modeling.xgb_train import train_xgb
from ..utils.config import dump_yaml, load_yaml, make_run_paths
from ..utils.sampling import sample_from_spec
from ..utils.seed import default_xgb_threads, rng_from_seed


def _utc_run_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def _assemble_X(split, feature_set: str) -> pd.DataFrame:
    if feature_set == "F0":
        return split.X_base
    if feature_set == "F1":
        return pd.concat([split.X_base, split.X_engineered], axis=1)
    raise ValueError(f"Unknown feature_set={feature_set!r}")


def _flatten_dict(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def _write_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _write_report(
    *,
    path: Path,
    run_id: str,
    cfg: dict[str, Any],
    apply_acceptance_filter: bool,
    acceptance_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    data_cfg = cfg.get("data", {})
    scenario_cfg = cfg.get("scenarios", {})
    acceptance_cfg = cfg.get("acceptance", {})
    xgb_cfg = cfg.get("xgb", {})

    lines: list[str] = []
    lines.append("# Run report")
    lines.append("")
    lines.append(f"- run_id: {run_id}")
    lines.append(f"- generated_utc: {now}")
    lines.append(f"- config_path: {path.parent / 'config.yaml'}")
    lines.append("")
    lines.append("## Data")
    lines.append(f"- n_train: {data_cfg.get('n_train')}")
    lines.append(f"- n_valid: {data_cfg.get('n_valid')}")
    lines.append(f"- n_test: {data_cfg.get('n_test')}")
    lines.append(f"- prevalence_target: {data_cfg.get('prevalence_target')}")
    lines.append("")
    lines.append("## Legend")
    lines.append("- `F0`: primitives only")
    lines.append("- `F1`: primitives + engineered ratio feature")
    lines.append("- `C0`: bylevel=1.0, bynode=1.0 (baseline)")
    lines.append("- `C1`: bylevel=s, bynode=1.0")
    lines.append("- `C2`: bylevel=1.0, bynode=s")
    lines.append("- `C3`: bylevel=s, bynode=s")
    lines.append("- `cooc_mean`: fraction of trees where at least one root→leaf path uses both features in the intended pair(s)")
    lines.append(
        "- `cooc_path_mean`: mean fraction of leaf-paths (weighted by XGB JSON `cover`) that contain both features in the intended pair(s)"
    )
    lines.append(
        "- `cooc_allfeat_tree`: fraction of trees with a path containing all intended primitive features (only for multicomponent when depth permits)"
    )
    lines.append(
        "- `cooc_allfeat_path`: mean fraction of leaf-paths (weighted by XGB JSON `cover`) containing all intended primitive features (only for multicomponent when depth permits)"
    )
    lines.append("")
    lines.append("## Scenario")
    lines.append(f"- dgp_name: {scenario_cfg.get('dgp_name')}")
    lines.append(f"- n_scenarios: {scenario_cfg.get('n_scenarios')}")
    lines.append(f"- n_reps: {scenario_cfg.get('n_reps')}")
    lines.append(f"- apply_acceptance_filter: {apply_acceptance_filter}")
    lines.append("")
    lines.append("## Acceptance")
    total_acc = len(acceptance_rows)
    accepted = sum(1 for r in acceptance_rows if bool(r.get("accepted")))
    lines.append(f"- total_rows: {total_acc}")
    lines.append(f"- accepted_rows: {accepted}")
    lines.append(f"- acceptance_rate: {accepted / total_acc:.3f}" if total_acc else "- acceptance_rate: n/a")
    lines.append(
        "- oracle_gap_thresholds: rocauc>="
        f"{acceptance_cfg.get('min_oracle_minus_best1d_rocauc')}, prauc>="
        f"{acceptance_cfg.get('min_oracle_minus_best1d_prauc')}"
    )
    lines.append(
        "- xgb_gap_frac_thresholds: rocauc>="
        f"{acceptance_cfg.get('min_xgb_closes_oracle_gap_frac')}, prauc>="
        f"{acceptance_cfg.get('min_xgb_closes_oracle_gap_frac_prauc')}"
    )
    lines.append("")
    lines.append("## Training grid")
    lines.append(f"- depths: {xgb_cfg.get('depths')}")
    lines.append(f"- colsamp_s: {xgb_cfg.get('colsamp_s')}")
    lines.append(f"- colsamp_arms: {xgb_cfg.get('colsamp_arms')}")
    lines.append(f"- regimes: {[r.get('name') for r in xgb_cfg.get('regimes', [])]}")
    lines.append("")
    lines.append("## Results")
    lines.append(f"- results_rows: {len(result_rows)}")
    if result_rows:
        by_key: dict[tuple[str, str], dict[str, list[float]]] = {}
        for r in result_rows:
            key = (str(r.get("feature_set")), str(r.get("colsamp_arm")))
            entry = by_key.setdefault(
                key,
                {
                    "test_prauc": [],
                    "test_rocauc": [],
                    "test_latent_corr": [],
                    "cooc_mean": [],
                    "cooc_path_mean": [],
                    "cooc_allfeat_tree": [],
                    "cooc_allfeat_path": [],
                },
            )
            if "test_prauc" in r:
                entry["test_prauc"].append(float(r["test_prauc"]))
            if "test_rocauc" in r:
                entry["test_rocauc"].append(float(r["test_rocauc"]))
            if "test_latent_corr" in r:
                entry["test_latent_corr"].append(float(r["test_latent_corr"]))
            if "cooc_mean" in r:
                entry["cooc_mean"].append(float(r["cooc_mean"]))
            if "cooc_path_mean" in r:
                entry["cooc_path_mean"].append(float(r["cooc_path_mean"]))
            if "cooc_allfeat_tree" in r:
                entry["cooc_allfeat_tree"].append(float(r["cooc_allfeat_tree"]))
            if "cooc_allfeat_path" in r:
                entry["cooc_allfeat_path"].append(float(r["cooc_allfeat_path"]))
        lines.append("")
        lines.append("### Mean metrics by feature_set / colsamp_arm")
        for (feature_set, colsamp_arm), vals in sorted(by_key.items()):
            lines.append(
                f"- {feature_set}/{colsamp_arm}: "
                f"test_prauc={_mean(vals['test_prauc']):.4f}, "
                f"test_rocauc={_mean(vals['test_rocauc']):.4f}, "
                f"test_latent_corr={_mean(vals['test_latent_corr']):.4f}, "
                f"cooc_mean={_mean(vals['cooc_mean']):.4f}, "
                f"cooc_path_mean={_mean(vals['cooc_path_mean']):.4f}, "
                f"cooc_allfeat_tree={_mean(vals['cooc_allfeat_tree']):.4f}, "
                f"cooc_allfeat_path={_mean(vals['cooc_allfeat_path']):.4f}"
            )
    else:
        lines.append("- no models trained (results.csv not created)")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--run-id", default=None, help="Override run_id (default: timestamp).")
    ap.add_argument(
        "--no-acceptance-filter",
        action="store_true",
        help="Run training even if acceptance fails.",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    run_id = args.run_id or _utc_run_id("phase0")
    paths = make_run_paths(run_id)
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(cfg, paths.config_yaml)

    reserve_cpus = int(cfg.get("compute", {}).get("reserve_cpus", 2))
    xgb_threads_cfg = cfg.get("compute", {}).get("xgb_threads", None)
    nthread = int(xgb_threads_cfg) if xgb_threads_cfg is not None else default_xgb_threads(reserve_cpus)

    seed_master = int(cfg["seed_master"])
    sizes = cfg["data"]
    n_train = int(sizes["n_train"])
    n_valid = int(sizes["n_valid"])
    n_test = int(sizes["n_test"])
    prevalence_target = float(sizes["prevalence_target"])

    scenario_cfg = cfg["scenarios"]
    dgp_name = str(scenario_cfg["dgp_name"])
    n_scenarios = int(scenario_cfg["n_scenarios"])
    n_reps = int(scenario_cfg["n_reps"])
    apply_acceptance_filter = bool(scenario_cfg.get("apply_acceptance_filter", True))
    if args.no_acceptance_filter:
        apply_acceptance_filter = False

    xgb_cfg = cfg["xgb"]
    xgb_fixed = dict(xgb_cfg["fixed"])
    depths = [int(x) for x in xgb_cfg["depths"]]
    colsamp_s = [float(x) for x in xgb_cfg["colsamp_s"]]
    regimes = list(xgb_cfg["regimes"])
    allowed_colsamp_arms = xgb_cfg.get("colsamp_arms", None)

    arms = build_arms(
        depths=depths,
        colsamp_s=colsamp_s,
        regimes=regimes,
        allowed_colsamp_arms=list(allowed_colsamp_arms) if allowed_colsamp_arms is not None else None,
    )

    acceptance_cfg = cfg["acceptance"]
    diagnostics_cfg = cfg.get("diagnostics", {})
    compute_cooc = bool(diagnostics_cfg.get("path_cooccurrence", False))

    wrote_any = False
    acceptance_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    # Write an initial report early so partial runs still leave context.
    _write_report(
        path=paths.outputs_dir / "report.md",
        run_id=run_id,
        cfg=cfg,
        apply_acceptance_filter=apply_acceptance_filter,
        acceptance_rows=acceptance_rows,
        result_rows=result_rows,
    )
    for scenario_id in range(n_scenarios):
        scenario_rng = rng_from_seed(seed_master + scenario_id * 10_000 + 17)
        dgp_spec = cfg["dgp"][dgp_name]
        dgp_params = sample_from_spec(scenario_rng, dgp_spec)

        for rep_id in range(n_reps):
            rep_seed = seed_master + scenario_id * 1_000 + rep_id
            rng = rng_from_seed(rep_seed)

            if dgp_name == "continuous_logratio":
                dataset = gen_continuous_logratio(
                    rng,
                    n_train=n_train,
                    n_valid=n_valid,
                    n_test=n_test,
                    prevalence_target=prevalence_target,
                    **dgp_params,
                )
            elif dgp_name == "count_exposure":
                dataset = gen_count_exposure(
                    rng,
                    n_train=n_train,
                    n_valid=n_valid,
                    n_test=n_test,
                    prevalence_target=prevalence_target,
                    **dgp_params,
                )
            elif dgp_name == "multicomponent":
                dataset = gen_multicomponent(
                    rng,
                    n_train=n_train,
                    n_valid=n_valid,
                    n_test=n_test,
                    prevalence_target=prevalence_target,
                    **dgp_params,
                )
            else:
                raise ValueError(f"Unknown dgp_name={dgp_name!r}")

            acc = evaluate_acceptance(
                dataset,
                acceptance_cfg=acceptance_cfg,
                xgb_fixed_cfg=xgb_fixed,
                nthread=nthread,
                seed=rep_seed,
            )

            acc_row: dict[str, Any] = {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "rep_id": rep_id,
                "dgp_name": dataset.dgp_name,
                **_flatten_dict(dataset.params, prefix="dgp."),
                "nthread": nthread,
                "oracle_rocauc": acc.oracle.roc_auc,
                "oracle_prauc": acc.oracle.pr_auc,
                "engineered_ratio_rocauc": acc.engineered_ratio.roc_auc,
                "engineered_ratio_prauc": acc.engineered_ratio.pr_auc,
                "best1d_rocauc": acc.best1d.roc_auc,
                "best1d_prauc": acc.best1d.pr_auc,
                "feas_xgb_rocauc": acc.feasibility_xgb.roc_auc,
                "feas_xgb_prauc": acc.feasibility_xgb.pr_auc,
                "gap_frac_rocauc": acc.closes_oracle_gap_frac_rocauc,
                "gap_frac_prauc": acc.closes_oracle_gap_frac_prauc,
                "accepted": acc.accepted,
            }
            _write_row(paths.outputs_dir / "acceptance.csv", acc_row)
            acceptance_rows.append(acc_row)

            if apply_acceptance_filter and not acc.accepted:
                _write_report(
                    path=paths.outputs_dir / "report.md",
                    run_id=run_id,
                    cfg=cfg,
                    apply_acceptance_filter=apply_acceptance_filter,
                    acceptance_rows=acceptance_rows,
                    result_rows=result_rows,
                )
                continue

            tr = dataset.splits["train"]
            va = dataset.splits["valid"]
            te = dataset.splits["test"]

            if dataset.dgp_name == "continuous_logratio":
                latent_valid = va.latents["V"]
                latent_test = te.latents["V"]
            elif dataset.dgp_name == "count_exposure":
                latent_valid = va.latents["V"]
                latent_test = te.latents["V"]
            elif dataset.dgp_name == "multicomponent":
                latent_valid = va.latents["V_sum"]
                latent_test = te.latents["V_sum"]
            else:
                raise ValueError(f"Unknown dgp_name={dataset.dgp_name!r}")

            for arm in arms:
                X_train = _assemble_X(tr, arm.feature_set)
                X_valid = _assemble_X(va, arm.feature_set)
                X_test = _assemble_X(te, arm.feature_set)

                train_res = train_xgb(
                    X_train=X_train,
                    y_train=tr.y,
                    X_valid=X_valid,
                    y_valid=va.y,
                    fixed_params=xgb_fixed,
                    arm_params={
                        "max_depth": arm.max_depth,
                        "colsample_bylevel": arm.colsample_bylevel,
                        "colsample_bynode": arm.colsample_bynode,
                    },
                    n_estimators=arm.n_estimators,
                    early_stopping=arm.early_stopping,
                    early_stopping_rounds=arm.early_stopping_rounds,
                    nthread=nthread,
                    seed=rep_seed,
                )

                valid_score = train_res.model.predict_proba(X_valid)[:, 1]
                test_score = train_res.model.predict_proba(X_test)[:, 1]
                valid_m = compute_binary_metrics(va.y, valid_score)
                test_m = compute_binary_metrics(te.y, test_score)
                valid_latent_corr = pearson_corr(logit(valid_score), latent_valid)
                test_latent_corr = pearson_corr(logit(test_score), latent_test)

                row: dict[str, Any] = {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "rep_id": rep_id,
                    "dgp_name": dataset.dgp_name,
                    **_flatten_dict(dataset.params, prefix="dgp."),
                    "feature_set": arm.feature_set,
                    "colsamp_arm": arm.colsamp_arm,
                    "s": arm.s,
                    "colsample_bylevel": arm.colsample_bylevel,
                    "colsample_bynode": arm.colsample_bynode,
                    "max_depth": arm.max_depth,
                    "regime": arm.regime,
                    "n_estimators_cap": arm.n_estimators,
                    "n_estimators_used": train_res.n_estimators_used,
                    "best_iteration": train_res.best_iteration,
                    "train_seconds": train_res.seconds,
                    "valid_rocauc": valid_m.roc_auc,
                    "valid_prauc": valid_m.pr_auc,
                    "test_rocauc": test_m.roc_auc,
                    "test_prauc": test_m.pr_auc,
                    "valid_latent_corr": valid_latent_corr,
                    "test_latent_corr": test_latent_corr,
                    "nthread": nthread,
                }

                if compute_cooc:
                    intended_features = [c for pair in dataset.intended_pairs for c in pair]
                    cooc = cooccurrence_for_pairs(train_res.model, dataset.intended_pairs)
                    row["cooc_mean"] = cooc.mean
                    for k, v in cooc.per_pair.items():
                        row[f"cooc_{k}"] = v
                    cooc_path = path_fraction_for_pairs(train_res.model, dataset.intended_pairs)
                    row["cooc_path_mean"] = cooc_path.mean
                    for k, v in cooc_path.per_pair.items():
                        row[f"cooc_path_{k}"] = v
                    if len(dataset.intended_pairs) > 1 and (2 * len(dataset.intended_pairs) <= arm.max_depth):
                        row["cooc_allfeat_tree"] = tree_level_all_features_cooccurrence_fraction(
                            train_res.model, intended_features
                        )
                        row["cooc_allfeat_path"] = tree_level_all_features_path_fraction(
                            train_res.model, intended_features
                        )

                _write_row(paths.results_csv, row)
                wrote_any = True
                result_rows.append(row)

            # Update report after each rep so partial runs still produce a useful summary.
            _write_report(
                path=paths.outputs_dir / "report.md",
                run_id=run_id,
                cfg=cfg,
                apply_acceptance_filter=apply_acceptance_filter,
                acceptance_rows=acceptance_rows,
                result_rows=result_rows,
            )

    if wrote_any:
        print(f"Wrote results to {paths.results_csv}")
    else:
        print(f"No models were trained; wrote acceptance stats to {paths.outputs_dir / 'acceptance.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
