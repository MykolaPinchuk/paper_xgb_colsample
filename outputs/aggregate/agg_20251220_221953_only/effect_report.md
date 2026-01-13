# Effect report (C? vs C0 deltas)

- generated_utc: 2025-12-20T22:22:01.463616+00:00
- rows: 96

## Legend
- `F0`: primitives only
- `F1`: primitives + engineered ratio feature
- `C0`: bylevel=1.0, bynode=1.0 (baseline)
- Deltas are computed within (run_id, scenario_id, rep_id, feature_set, depth, regime) relative to `C0`.
- `Δlatent corr`: change in Pearson correlation between model logit(pred) and the true latent signal (V or V_sum).

## continuous_logratio / F0
- colsamp_arms: ['C3']

| colsamp_arm | s | ΔPR-AUC mean (95% CI) | PR effect | ΔROC-AUC mean (95% CI) | ROC effect | Δlatent corr mean (95% CI) | latent effect | Δcooc mean (95% CI) | cooc effect | n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 | 0.8 | -0.0036 [-0.0077, 0.0005] | inconclusive | -0.0051 [-0.0086, -0.0017] | hurts | -0.0139 [-0.0341, 0.0063] | inconclusive | 0.0038 [-0.0360, 0.0436] | inconclusive | 16 |
| C3 | 0.9 | -0.0004 [-0.0023, 0.0014] | inconclusive | -0.0012 [-0.0024, 0.0000] | inconclusive | 0.0017 [-0.0072, 0.0105] | inconclusive | -0.0025 [-0.0507, 0.0458] | inconclusive | 16 |

## continuous_logratio / F1
- colsamp_arms: ['C3']

| colsamp_arm | s | ΔPR-AUC mean (95% CI) | PR effect | ΔROC-AUC mean (95% CI) | ROC effect | Δlatent corr mean (95% CI) | latent effect | Δcooc mean (95% CI) | cooc effect | n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 | 0.8 | 0.0030 [0.0003, 0.0057] | helps | -0.0007 [-0.0023, 0.0008] | inconclusive | -0.0266 [-0.0673, 0.0141] | inconclusive | 0.2138 [0.1764, 0.2513] | helps | 16 |
| C3 | 0.9 | 0.0022 [0.0001, 0.0043] | helps | 0.0004 [-0.0012, 0.0021] | inconclusive | -0.0311 [-0.0577, -0.0045] | hurts | 0.1027 [0.0795, 0.1259] | helps | 16 |

