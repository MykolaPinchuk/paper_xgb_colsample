# Effect report (C? vs C0 deltas)

- generated_utc: 2025-12-20T22:21:37.389796+00:00
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
| C3 | 0.8 | -0.0114 [-0.0170, -0.0058] | hurts | -0.0195 [-0.0268, -0.0121] | hurts | -0.0437 [-0.0674, -0.0200] | hurts | -0.0055 [-0.0385, 0.0276] | inconclusive | 16 |
| C3 | 0.9 | -0.0051 [-0.0096, -0.0007] | hurts | -0.0092 [-0.0141, -0.0044] | hurts | -0.0254 [-0.0453, -0.0055] | hurts | 0.0118 [-0.0318, 0.0553] | inconclusive | 16 |

## continuous_logratio / F1
- colsamp_arms: ['C3']

| colsamp_arm | s | ΔPR-AUC mean (95% CI) | PR effect | ΔROC-AUC mean (95% CI) | ROC effect | Δlatent corr mean (95% CI) | latent effect | Δcooc mean (95% CI) | cooc effect | n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 | 0.8 | 0.0024 [-0.0000, 0.0048] | inconclusive | -0.0014 [-0.0033, 0.0004] | inconclusive | -0.0130 [-0.0386, 0.0127] | inconclusive | 0.0458 [0.0340, 0.0575] | helps | 16 |
| C3 | 0.9 | 0.0007 [-0.0024, 0.0039] | inconclusive | -0.0007 [-0.0030, 0.0017] | inconclusive | 0.0004 [-0.0395, 0.0402] | inconclusive | 0.0160 [0.0053, 0.0268] | helps | 16 |

