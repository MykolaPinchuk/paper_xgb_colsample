# Effect report (C? vs C0 deltas)

- generated_utc: 2025-12-20T22:21:49.405070+00:00
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
| C3 | 0.8 | -0.0066 [-0.0086, -0.0046] | hurts | -0.0051 [-0.0070, -0.0031] | hurts | -0.0078 [-0.0195, 0.0039] | inconclusive | 0.0361 [-0.0111, 0.0833] | inconclusive | 16 |
| C3 | 0.9 | -0.0038 [-0.0074, -0.0002] | hurts | -0.0009 [-0.0034, 0.0015] | inconclusive | -0.0007 [-0.0104, 0.0090] | inconclusive | 0.0354 [-0.0118, 0.0826] | inconclusive | 16 |

## continuous_logratio / F1
- colsamp_arms: ['C3']

| colsamp_arm | s | ΔPR-AUC mean (95% CI) | PR effect | ΔROC-AUC mean (95% CI) | ROC effect | Δlatent corr mean (95% CI) | latent effect | Δcooc mean (95% CI) | cooc effect | n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 | 0.8 | 0.0034 [0.0006, 0.0062] | helps | 0.0000 [-0.0016, 0.0017] | inconclusive | -0.0138 [-0.0297, 0.0021] | inconclusive | 0.1183 [0.0906, 0.1460] | helps | 16 |
| C3 | 0.9 | 0.0009 [-0.0019, 0.0036] | inconclusive | -0.0006 [-0.0020, 0.0008] | inconclusive | 0.0090 [-0.0322, 0.0501] | inconclusive | 0.0487 [0.0290, 0.0683] | helps | 16 |

