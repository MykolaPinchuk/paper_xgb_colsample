# Effect report (C? vs C0 deltas)

- generated_utc: 2025-12-20T22:36:34.548059+00:00
- rows: 312

## Legend
- `F0`: primitives only
- `F1`: primitives + engineered ratio feature
- `C0`: bylevel=1.0, bynode=1.0 (baseline)
- Deltas are computed within (run_id, scenario_id, rep_id, feature_set, depth, regime) relative to `C0`.
- `Δlatent corr`: change in Pearson correlation between model logit(pred) and the true latent signal (V or V_sum).

## continuous_logratio / F0
- colsamp_arms: ['C1', 'C2', 'C3']

| colsamp_arm | s | ΔPR-AUC mean (95% CI) | PR effect | ΔROC-AUC mean (95% CI) | ROC effect | Δlatent corr mean (95% CI) | latent effect | Δcooc mean (95% CI) | cooc effect | n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 0.4 | -0.0381 [-0.0458, -0.0304] | hurts | -0.0576 [-0.0736, -0.0416] | hurts | -0.1580 [-0.2079, -0.1081] | hurts | -0.1502 [-0.1930, -0.1074] | hurts | 12 |
| C1 | 0.6 | -0.0198 [-0.0253, -0.0144] | hurts | -0.0282 [-0.0413, -0.0151] | hurts | -0.0822 [-0.1261, -0.0383] | hurts | -0.0228 [-0.0703, 0.0247] | inconclusive | 12 |
| C1 | 0.8 | -0.0063 [-0.0121, -0.0006] | hurts | -0.0134 [-0.0247, -0.0022] | hurts | -0.0303 [-0.0688, 0.0082] | inconclusive | 0.0018 [-0.0345, 0.0380] | inconclusive | 12 |
| C1 | 0.9 | -0.0032 [-0.0082, 0.0018] | inconclusive | -0.0072 [-0.0148, 0.0004] | inconclusive | -0.0191 [-0.0416, 0.0035] | inconclusive | 0.0205 [-0.0188, 0.0598] | inconclusive | 12 |
| C2 | 0.4 | -0.0359 [-0.0466, -0.0251] | hurts | -0.0516 [-0.0653, -0.0379] | hurts | -0.1432 [-0.1872, -0.0991] | hurts | -0.0536 [-0.0924, -0.0148] | hurts | 12 |
| C2 | 0.6 | -0.0202 [-0.0293, -0.0111] | hurts | -0.0267 [-0.0389, -0.0146] | hurts | -0.0783 [-0.1291, -0.0275] | hurts | 0.0220 [-0.0366, 0.0806] | inconclusive | 12 |
| C2 | 0.8 | -0.0093 [-0.0160, -0.0025] | hurts | -0.0148 [-0.0249, -0.0048] | hurts | -0.0397 [-0.0746, -0.0048] | hurts | 0.0312 [-0.0028, 0.0652] | inconclusive | 12 |
| C2 | 0.9 | -0.0027 [-0.0074, 0.0021] | inconclusive | -0.0045 [-0.0126, 0.0036] | inconclusive | -0.0205 [-0.0491, 0.0082] | inconclusive | 0.0322 [-0.0148, 0.0791] | inconclusive | 12 |
| C3 | 0.4 | -0.0897 [-0.0960, -0.0833] | hurts | -0.1705 [-0.1880, -0.1530] | hurts | -0.4694 [-0.5051, -0.4337] | hurts | -0.3432 [-0.4047, -0.2818] | hurts | 12 |
| C3 | 0.6 | -0.0459 [-0.0577, -0.0341] | hurts | -0.0693 [-0.0877, -0.0509] | hurts | -0.2070 [-0.2637, -0.1503] | hurts | -0.0874 [-0.1311, -0.0436] | hurts | 12 |
| C3 | 0.8 | -0.0104 [-0.0172, -0.0036] | hurts | -0.0184 [-0.0280, -0.0089] | hurts | -0.0414 [-0.0707, -0.0121] | hurts | 0.0011 [-0.0332, 0.0354] | inconclusive | 12 |
| C3 | 0.9 | -0.0071 [-0.0116, -0.0026] | hurts | -0.0095 [-0.0153, -0.0037] | hurts | -0.0317 [-0.0536, -0.0099] | hurts | 0.0377 [-0.0064, 0.0817] | inconclusive | 12 |

## continuous_logratio / F1
- colsamp_arms: ['C1', 'C2', 'C3']

| colsamp_arm | s | ΔPR-AUC mean (95% CI) | PR effect | ΔROC-AUC mean (95% CI) | ROC effect | Δlatent corr mean (95% CI) | latent effect | Δcooc mean (95% CI) | cooc effect | n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 0.4 | 0.0008 [-0.0053, 0.0069] | inconclusive | -0.0056 [-0.0111, -0.0002] | hurts | 0.0037 [-0.0534, 0.0608] | inconclusive | 0.0455 [0.0304, 0.0606] | helps | 12 |
| C1 | 0.6 | 0.0009 [-0.0053, 0.0070] | inconclusive | -0.0003 [-0.0027, 0.0022] | inconclusive | 0.0187 [-0.0211, 0.0585] | inconclusive | 0.0240 [0.0135, 0.0346] | helps | 12 |
| C1 | 0.8 | 0.0036 [0.0004, 0.0068] | helps | 0.0002 [-0.0013, 0.0018] | inconclusive | 0.0122 [-0.0242, 0.0486] | inconclusive | 0.0058 [-0.0010, 0.0127] | inconclusive | 12 |
| C1 | 0.9 | 0.0017 [-0.0008, 0.0042] | inconclusive | -0.0001 [-0.0015, 0.0014] | inconclusive | -0.0008 [-0.0282, 0.0266] | inconclusive | 0.0035 [-0.0011, 0.0081] | inconclusive | 12 |
| C2 | 0.4 | 0.0012 [-0.0028, 0.0051] | inconclusive | -0.0075 [-0.0140, -0.0010] | hurts | -0.0199 [-0.0666, 0.0269] | inconclusive | 0.0746 [0.0576, 0.0917] | helps | 12 |
| C2 | 0.6 | 0.0034 [-0.0001, 0.0068] | inconclusive | -0.0008 [-0.0030, 0.0013] | inconclusive | 0.0108 [-0.0385, 0.0600] | inconclusive | 0.0479 [0.0228, 0.0731] | helps | 12 |
| C2 | 0.8 | 0.0002 [-0.0027, 0.0031] | inconclusive | 0.0009 [-0.0007, 0.0025] | inconclusive | 0.0101 [-0.0334, 0.0536] | inconclusive | 0.0142 [0.0049, 0.0234] | helps | 12 |
| C2 | 0.9 | 0.0018 [-0.0004, 0.0040] | inconclusive | 0.0004 [-0.0009, 0.0016] | inconclusive | 0.0279 [-0.0009, 0.0567] | inconclusive | 0.0068 [0.0015, 0.0120] | helps | 12 |
| C3 | 0.4 | -0.0085 [-0.0123, -0.0048] | hurts | -0.0118 [-0.0187, -0.0050] | hurts | 0.0197 [-0.0275, 0.0668] | inconclusive | 0.0636 [0.0525, 0.0748] | helps | 12 |
| C3 | 0.6 | -0.0020 [-0.0076, 0.0036] | inconclusive | -0.0062 [-0.0111, -0.0013] | hurts | -0.0028 [-0.0567, 0.0511] | inconclusive | 0.0856 [0.0629, 0.1083] | helps | 12 |
| C3 | 0.8 | 0.0027 [-0.0002, 0.0057] | inconclusive | -0.0019 [-0.0042, 0.0004] | inconclusive | -0.0111 [-0.0436, 0.0213] | inconclusive | 0.0448 [0.0321, 0.0574] | helps | 12 |
| C3 | 0.9 | 0.0013 [-0.0010, 0.0036] | inconclusive | -0.0008 [-0.0028, 0.0012] | inconclusive | 0.0052 [-0.0392, 0.0496] | inconclusive | 0.0165 [0.0049, 0.0281] | helps | 12 |

