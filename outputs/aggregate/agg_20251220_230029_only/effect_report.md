# Effect report (C? vs C0 deltas)

- generated_utc: 2025-12-20T23:04:21.217258+00:00
- rows: 312

## Legend
- `F0`: primitives only
- `F1`: primitives + engineered ratio feature
- `C0`: bylevel=1.0, bynode=1.0 (baseline)
- Deltas are computed within (run_id, scenario_id, rep_id, feature_set, depth, regime) relative to `C0`.
- `Δlatent corr`: change in Pearson correlation between model logit(pred) and the true latent signal (V or V_sum).

## count_exposure / F0
- colsamp_arms: ['C1', 'C2', 'C3']

| colsamp_arm | s | ΔPR-AUC mean (95% CI) | PR effect | ΔROC-AUC mean (95% CI) | ROC effect | Δlatent corr mean (95% CI) | latent effect | Δcooc mean (95% CI) | cooc effect | n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 0.4 | 0.0003 [-0.0049, 0.0055] | inconclusive | -0.0024 [-0.0059, 0.0011] | inconclusive | 0.0199 [-0.0256, 0.0655] | inconclusive | -0.3253 [-0.3816, -0.2691] | hurts | 12 |
| C1 | 0.6 | 0.0026 [-0.0010, 0.0062] | inconclusive | -0.0023 [-0.0074, 0.0028] | inconclusive | -0.0010 [-0.0439, 0.0419] | inconclusive | -0.0738 [-0.1187, -0.0289] | hurts | 12 |
| C1 | 0.8 | 0.0023 [-0.0010, 0.0057] | inconclusive | 0.0000 [-0.0033, 0.0033] | inconclusive | 0.0066 [-0.0281, 0.0413] | inconclusive | -0.0309 [-0.1039, 0.0420] | inconclusive | 12 |
| C1 | 0.9 | 0.0021 [-0.0005, 0.0046] | inconclusive | -0.0000 [-0.0021, 0.0021] | inconclusive | 0.0021 [-0.0221, 0.0264] | inconclusive | -0.0011 [-0.0687, 0.0666] | inconclusive | 12 |
| C2 | 0.4 | 0.0008 [-0.0030, 0.0046] | inconclusive | -0.0015 [-0.0062, 0.0032] | inconclusive | 0.0388 [0.0034, 0.0742] | helps | -0.1374 [-0.1752, -0.0995] | hurts | 12 |
| C2 | 0.6 | 0.0023 [-0.0008, 0.0054] | inconclusive | -0.0012 [-0.0033, 0.0009] | inconclusive | 0.0132 [-0.0105, 0.0368] | inconclusive | -0.0602 [-0.1346, 0.0142] | inconclusive | 12 |
| C2 | 0.8 | 0.0020 [-0.0006, 0.0047] | inconclusive | -0.0002 [-0.0025, 0.0021] | inconclusive | -0.0014 [-0.0367, 0.0338] | inconclusive | 0.0164 [-0.0290, 0.0619] | inconclusive | 12 |
| C2 | 0.9 | 0.0041 [0.0009, 0.0072] | helps | 0.0004 [-0.0017, 0.0025] | inconclusive | 0.0016 [-0.0221, 0.0253] | inconclusive | 0.0150 [0.0009, 0.0291] | helps | 12 |
| C3 | 0.4 | -0.0067 [-0.0112, -0.0023] | hurts | -0.0111 [-0.0166, -0.0055] | hurts | 0.0107 [-0.0349, 0.0564] | inconclusive | -0.5894 [-0.6367, -0.5421] | hurts | 12 |
| C3 | 0.6 | -0.0036 [-0.0090, 0.0018] | inconclusive | -0.0051 [-0.0093, -0.0009] | hurts | 0.0010 [-0.0477, 0.0498] | inconclusive | -0.2036 [-0.2550, -0.1523] | hurts | 12 |
| C3 | 0.8 | 0.0017 [-0.0023, 0.0056] | inconclusive | -0.0016 [-0.0051, 0.0019] | inconclusive | 0.0117 [-0.0216, 0.0449] | inconclusive | -0.0174 [-0.0739, 0.0392] | inconclusive | 12 |
| C3 | 0.9 | -0.0002 [-0.0034, 0.0031] | inconclusive | -0.0008 [-0.0029, 0.0013] | inconclusive | 0.0104 [-0.0177, 0.0386] | inconclusive | -0.0487 [-0.1296, 0.0323] | inconclusive | 12 |

## count_exposure / F1
- colsamp_arms: ['C1', 'C2', 'C3']

| colsamp_arm | s | ΔPR-AUC mean (95% CI) | PR effect | ΔROC-AUC mean (95% CI) | ROC effect | Δlatent corr mean (95% CI) | latent effect | Δcooc mean (95% CI) | cooc effect | n |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 0.4 | 0.0051 [-0.0002, 0.0105] | inconclusive | 0.0010 [-0.0012, 0.0031] | inconclusive | 0.0020 [-0.0315, 0.0356] | inconclusive | 0.2455 [0.1901, 0.3008] | helps | 12 |
| C1 | 0.6 | 0.0048 [-0.0010, 0.0107] | inconclusive | 0.0017 [0.0000, 0.0034] | helps | -0.0138 [-0.0514, 0.0238] | inconclusive | 0.2981 [0.2517, 0.3445] | helps | 12 |
| C1 | 0.8 | 0.0019 [-0.0038, 0.0077] | inconclusive | 0.0008 [-0.0011, 0.0026] | inconclusive | -0.0034 [-0.0277, 0.0209] | inconclusive | 0.1349 [0.1044, 0.1655] | helps | 12 |
| C1 | 0.9 | 0.0022 [-0.0033, 0.0077] | inconclusive | 0.0009 [-0.0003, 0.0021] | inconclusive | -0.0084 [-0.0362, 0.0195] | inconclusive | 0.0746 [0.0400, 0.1093] | helps | 12 |
| C2 | 0.4 | 0.0036 [-0.0020, 0.0092] | inconclusive | 0.0005 [-0.0014, 0.0024] | inconclusive | -0.0207 [-0.0541, 0.0127] | inconclusive | 0.4694 [0.4273, 0.5115] | helps | 12 |
| C2 | 0.6 | 0.0032 [-0.0033, 0.0096] | inconclusive | 0.0008 [-0.0017, 0.0034] | inconclusive | -0.0183 [-0.0514, 0.0149] | inconclusive | 0.3938 [0.3373, 0.4502] | helps | 12 |
| C2 | 0.8 | 0.0018 [-0.0036, 0.0071] | inconclusive | 0.0010 [-0.0012, 0.0031] | inconclusive | -0.0046 [-0.0409, 0.0317] | inconclusive | 0.1674 [0.1303, 0.2046] | helps | 12 |
| C2 | 0.9 | 0.0011 [-0.0041, 0.0062] | inconclusive | -0.0002 [-0.0025, 0.0021] | inconclusive | 0.0046 [-0.0259, 0.0351] | inconclusive | 0.0702 [0.0477, 0.0927] | helps | 12 |
| C3 | 0.4 | -0.0030 [-0.0073, 0.0012] | inconclusive | -0.0083 [-0.0137, -0.0028] | hurts | -0.0144 [-0.0555, 0.0268] | inconclusive | 0.2404 [0.1966, 0.2843] | helps | 12 |
| C3 | 0.6 | 0.0027 [-0.0022, 0.0076] | inconclusive | -0.0009 [-0.0040, 0.0022] | inconclusive | -0.0076 [-0.0478, 0.0326] | inconclusive | 0.4049 [0.3610, 0.4488] | helps | 12 |
| C3 | 0.8 | 0.0058 [-0.0008, 0.0124] | inconclusive | 0.0020 [0.0002, 0.0038] | helps | -0.0001 [-0.0309, 0.0307] | inconclusive | 0.2960 [0.2551, 0.3369] | helps | 12 |
| C3 | 0.9 | 0.0041 [-0.0013, 0.0095] | inconclusive | 0.0008 [-0.0007, 0.0024] | inconclusive | -0.0124 [-0.0469, 0.0222] | inconclusive | 0.1555 [0.1154, 0.1955] | helps | 12 |

