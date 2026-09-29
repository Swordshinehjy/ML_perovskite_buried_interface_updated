# Comparison of the four XGBoost models

- Target: `delta_PCE = PCE - control_PCE`; reconstruction `PCE = delta_PCE + control_PCE`
- `*_cv`     : plain CV out-of-fold estimate, tuned on the same folds, **optimistic**
- `*_nested` : nested CV out-of-fold estimate, outer folds never used for tuning, **unbiased**
- `*_test`   : held-out rows removed before training, identical protocol for all models

## 1. Primary metrics (RMSE / Pearson r / R2)

| label | RMSE_cv | RMSE_nested | RMSE_test | Pearson_r_cv | Pearson_r_nested | Pearson_r_test | R2_cv | R2_nested | R2_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random split + CV | 1.1973 |  | 1.5652 | 0.3499 |  | 0.3004 | 0.1211 |  | 0.0571 |
| Random split + NestedCV | 1.2633 | 1.2765 | 1.6079 | 0.2535 | 0.1672 | 0.275 | 0.0216 | 0.0009 | 0.0049 |
| Group split + CV | 1.2691 |  | 1.438 | 0.2729 |  | 0.1295 | 0.0738 |  | 0.0107 |
| Group split + NestedCV | 1.3144 | 1.3678 | 1.5179 | 0.2244 | 0.114 | 0.0554 | 0.0065 | -0.076 | -0.1022 |

## 2. Between-fold standard deviation (stability)

| label | RMSE_cv_std | Pearson_r_cv_std | R2_cv_std | RMSE_nested_std | Pearson_r_nested_std | R2_nested_std |
| --- | --- | --- | --- | --- | --- | --- |
| Random split + CV | 0.2913 | 0.1673 | 0.183 |  |  |  |
| Random split + NestedCV | 0.2947 | 0.1759 | 0.1968 | 0.3009 | 0.2038 | 0.1638 |
| Group split + CV | 0.2195 | 0.1506 | 0.1202 |  |  |  |
| Group split + NestedCV | 0.2116 | 0.1605 | 0.3682 | 0.1944 | 0.1615 | 0.4784 |

## 3. Optimism of plain CV (nested CV as reference)

| split | gap_RMSE | plain_RMSE | nested_RMSE | gap_Pearson_r | plain_Pearson_r | nested_Pearson_r | gap_R2 | plain_R2 | nested_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 0.0793 | 1.1973 | 1.2765 | 0.1828 | 0.3499 | 0.1672 | 0.1202 | 0.1211 | 0.0009 |
| group | 0.0987 | 1.2691 | 1.3678 | 0.1589 | 0.2729 | 0.114 | 0.1497 | 0.0738 | -0.076 |

The gap is sign-corrected: positive always means plain CV looks better (nested - plain for RMSE, plain - nested for r and R2).