# ML_perovskite_buried_interface_updated

Machine-learning modelling and screening of molecules for buried-interface
perovskite solar cells. Four **XGBRegressor** models are built and compared
systematically across **two data-splitting strategies x two cross-validation
protocols**, with hyperparameter tuning, SHAP interpretation and a
Y-scrambling validity check.

Cooperated with Qi Zhang, NWPU.

## Notes

* A few dozen new molecules have been added to the new dataset compared with [https://github.com/Swordshinehjy/ML_perovskite_buired_interface](https://github.com/Swordshinehjy/ML_perovskite_buired_interface). If a molecule has multiple possible coordination sites, the original dataset labeled only the most important one, whereas the new dataset labels all of them as 1. Theoretically, heteroatoms that can coordinate with Pb in perovskite are the main reason for performance improvement, so this project lists the heteroatoms that play a role in the coordinating functional groups, rather than using classic molecular fingerprints. Considering that papers are subject to survivor effects (published papers are all better than the reference and negative examples are severely lacking), it is difficult for the model to learn relevant effective information during training. Notably, the control performance itself is highly linearly correlated with the final performance, and using control as a feature may lead to data leakage, causing the model to take shortcuts. Therefore, this scheme adopts delta-learning, that is, it learns the difference. Literature data in the perovskite field are very noisy (it is even difficult to judge how much trickery there is in these papers), so the ML results of this project are for reference only. The conclusions are basically consistent with the original paper: strongly polar functional groups contribute mainly to performance.
* Because a few molecular structures are duplicated (but with different device parameters), there is molecular structure leakage. This repository compares random and Group split, and also compares traditional CV (with slight hyperparameter tuning leakage) and nested CV, for a total of four models. It appears that the metrics become worse after adding new entries to the dataset. The model did not find chemical rules, but may have overfitted to the lab noise, assuming that the literature data is authentic and reliable without data manipulation. 
* At present, data leakage is almost everywhere in materials science ML papers. If you are doing related research, please be sure to pay attention to this issue.


## 1. Environment

```bash
conda create -n rdkit -c conda-forge python=3.13 rdkit xgboost scikit-learn optuna shap matplotlib scipy pandas pytest
conda activate rdkit
```

All paths are resolved from the file location, so the working directory does not
matter.


## 2. Quick start

```bash
python cli.py all                 # tune -> train -> predict -> shap -> scramble -> compare
python cli.py all --help          # every knob and every stage
python cli.py train               # one stage only
python cli.py tune --models group_nested --trials 200
```

The individual scripts under `src/` accept the same arguments and can be run on
their own (`python src/train.py --help`).

Reference timings on a laptop CPU: `tune` ~11 min, `train` <1 min, `predict` <1 min,
`shap` ~15 s, `scramble` (200 permutations x 4 models) ~12 min, `compare` <1 min.


## 3. Modelling setup

| Item | Value |
| --- | --- |
| Target | `delta_PCE = PCE - control_PCE` (the molecular gain over the batch control) |
| Reconstruction | `PCE = delta_PCE + control_PCE` |
| Features | the 27 columns of `dataset.csv` other than `SMILES` and `PCE` (`control_PCE` kept as the baseline feature) |
| Estimator | `XGBRegressor(objective="reg:squarederror")` with `StandardScaler` inside a `Pipeline` |
| Tuning | Optuna (TPE), objective = mean in-fold RMSE |
| Metrics | **RMSE / Pearson r / R2** (plus MAE and Spearman rho) |

Optionally place an `xgb_search_space.json` at the repository root to override
individual ranges of the built-in search space; it is applied when
`src/model_utils.py` is imported, before any other module binds the space.

Predicting absolute PCE directly would be misleading: `corr(PCE, control_PCE) = 0.93`,
so the model would mostly copy the control. The target is therefore the gain,
and metrics are reported additionally on the reconstructed PCE scale.

### The four models

| Directory | Split | CV protocol | Meaning |
| --- | --- | --- | --- |
| `random_cv` | random | plain CV (KFold) | replicates of one molecule may straddle train/test |
| `random_nested` | random | nested CV | unbiased generalisation estimate |
| `group_cv` | **grouped by canonical SMILES** | plain CV (GroupKFold) | **every replicate of a molecule stays on one side** |
| `group_nested` | grouped | nested CV | unbiased estimate under the strict splitting rule |

* **Group splitting.** RDKit canonicalises `SMILES` into a group id, then
  `GroupShuffleSplit` / `GroupKFold` use it. The dataset has 222 rows over 196
  unique molecules, with up to 5 replicates of one molecule, so a random split
  leaks information between twins.
* **Plain vs nested CV.** Plain CV tunes and reports on the same folds, hence
  optimistically biased. In nested CV each outer fold is scored by a model whose
  hyperparameters were tuned only on that fold's outer-train part. The difference
  between the two is quantified in `optimism_gap.csv`.


## 4. Project layout

```
cli.py                        command line entry point (see its docstring for usage)
src/
  config.py                   paths, seed, target definition, the four ModelSpecs
  log_utils.py                logger assembly (file + console, UTF-8)
  data_utils.py               loading, canonical-SMILES grouping, splits, folds
  model_utils.py              search space, parameter sampling, estimator/pipeline factory
  metrics_utils.py            RMSE / r / R2 / Spearman and permutation statistics
  evaluation.py               plain CV, nested CV and held-out test protocols
  tuning.py                   Optuna search + parameter json I/O + importance
  plotting.py                 every figure
  hyperparam_tuning.py  (1)   tuning -> best_params.json + tuning figures
  train.py              (2)   read json -> train -> metrics + figures + model.json
  predict_new.py        (3)   screening-table predictions
  shap_analysis.py      (4)   TreeExplainer SHAP interpretation
  y_scrambling.py       (5)   response-permutation validity test
  compare_models.py     (6)   cross-model comparison
tests/                        pytest suite reproducing the reviewed bugs
```

Outputs (one folder per model):

```
outputs/<model>/
  best_params.json            hyperparameters used by train.py
  nested_folds.json           per-outer-fold parameters (nested models)
  tuning_trials.csv           one row per Optuna trial
  metrics.json model.json     metrics (incl. out-of-fold arrays) / deployment
  model_bundle.json           model (XGBoost native JSON) + scaler/metadata sidecar
  cv_folds.csv new_molecule_predictions.csv
  y_scrambling.csv y_scrambling_summary.json
  shap/                       shap_importance.csv shap_values.csv shap_summary.json
  figures/                    parity, metric bars, per-fold, tuning, scrambling, SHAP
outputs/_comparison/          model_comparison.csv optimism_gap.csv
                              y_scrambling_summary.csv shap_consensus_ranking.csv
                              new_molecule_*_all_models.csv comparison_report.md
                              figures/*.png
outputs/tuning_summary.csv train_summary.csv *.log
```

Deployment models are stored in a code-free format: the booster is written with
XGBoost's native JSON serialisation (`model.json`) and the fitted
`StandardScaler` statistics plus the metadata go into a plain JSON sidecar
(`model_bundle.json`). No pickle/joblib artefacts are produced, so loading a
model cannot execute arbitrary code.


## 5. Reading the results

* `*_cv` is the tuning objective value, **not** an unbiased generalisation
  estimate. Report `*_nested` (or the held-out test set) instead.
* `optimism_gap.csv` is sign-corrected so a positive value always means "plain CV
  looks better": `nested - plain` for RMSE, `plain - nested` for r and R2.
* Metrics under the group split are usually **lower** than under the random split;
  the difference measures how much the random split was profiting from replicate
  leakage.
* In y-scrambling the smallest attainable p is `1 / (n_scrambles + 1)`, i.e.
  about 0.005 for 200 permutations. The real model sits 3-5 standard deviations
  above the null distribution for Pearson r, so the learned relationship is real.
* In SHAP output `mean_abs_shap` ranks global influence and
  `mean_shap_signed_corr` gives the direction (positive = the feature raises the
  predicted gain). `control_PCE`, `surface_max`, `HOMO_calc` and `MPI` dominate.
* On the `delta_PCE` scale the parity regression slope is necessarily below 1
  (regression towards the mean); that is a property of an RMSE-optimal predictor,
  not a defect. Judge ranking ability by Pearson r / Spearman rho.

## 6. Data

* `dataset.csv` - 222 buried-interface molecular records (`SMILES`, `PCE`,
  `control_PCE`, 27 physicochemical/structural features).
* `new-mol.csv` - 9 candidate molecules for screening (no `PCE` column;
  `control_PCE` present for reconstructing the absolute efficiency).
