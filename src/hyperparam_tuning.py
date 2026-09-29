"""Entry script 1/6: hyperparameter tuning.

Runs an Optuna search for each of the four models (random / group split x
plain CV / nested CV), writes outputs/<model>/best_params.json (plus
nested_folds.json for the nested models) and the tuning diagnostics plots.
This script does not train the final models; train.py reads the json instead.

How the four models are tuned:

  * *_cv     : one complete search on the training split, objective = mean
               in-fold RMSE of that split's CV (KFold or GroupKFold). The best
               trial becomes the deployment hyperparameter set.
  * *_nested : a nested CV run, i.e. an independent search inside every outer
               fold (that fold's outer-test rows never take part). The deployed
               set is the one chosen in the outer fold with the lowest inner CV
               RMSE; all per-fold choices go to nested_folds.json so train.py
               can reproduce the unbiased out-of-fold predictions without
               repeating a single Optuna trial.

Usage:
    conda activate rdkit
    python src/hyperparam_tuning.py
    python src/hyperparam_tuning.py --models group_cv
    python src/hyperparam_tuning.py --trials 150 --inner-trials 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import plotting
from data_utils import build_matrices, load_dataset, make_folds, split_train_test
from log_utils import install_excepthook, setup_logger
from model_utils import SEARCH_SPACE
from tuning import (
    make_params_payload,
    param_importance_from_trials,
    run_optuna_search,
    save_params,
    trials_to_frame,
)


def tune_single(spec: cfg.ModelSpec, X, y, groups, seed: int, n_trials: int,
                n_inner_trials: int, logger) -> dict:
    """Run one model's tuning protocol and persist the result."""
    tr, te = split_train_test(len(y), spec.split, groups, cfg.TEST_SIZE, seed)
    X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
    outer_folds = make_folds(spec.split, X_tr, y_tr, g_tr, cfg.N_SPLITS_OUTER, seed)

    logger.info(">>> [%s] %s | train=%d test=%d | outer folds=%d",
                spec.key, spec.label, len(tr), len(te), len(outer_folds))

    if not spec.is_nested:
        study = run_optuna_search(
            X_tr, y_tr, outer_folds, n_trials, seed=seed, study_name=f"{spec.key}_tuning",
            progress=lambda i, n, b: logger.info("    trial %3d/%d | best CV RMSE = %.4f", i, n, b)
            if (i % 25 == 0 or i == n) else None,
        )
        best_params = dict(study.best_params)
        best_cv_rmse = float(study.best_value)
        selection = (f"single Optuna search ({n_trials} trials) on the training split; "
                     f"objective = mean {spec.fold_name} RMSE")
        trials_df = trials_to_frame(study)
        trials_df.insert(0, "outer_fold", 0)
        trials_df["trial_global"] = trials_df["trial"]
        trials_df.to_csv(cfg.tuning_trials_path(spec.key), index=False)
        logger.info("    best CV RMSE = %.4f | params = %s", best_cv_rmse, best_params)
    else:
        fold_records, all_trials = [], []
        for k, (otr, ote) in enumerate(outer_folds):
            inner_folds = make_folds(spec.split, X_tr[otr], y_tr[otr], g_tr[otr],
                                     cfg.N_SPLITS_INNER, seed)
            study = run_optuna_search(X_tr[otr], y_tr[otr], inner_folds, n_inner_trials,
                                      seed=seed, study_name=f"{spec.key}_outer{k + 1}")
            rec = {"fold": k + 1, "n_train": int(len(otr)), "n_test": int(len(ote)),
                   "inner_cv_rmse": float(study.best_value)}
            rec.update(dict(study.best_params))
            fold_records.append(rec)
            tdf = trials_to_frame(study)
            tdf.insert(0, "outer_fold", k + 1)
            # Global trial index: each fold restarts at 1, so the concatenated
            # history needs an offset to stay monotonic along the x axis.
            tdf["trial_global"] = tdf["trial"] + k * n_inner_trials
            all_trials.append(tdf)
            logger.info("    outer fold %d/%d | inner best CV RMSE = %.4f | n_train=%d n_test=%d",
                        k + 1, len(outer_folds), study.best_value, len(otr), len(ote))

        fold_df = pd.DataFrame(fold_records)
        best_row = fold_df.loc[fold_df["inner_cv_rmse"].idxmin()]
        best_params = {
            name: (int(best_row[name]) if sp.get("type") == "int" else float(best_row[name]))
            for name, sp in SEARCH_SPACE.items()
        }
        best_cv_rmse = float(best_row["inner_cv_rmse"])
        selection = (f"nested CV: independent Optuna search per outer fold "
                     f"({n_inner_trials} trials each); deployed params taken from outer fold "
                     f"{int(best_row['fold'])} (lowest inner CV RMSE = {best_cv_rmse:.4f})")

        pd.concat(all_trials, ignore_index=True).to_csv(cfg.tuning_trials_path(spec.key), index=False)
        save_params(cfg.nested_folds_path(spec.key), {
            "model": spec.key,
            "split": spec.split,
            "n_outer_folds": len(fold_records),
            "n_inner_trials": n_inner_trials,
            "seed": seed,
            "folds": fold_records,
        })
        logger.info("    deployed params from outer fold %d (inner CV RMSE %.4f): %s",
                    int(best_row["fold"]), best_cv_rmse, best_params)

    payload = make_params_payload(
        spec.key, best_params, split=spec.split, cv=spec.cv,
        best_cv_rmse=best_cv_rmse,
        n_trials=n_trials if not spec.is_nested else n_inner_trials * len(outer_folds),
        seed=seed, selection=selection,
    )
    payload.update({
        "label": spec.label,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_outer_folds": len(outer_folds),
        "folds_note": f"{spec.fold_name}({cfg.N_SPLITS_OUTER})",
    })
    save_params(cfg.params_path(spec.key), payload)
    logger.info("    saved -> %s", cfg.params_path(spec.key).name)
    return payload


def plot_tuning(spec: cfg.ModelSpec, logger) -> None:
    """Produce every tuning figure for one model."""
    fig_dir = cfg.figure_dir(spec.key)
    trials_path = cfg.tuning_trials_path(spec.key)
    if not trials_path.is_file():
        logger.warning("[%s] tuning_trials.csv missing, skipping tuning figures", spec.key)
        return

    trials = pd.read_csv(trials_path)
    short = spec.short
    plotting.plot_optimization_history(trials, fig_dir / "tuning_history.png",
                                       f"{short}: Optuna search history")

    try:
        plotting.plot_trial_param_scatter(trials, fig_dir / "tuning_param_scatter.png",
                                          f"{short}: CV RMSE vs hyperparameters")
    except Exception as exc:
        logger.warning("[%s] parameter scatter failed: %s", spec.key, exc)

    try:
        importance = param_importance_from_trials(trials)
        plotting.plot_param_importance(importance, fig_dir / "tuning_param_importance.png",
                                       f"{short}: hyperparameter importance")
    except Exception as exc:
        logger.warning("[%s] parameter importance failed: %s", spec.key, exc)

    if spec.is_nested:
        folds_path = cfg.nested_folds_path(spec.key)
        if folds_path.is_file():
            recs = json.loads(folds_path.read_text(encoding="utf-8"))["folds"]
            fold_df = pd.DataFrame(recs)
            plotting.plot_nested_fold_params(
                fold_df, fig_dir / "tuning_nested_fold_params.png",
                f"{short}: hyperparameters chosen per outer fold")
            plotting.plot_nested_fold_scores(
                fold_df, fig_dir / "tuning_nested_inner_scores.png",
                f"{short}: inner tuning objective per outer fold")

    logger.info("    figures -> %s", fig_dir)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune the four XGBoost models")
    p.add_argument("--models", nargs="*", default=None, choices=cfg.MODEL_KEYS)
    p.add_argument("--trials", type=int, default=cfg.N_TRIALS)
    p.add_argument("--inner-trials", type=int, default=cfg.N_TRIALS_INNER)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logger = setup_logger(cfg.OUTPUT_DIR / "hyperparam_tuning.log")
    install_excepthook(logger)

    df = load_dataset()
    X, y, ctrl, groups, feats = build_matrices(df)
    logger.info("data: n=%d, features=%d, unique molecules=%d",
                len(y), X.shape[1], pd.Series(groups).nunique())
    logger.info("target: %s = %s - %s (mean=%.3f, std=%.3f)",
                cfg.DELTA_COL, cfg.TARGET_COL, cfg.CONTROL_COL, y.mean(), y.std())
    logger.info("trials=%d, inner_trials=%d, seed=%d", args.trials, args.inner_trials, args.seed)

    specs = [cfg.SPEC_BY_KEY[k] for k in (args.models or cfg.MODEL_KEYS)]
    t_all = time.time()
    summaries = []
    for spec in specs:
        cfg.ensure_model_dirs(spec.key)
        t0 = time.time()
        try:
            payload = tune_single(spec, X, y, groups, args.seed, args.trials,
                                  args.inner_trials, logger)
            plot_tuning(spec, logger)
            payload["elapsed_s"] = time.time() - t0
            payload["label"] = spec.label
            summaries.append(payload)
            logger.info("    done in %.1fs", payload["elapsed_s"])
        except Exception:
            logger.exception("[FAILED] %s", spec.key)

    rows = [{
        "model": s["model"], "label": s["label"], "split": s["split"], "cv": s["cv"],
        "best_cv_rmse": round(s["best_cv_rmse"], 4),
        "n_trials": s["n_trials"], "elapsed_s": round(s["elapsed_s"], 1),
        **{f"p_{k}": v for k, v in s["params"].items()},
    } for s in summaries]
    summary = pd.DataFrame(rows)
    summary.to_csv(cfg.OUTPUT_DIR / "tuning_summary.csv", index=False)

    logger.info("tuning summary (best_cv_rmse is the search objective, not an unbiased estimate)")
    for line in summary.drop(columns=[c for c in summary.columns if c.startswith("p_")]).to_string(index=False).splitlines():
        logger.info(line)
    logger.info("parameters -> outputs/<model>/best_params.json, summary -> tuning_summary.csv")
    logger.info("total elapsed %.1fs", time.time() - t_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
