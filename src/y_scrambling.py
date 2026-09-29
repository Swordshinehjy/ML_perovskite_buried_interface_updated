"""Entry script 5/6: Y-scrambling (response permutation test).

Keeps the design matrix and the folds fixed and randomly permutes only the
training target, then rebuilds the out-of-fold predictions. If the real model is
clearly better than the permuted null distribution, the model captured a genuine
X-y relationship rather than a chance correlation.

Protocol note: the observed statistic is the plain-CV out-of-fold performance of
the *deployed* hyperparameter set on the model's own folds (KFold for a random
split, GroupKFold for a group split). This is deliberately the same computation
as the `cv` block in train.py. For nested models it therefore does **not** reuse
the per-outer-fold parameters from nested_folds.json: a permutation test must
hold the fitting procedure fixed and vary only y, so one consistent protocol is
used for all four models. The `cv_protocol` field below identifies the model, it
does not describe this statistic; see `observed_basis`.

Usage:
    conda activate rdkit
    python src/y_scrambling.py
    python src/y_scrambling.py --n-scrambles 500 --models random_cv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import plotting
from data_utils import build_matrices, load_dataset, make_folds, split_train_test
from evaluation import oof_predictions
from log_utils import install_excepthook, setup_logger
from metrics_utils import permutation_pvalue, regression_metrics, to_native, z_score
from tuning import load_params

HIGHER_IS_BETTER = ("Pearson_r", "R2")


def scramble_one(spec: cfg.ModelSpec, X, y, groups, seed: int, n_scrambles: int,
                 logger) -> dict:
    params = load_params(cfg.params_path(spec.key))["params"]
    tr, _ = split_train_test(len(y), spec.split, groups, cfg.TEST_SIZE, seed)
    X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
    folds = make_folds(spec.split, X_tr, y_tr, g_tr, cfg.N_SPLITS_OUTER, seed)

    logger.info(">>> [%s] %s | %d training rows | %d folds | %d permutations",
                spec.key, spec.label, len(tr), len(folds), n_scrambles)

    yt, yp, _ = oof_predictions(params, X_tr, y_tr, folds, seed=seed)
    observed = regression_metrics(yt, yp)
    logger.info("    observed: RMSE=%.4f  r=%.4f  R2=%.4f",
                observed["RMSE"], observed["Pearson_r"], observed["R2"])

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_scrambles):
        y_scr = rng.permutation(y_tr)
        _, yp_s, _ = oof_predictions(params, X_tr, y_scr, folds, seed=seed)
        m = regression_metrics(y_scr, yp_s)
        m["scramble"] = i + 1
        rows.append(m)
        if (i + 1) % 50 == 0:
            logger.info("    scrambled %d/%d", i + 1, n_scrambles)

    null_df = pd.DataFrame(rows)
    null_df.to_csv(cfg.scramble_csv_path(spec.key), index=False)

    stats_ = {}
    for metric in HIGHER_IS_BETTER:
        null = null_df[metric].to_numpy(dtype=float)
        stats_[f"p_{metric}"] = permutation_pvalue(observed[metric], null, lower_is_better=False)
        stats_[f"z_{metric}"] = z_score(observed[metric], null)
        stats_[f"null_mean_{metric}"] = float(np.mean(null))
        stats_[f"null_std_{metric}"] = float(np.std(null, ddof=1))
    null_rmse = null_df["RMSE"].to_numpy(dtype=float)
    stats_["p_RMSE"] = permutation_pvalue(observed["RMSE"], null_rmse, lower_is_better=True)
    stats_["z_RMSE"] = z_score(observed["RMSE"], null_rmse)
    stats_["null_mean_RMSE"] = float(np.mean(null_rmse))
    stats_["null_std_RMSE"] = float(np.std(null_rmse, ddof=1))
    stats_["baseline_rmse"] = float(np.std(y_tr))

    summary = {
        "model": spec.key,
        "label": spec.label,
        "split": spec.split,
        "cv_protocol": spec.cv,
        "fold_name": spec.fold_name,
        "observed_basis": ("plain CV out-of-fold with the deployed hyperparameters; "
                           "for nested models this is the train.py 'cv' block, not the "
                           "per-outer-fold nested parameters"),
        "n_scrambles": n_scrambles,
        "n_train": int(len(tr)),
        "folds": len(folds),
        "params": params,
        "observed": observed,
        "stats": stats_,
    }
    cfg.scramble_summary_path(spec.key).write_text(
        json.dumps(to_native(summary), indent=2), encoding="utf-8")

    logger.info("    null r    = %.4f +/- %.4f | observed r = %.4f | p = %.5f",
                stats_["null_mean_Pearson_r"], stats_["null_std_Pearson_r"],
                observed["Pearson_r"], stats_["p_Pearson_r"])
    logger.info("    null RMSE = %.4f +/- %.4f | observed   = %.4f | p = %.5f",
                stats_["null_mean_RMSE"], stats_["null_std_RMSE"],
                observed["RMSE"], stats_["p_RMSE"])

    fig_dir = cfg.figure_dir(spec.key)
    plotting.plot_y_scrambling(null_df, observed, stats_, fig_dir / "y_scrambling.png",
                               f"{spec.short}: Y-scrambling ({n_scrambles} permutations)")
    plotting.plot_scrambling_convergence(
        null_df, observed, fig_dir / "y_scrambling_convergence.png",
        f"{spec.short}: permutation stability")
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Y-scrambling test for the four models")
    p.add_argument("--models", nargs="*", default=None, choices=cfg.MODEL_KEYS)
    p.add_argument("--n-scrambles", type=int, default=cfg.N_SCRAMBLES)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logger = setup_logger(cfg.OUTPUT_DIR / "y_scrambling.log")
    install_excepthook(logger)

    df = load_dataset()
    X, y, ctrl, groups, feats = build_matrices(df)

    summaries = []
    for key in (args.models or cfg.MODEL_KEYS):
        spec = cfg.SPEC_BY_KEY[key]
        cfg.ensure_model_dirs(key)
        try:
            summaries.append(scramble_one(spec, X, y, groups, args.seed,
                                          args.n_scrambles, logger))
        except Exception:
            logger.exception("[FAILED] %s", key)

    if summaries:
        rows = []
        for s in summaries:
            st, ob = s["stats"], s["observed"]
            rows.append({
                "model": s["model"], "label": s["label"],
                "n_scrambles": s["n_scrambles"],
                "obs_RMSE": round(ob["RMSE"], 4),
                "null_RMSE": round(st["null_mean_RMSE"], 4),
                "p_RMSE": st["p_RMSE"], "z_RMSE": round(st["z_RMSE"], 2),
                "obs_Pearson_r": round(ob["Pearson_r"], 4),
                "null_Pearson_r": round(st["null_mean_Pearson_r"], 4),
                "p_Pearson_r": st["p_Pearson_r"], "z_Pearson_r": round(st["z_Pearson_r"], 2),
                "obs_R2": round(ob["R2"], 4),
                "null_R2": round(st["null_mean_R2"], 4),
                "p_R2": st["p_R2"],
            })
        summary = pd.DataFrame(rows)
        out = cfg.COMPARISON_DIR / "y_scrambling_summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        logger.info("Y-scrambling summary")
        for line in summary.to_string(index=False).splitlines():
            logger.info(line)
        logger.info("saved -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
