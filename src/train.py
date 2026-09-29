"""Entry script 2/6: train the models using the tuned hyperparameters.

Reads outputs/<model>/best_params.json and, for each model,

  * fits on the training rows and scores the held-out rows;
  * runs plain CV out-of-fold prediction (all four models);
  * runs nested CV out-of-fold prediction for the nested models, reusing the
    per-fold parameters stored during tuning so no search is repeated;
  * plots RMSE / Pearson r / R2 and writes metrics.json plus a deployment model
    refitted on all rows.

Usage:
    conda activate rdkit
    python src/train.py
    python src/train.py --models random_cv
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
from evaluation import holdout_predictions, nested_oof_predictions, oof_predictions
from log_utils import install_excepthook, setup_logger
from metrics_utils import regression_metrics, to_native
from model_utils import build_pipeline_from_params, save_deployment_model
from tuning import load_params


def extract_fold_params(key: str, fold_records: list[dict], params: dict) -> list[dict]:
    """Match the deployed hyperparameters against the per-fold tuning records.

    A bare KeyError here means best_params.json and nested_folds.json come from
    two tuning runs that used different search spaces, which is only fixable by
    re-running the tuning stage.
    """
    missing = sorted({k for k in params if any(k not in rec for rec in fold_records)})
    if missing:
        raise ValueError(
            f"[{key}] nested_folds.json does not contain {missing}; best_params.json and "
            "nested_folds.json come from different tuning runs. Re-run "
            "src/hyperparam_tuning.py for this model.")
    return [{k: rec[k] for k in params} for rec in fold_records]


def train_single(spec: cfg.ModelSpec, X, y, ctrl, groups, feature_names, seed: int,
                 logger) -> dict:
    bundle = load_params(cfg.params_path(spec.key))
    params = bundle["params"]
    logger.info(">>> [%s] %s | reading parameters from %s", spec.key, spec.label,
                cfg.params_path(spec.key).name)
    logger.info("    selection: %s", bundle.get("param_selection", "-"))

    tr, te = split_train_test(len(y), spec.split, groups, cfg.TEST_SIZE, seed)
    X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
    folds = make_folds(spec.split, X_tr, y_tr, g_tr, cfg.N_SPLITS_OUTER, seed)

    y_tr_true, y_tr_pred, y_te_true, y_te_pred = holdout_predictions(
        params, X, y, tr, te, seed=seed)
    train_metrics = regression_metrics(y_tr_true, y_tr_pred)
    test_metrics = regression_metrics(y_te_true, y_te_pred)

    cv_true, cv_pred, cv_folds = oof_predictions(params, X_tr, y_tr, folds, seed=seed)
    cv_metrics = regression_metrics(cv_true, cv_pred)

    nested_block = None
    if spec.is_nested:
        nf_path = cfg.nested_folds_path(spec.key)
        if not nf_path.is_file():
            logger.warning("[%s] %s missing, skipping nested CV", spec.key, nf_path.name)
        else:
            recs = sorted(json.loads(nf_path.read_text(encoding="utf-8"))["folds"],
                          key=lambda r: r["fold"])
            fold_params = extract_fold_params(spec.key, recs, params)
            n_true, n_pred, n_folds = nested_oof_predictions(
                fold_params, X_tr, y_tr, folds, seed=seed)
            nested_metrics = regression_metrics(n_true, n_pred)
            for rec, fm in zip(recs, n_folds):
                fm["inner_cv_rmse"] = rec["inner_cv_rmse"]
            nested_block = {"metrics": nested_metrics, "per_fold": n_folds,
                            "oof_true": n_true, "oof_pred": n_pred}
            logger.info("    nested CV: RMSE=%.4f  r=%.4f  R2=%.4f",
                        nested_metrics["RMSE"], nested_metrics["Pearson_r"],
                        nested_metrics["R2"])

    deploy_pipe = build_pipeline_from_params(params, seed=seed)
    deploy_pipe.fit(X, y)
    save_deployment_model(deploy_pipe, {
        "params": params,
        "feature_names": feature_names,
        "model_key": spec.key,
        "split": spec.split,
        "cv_protocol": spec.cv,
        "label": spec.label,
        "target": cfg.DELTA_COL,
        "control_col": cfg.CONTROL_COL,
        "reconstruction": cfg.RECONSTRUCTION,
        "seed": seed,
        "train_index": tr.tolist(),
        "test_index": te.tolist(),
        "cv_metrics": cv_metrics,
        "nested_metrics": nested_block["metrics"] if nested_block else None,
        "test_metrics": test_metrics,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }, cfg.model_path(spec.key), cfg.model_bundle_path(spec.key))

    result = {
        "model": spec.key,
        "label": spec.label,
        "short": spec.short,
        "split": spec.split,
        "protocol": spec.cv,
        "params": params,
        "param_selection": bundle.get("param_selection", ""),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_features": int(X.shape[1]),
        "train": train_metrics,
        "test": test_metrics,
        "cv": cv_metrics,
        "cv_per_fold": cv_folds,
        "nested": nested_block["metrics"] if nested_block else None,
        "nested_per_fold": nested_block["per_fold"] if nested_block else None,
        "oof": {
            "cv_true": cv_true, "cv_pred": cv_pred,
            "nested_true": nested_block["oof_true"] if nested_block else None,
            "nested_pred": nested_block["oof_pred"] if nested_block else None,
            "test_true": y_te_true, "test_pred": y_te_pred,
            "train_true": y_tr_true, "train_pred": y_tr_pred,
        },
        "split_index": {"train": tr.tolist(), "test": te.tolist()},
        "seed": seed,
    }
    logger.info("    Train RMSE=%.4f r=%.4f R2=%.4f | Test RMSE=%.4f r=%.4f R2=%.4f",
                train_metrics["RMSE"], train_metrics["Pearson_r"], train_metrics["R2"],
                test_metrics["RMSE"], test_metrics["Pearson_r"], test_metrics["R2"])
    logger.info("    CV    RMSE=%.4f r=%.4f R2=%.4f",
                cv_metrics["RMSE"], cv_metrics["Pearson_r"], cv_metrics["R2"])
    return result


def plot_model(spec: cfg.ModelSpec, res: dict, ctrl: np.ndarray, logger) -> None:
    fig_dir = cfg.figure_dir(spec.key)
    oof = res["oof"]
    short = spec.short
    te = np.asarray(res["split_index"]["test"], dtype=int)

    plotting.plot_train_test_parity(
        oof["train_true"], oof["train_pred"], oof["test_true"], oof["test_pred"],
        fig_dir / "parity_train_test.png", f"{short}: measured vs predicted")
    plotting.plot_parity(oof["cv_true"], oof["cv_pred"], fig_dir / "parity_cv.png",
                         f"{short}: plain CV out-of-fold")
    if oof.get("nested_true") is not None:
        plotting.plot_parity(oof["nested_true"], oof["nested_pred"],
                             fig_dir / "parity_nested.png",
                             f"{short}: nested CV out-of-fold (unbiased)",
                             color=plotting.NESTED_COLOR)

    ctrl_te = ctrl[te]
    plotting.plot_pce_parity(oof["test_true"] + ctrl_te, oof["test_pred"] + ctrl_te,
                             fig_dir / "parity_pce_test.png",
                             f"{short}: test set on the PCE scale")

    groups = {"Train": res["train"], "Test": res["test"]}
    groups[f"{spec.fold_name} CV"] = res["cv"]
    if res["nested"]:
        groups["Nested CV"] = res["nested"]
    plotting.plot_metric_bars(groups, fig_dir / "metrics_bars.png",
                              f"{short}: RMSE / Pearson r / R2")

    plotting.plot_fold_metrics(pd.DataFrame(res["cv_per_fold"]),
                               fig_dir / "metrics_cv_folds.png",
                               f"{short}: plain CV, per fold")

    if res["nested_per_fold"]:
        ndf = pd.DataFrame(res["nested_per_fold"])
        plotting.plot_fold_metrics(ndf, fig_dir / "metrics_nested_folds.png",
                                   f"{short}: nested CV, per fold")
        if "inner_cv_rmse" in ndf.columns:
            plotting.plot_nested_fold_scores(
                ndf, fig_dir / "nested_inner_vs_outer.png",
                f"{short}: inner objective vs outer RMSE")

    logger.info("    figures -> %s", fig_dir)


def build_summary_row(res: dict) -> dict:
    row = {"model": res["model"], "label": res["label"],
           "split": res["split"], "cv": res["protocol"]}
    for tag, block in (("train", res["train"]), ("test", res["test"])):
        for m in cfg.PRIMARY_METRICS:
            row[f"{m}_{tag}"] = round(block[m], 4)
    for m in cfg.PRIMARY_METRICS:
        row[f"{m}_cv"] = round(res["cv"][m], 4)
        nv = res["nested"][m] if res["nested"] else float("nan")
        row[f"{m}_nested"] = round(nv, 4) if nv == nv else np.nan
    folds = pd.DataFrame(res["cv_per_fold"])
    for m in cfg.PRIMARY_METRICS:
        row[f"{m}_cv_std"] = round(float(folds[m].std(ddof=1)), 4)
    return row


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the four XGBoost models from json parameters")
    p.add_argument("--models", nargs="*", default=None, choices=cfg.MODEL_KEYS)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logger = setup_logger(cfg.OUTPUT_DIR / "train.log")
    install_excepthook(logger)

    df = load_dataset()
    X, y, ctrl, groups, feats = build_matrices(df)
    logger.info("data: n=%d, features=%d, unique molecules=%d, target=%s",
                len(y), X.shape[1], pd.Series(groups).nunique(), cfg.DELTA_COL)

    t_all = time.time()
    rows = []
    for key in (args.models or cfg.MODEL_KEYS):
        spec = cfg.SPEC_BY_KEY[key]
        cfg.ensure_model_dirs(key)
        t0 = time.time()
        try:
            res = train_single(spec, X, y, ctrl, groups, feats, args.seed, logger)
            plot_model(spec, res, ctrl, logger)
            res["elapsed_s"] = time.time() - t0
            payload = {k: v for k, v in res.items() if k != "oof"}
            payload["oof"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                              for k, v in res["oof"].items()}
            cfg.metrics_path(key).write_text(json.dumps(to_native(payload), indent=2),
                                             encoding="utf-8")
            pd.DataFrame(res["cv_per_fold"]).to_csv(cfg.folds_path(key), index=False)
            rows.append(build_summary_row(res))
            logger.info("    done in %.1fs -> %s", res["elapsed_s"], cfg.metrics_path(key).name)
        except Exception:
            logger.exception("[FAILED] %s", key)

    if rows:
        pd.DataFrame(rows).to_csv(cfg.OUTPUT_DIR / "train_summary.csv", index=False)
        logger.info("training summary")
        for line in pd.DataFrame(rows).to_string(index=False).splitlines():
            logger.info(line)
    logger.info("total elapsed %.1fs", time.time() - t_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
