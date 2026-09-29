"""Evaluation protocols: plain CV, nested CV and held-out test assessment.

Single responsibility: decide *how* performance is estimated. No tuning, no plotting.

Why the three estimates differ (this is what the project compares):

  * plain CV  : tune the hyperparameters on the full training set, then predict
                out-of-fold on that same set of folds. The tuning already saw
                those labels, so the score is optimistically biased.
  * nested CV : each outer fold is predicted by a model whose hyperparameters
                were tuned exclusively on that fold's outer-train part, so the
                concatenated out-of-fold predictions are unbiased. The per-fold
                parameters are handed in from the tuning stage, which means no
                Optuna search is repeated here.
  * test      : a held-out set removed before any training, identical for all
                four models, hence the fairest head-to-head comparison.
"""

from __future__ import annotations

import numpy as np

from data_utils import make_folds
from metrics_utils import regression_metrics
from model_utils import build_pipeline_from_params


def fit_predict_fold(params: dict, X: np.ndarray, y: np.ndarray, tr: np.ndarray,
                     te: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit on tr, predict on te, return (y_true, y_pred)."""
    pipe = build_pipeline_from_params(params, seed=seed)
    pipe.fit(X[tr], y[tr])
    return y[te], pipe.predict(X[te])


def oof_predictions(params: dict, X: np.ndarray, y: np.ndarray, folds,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Out-of-fold predictions for a fixed hyperparameter set.

    Returns (y_true_oof, y_pred_oof, per_fold_metrics).
    """
    y_true_parts, y_pred_parts, per_fold = [], [], []
    for k, (tr, te) in enumerate(folds):
        yt, yp = fit_predict_fold(params, X, y, tr, te, seed)
        y_true_parts.append(yt)
        y_pred_parts.append(yp)
        m = {"fold": k + 1, "n_train": int(len(tr)), "n_test": int(len(te))}
        m.update(regression_metrics(yt, yp))
        per_fold.append(m)
    return np.concatenate(y_true_parts), np.concatenate(y_pred_parts), per_fold


def nested_oof_predictions(fold_params: list[dict], X: np.ndarray, y: np.ndarray,
                           outer_folds, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Out-of-fold predictions of a nested CV run.

    Outer fold k uses fold_params[k], which was selected using only that fold's
    outer-train rows, so the pooled prediction is unbiased.
    """
    if len(fold_params) != len(outer_folds):
        raise ValueError(
            f"fold_params has {len(fold_params)} entries but there are {len(outer_folds)} outer folds")

    y_true_parts, y_pred_parts, per_fold = [], [], []
    for k, (tr, te) in enumerate(outer_folds):
        yt, yp = fit_predict_fold(fold_params[k], X, y, tr, te, seed)
        y_true_parts.append(yt)
        y_pred_parts.append(yp)
        m = {"fold": k + 1, "n_train": int(len(tr)), "n_test": int(len(te))}
        m.update(regression_metrics(yt, yp))
        per_fold.append(m)
    return np.concatenate(y_true_parts), np.concatenate(y_pred_parts), per_fold


def holdout_predictions(params: dict, X: np.ndarray, y: np.ndarray, tr: np.ndarray,
                        te: np.ndarray, seed: int = 0):
    """Fit on the training rows, score the held-out rows."""
    pipe = build_pipeline_from_params(params, seed=seed)
    pipe.fit(X[tr], y[tr])
    return y[tr], pipe.predict(X[tr]), y[te], pipe.predict(X[te])


def inner_folds_for(split: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    n_splits: int, seed: int):
    """Rebuild the inner folds on a subset (once per nested outer fold)."""
    return make_folds(split, X, y, groups, n_splits=n_splits, seed=seed)
