"""Regression metrics and permutation-test statistics.

Single responsibility: turn (y_true, y_pred) into a metric dict. No plotting, no I/O.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(stats.pearsonr(y_true, y_pred)[0])


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(stats.spearmanr(y_true, y_pred)[0])


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Core regression metrics: RMSE / MAE / Pearson r / R2 / Spearman rho."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Pearson_r": _pearson(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman_rho": _spearman(y_true, y_pred),
        "n": int(len(y_true)),
    }


def to_native(obj):
    """Recursively convert numpy scalars/arrays to plain Python for json dumping."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def mean_std(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), 0.0
    return float(v.mean()), float(v.std(ddof=1))


def fmt_mean_std(mean: float, std: float, nd: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "-"
    return f"{mean:.{nd}f} +/- {std:.{nd}f}"


def permutation_pvalue(observed: float, null: np.ndarray, lower_is_better: bool) -> float:
    """One-tailed permutation p-value with the +1 correction (never exactly 0)."""
    null = np.asarray(null, dtype=float)
    n = null.size
    if n == 0:
        return float("nan")
    hits = int(np.sum(null <= observed)) if lower_is_better else int(np.sum(null >= observed))
    return (1.0 + hits) / (n + 1.0)


def z_score(observed: float, null: np.ndarray) -> float:
    """Standardised distance of the observed value from the null distribution."""
    null = np.asarray(null, dtype=float)
    if null.size < 2 or null.std(ddof=1) < 1e-12:
        return float("nan")
    return float((observed - null.mean()) / null.std(ddof=1))
