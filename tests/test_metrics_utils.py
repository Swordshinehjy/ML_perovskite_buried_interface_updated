"""Tests for regression metrics and permutation statistics."""

from __future__ import annotations

import numpy as np
import pytest

from metrics_utils import (
    fmt_mean_std,
    mean_std,
    permutation_pvalue,
    regression_metrics,
    to_native,
    z_score,
)


def test_metrics_on_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert m["RMSE"] == pytest.approx(0.0)
    assert m["Pearson_r"] == pytest.approx(1.0)
    assert m["R2"] == pytest.approx(1.0)
    assert m["n"] == 4


def test_metrics_handle_constant_predictions():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, np.full(3, 2.0))
    assert m["Pearson_r"] == 0.0
    assert m["Spearman_rho"] == 0.0
    assert np.isfinite(m["RMSE"])


def test_r2_can_be_negative():
    """Worse-than-mean predictions give a negative R2; nothing may clip that."""
    y = np.array([0.0, 1.0, 2.0])
    m = regression_metrics(y, np.array([5.0, -5.0, 5.0]))
    assert m["R2"] < 0


def test_permutation_pvalue_is_never_zero():
    null = np.array([0.0, 0.1, -0.1, 0.05])
    p = permutation_pvalue(10.0, null, lower_is_better=False)
    assert p == pytest.approx(1.0 / (len(null) + 1))


def test_permutation_pvalue_direction():
    null = np.array([1.0, 1.1, 0.9, 1.05])
    low = permutation_pvalue(0.5, null, lower_is_better=True)
    high = permutation_pvalue(0.5, null, lower_is_better=False)
    assert low < high


def test_z_score_and_mean_std():
    null = np.array([0.0, 1.0, 2.0, 3.0])
    expected = (2.0 - null.mean()) / null.std(ddof=1)
    assert z_score(2.0, null) == pytest.approx(expected)
    assert z_score(4.0, null) > 0 and z_score(-1.0, null) < 0
    m, s = mean_std([1.0, 3.0])
    assert m == pytest.approx(2.0) and s == pytest.approx(np.sqrt(2.0))


def test_fmt_mean_std_handles_nan():
    assert fmt_mean_std(float("nan"), float("nan")) == "-"


def test_to_native_strips_numpy():
    out = to_native({"a": np.float64(1.0), "b": [np.int64(2)], "c": np.array([1.0, 2.0])})
    assert isinstance(out["a"], float)
    assert isinstance(out["b"][0], int)
    assert out["c"] == [1.0, 2.0]
