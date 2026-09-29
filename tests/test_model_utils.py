"""Tests for the estimator factory and the Optuna distribution builder.

BUG-03 (reviewed): hyperparameter importance was recomputed from archived trials
by rebuilding an Optuna study with `create_trial(params=..., value=...)` but
without `distributions=`. Optuna rejected every trial with
"Inconsistent parameters ... and distributions set()", so the importance figure
was silently replaced by a placeholder.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest

from model_utils import (
    SEARCH_SPACE,
    build_distributions,
    build_pipeline_from_params,
    build_xgb,
    is_int_param,
)
from tuning import param_importance_from_trials, trials_to_frame


def test_int_params_are_detected():
    assert is_int_param("n_estimators")
    assert is_int_param("max_depth")
    assert not is_int_param("learning_rate")


def test_distributions_cover_the_whole_search_space():
    dists = build_distributions()
    assert set(dists) == set(SEARCH_SPACE)
    assert isinstance(dists["n_estimators"], optuna.distributions.IntDistribution)
    assert dists["n_estimators"].step == SEARCH_SPACE["n_estimators"]["step"]
    assert isinstance(dists["learning_rate"], optuna.distributions.FloatDistribution)
    assert dists["learning_rate"].log is True


def test_distributions_accept_sampled_values():
    dists = build_distributions()

    def objective(trial):
        from model_utils import suggest_params
        params = suggest_params(trial)
        for name, value in params.items():
            assert dists[name]._contains(
                dists[name].to_internal_repr(value)), f"{name}={value} outside its distribution"
        return float(value) if False else 1.0

    optuna.create_study(direction="minimize").optimize(objective, n_trials=5)


def _valid_param_value(name, spec, i, n_trials):
    """Pick a value that lies strictly inside the declared range and varies per trial."""
    lo, hi = float(spec["low"]), float(spec["high"])
    frac = (i + 1) / (n_trials + 1)
    if spec.get("type") == "int" or spec.get("step"):
        step = int(spec.get("step", 1))
        offset = int(round((hi - lo) * frac / step)) * step
        return int(min(lo + offset, hi))
    value = lo + (hi - lo) * frac
    if spec.get("log"):
        value = float(np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * frac))
    return float(value)


def test_importance_from_archived_trials_does_not_raise():
    """The regression that used to fail with 'distributions set()'."""
    rng = np.random.default_rng(0)
    n_trials = 12
    rows = []
    for i in range(n_trials):
        row = {"trial": i + 1, "cv_rmse": float(rng.normal(1.3, 0.1)), "state": "COMPLETE"}
        row.update({name: _valid_param_value(name, spec, i, n_trials)
                    for name, spec in SEARCH_SPACE.items()})
        rows.append(row)
    trials = pd.DataFrame(rows)
    trials["cv_rmse"] = np.linspace(1.0, 2.0, len(trials))
    trials["best_so_far"] = trials["cv_rmse"].cummin()

    importance = param_importance_from_trials(trials)
    assert isinstance(importance, dict)


def test_trials_to_frame_roundtrip():
    def objective(trial):
        from model_utils import suggest_params
        suggest_params(trial)
        return float(trial.number)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=4)
    df = trials_to_frame(study)
    assert len(df) == 4
    assert df["best_so_far"].is_monotonic_decreasing
    assert set(SEARCH_SPACE).issubset(df.columns)


def test_pipeline_scales_then_fits():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3)) * 10 + 5
    y = X[:, 0] * 2 - X[:, 1]
    params = {k: (100 if v.get("type") == "int" else 0.1) for k, v in SEARCH_SPACE.items()}
    pipe = build_pipeline_from_params(params)
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()


def test_build_xgb_is_deterministic_for_a_seed():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    y = X[:, 0]
    params = {k: (100 if v.get("type") == "int" else 0.1) for k, v in SEARCH_SPACE.items()}
    a = build_xgb(params, seed=7).fit(X, y).predict(X)
    b = build_xgb(params, seed=7).fit(X, y).predict(X)
    assert np.allclose(a, b)
