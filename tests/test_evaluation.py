"""Tests for the three evaluation protocols."""

from __future__ import annotations

import json

import numpy as np
import pytest

from config import SEED
from data_utils import make_folds
from evaluation import holdout_predictions, nested_oof_predictions, oof_predictions
from model_utils import SEARCH_SPACE

PARAMS = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.2,
          "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1,
          "gamma": 1e-4, "reg_alpha": 1e-4, "reg_lambda": 1e-4}


def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.2, n)
    return X, y


def test_oof_covers_training_rows_exactly_once():
    X, y = _data()
    folds = make_folds("random", X, y, None, n_splits=4, seed=SEED)
    yt, yp, per_fold = oof_predictions(PARAMS, X, y, folds)
    assert len(yt) == len(y)
    assert len(yp) == len(y)
    assert len(per_fold) == 4
    assert {m["fold"] for m in per_fold} == {1, 2, 3, 4}


def test_nested_requires_one_param_set_per_fold():
    X, y = _data()
    folds = make_folds("random", X, y, None, n_splits=4, seed=SEED)
    with pytest.raises(ValueError, match="outer folds"):
        nested_oof_predictions([PARAMS] * 3, X, y, folds)


def test_nested_uses_the_parameter_set_of_its_own_fold():
    """A deliberately bad parameter set must degrade that fold's predictions only."""
    X, y = _data(n=120)
    folds = make_folds("random", X, y, None, n_splits=4, seed=SEED)
    good = dict(PARAMS)
    bad = dict(PARAMS, n_estimators=1, max_depth=1, learning_rate=0.5)

    _, pred_good, _ = nested_oof_predictions([good] * 4, X, y, folds)
    params = [good] * 4
    params[2] = bad
    _, pred_mixed, folds_metrics = nested_oof_predictions(params, X, y, folds)

    changed = ~np.isclose(pred_good, pred_mixed)
    assert changed.any()
    fold_sizes = [len(te) for _, te in folds]
    start = sum(fold_sizes[:2])
    stop = start + fold_sizes[2]
    assert changed[start:stop].all(), "only fold 3 should change"


def test_holdout_returns_four_aligned_arrays():
    X, y = _data()
    tr = np.arange(0, 40)
    te = np.arange(40, 60)
    y_tr, p_tr, y_te, p_te = holdout_predictions(PARAMS, X, y, tr, te)
    assert np.array_equal(y_tr, y[tr]) and np.array_equal(y_te, y[te])
    assert p_tr.shape == y_tr.shape and p_te.shape == y_te.shape


def test_params_roundtrip_through_json(tmp_path):
    from tuning import load_params, make_params_payload, save_params

    payload = make_params_payload("random_cv", PARAMS, split="random", cv="cv",
                                  best_cv_rmse=1.23, n_trials=10, seed=SEED,
                                  selection="test")
    path = tmp_path / "best_params.json"
    save_params(path, payload)
    loaded = load_params(path)
    assert loaded["params"]["n_estimators"] == PARAMS["n_estimators"]
    assert loaded["best_cv_rmse"] == pytest.approx(1.23)
    assert set(SEARCH_SPACE) == set(loaded["search_space"])
    json.dumps(loaded)  # must stay json serialisable


def test_load_params_raises_a_helpful_error(tmp_path):
    from tuning import load_params

    with pytest.raises(FileNotFoundError):
        load_params(tmp_path / "missing.json")
