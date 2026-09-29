"""Tests for the SHAP interpretation stage.

The stage must attribute importance to the feature that actually drives the
target, and the exported artefacts must stay consistent with the model bundle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import CONTROL_COL, DELTA_COL, TARGET_COL


def _fitted_pipeline(n=120, seed=0):
    from model_utils import build_pipeline_from_params

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = 3.0 * X[:, 2] + 0.5 * X[:, 0] + rng.normal(0, 0.1, n)   # feature 2 dominates
    params = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.2,
              "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1,
              "gamma": 1e-4, "reg_alpha": 1e-4, "reg_lambda": 1e-4}
    pipe = build_pipeline_from_params(params)
    pipe.fit(X, y)
    return pipe, X, ["f0", "f1", "f2", "f3"]


def test_shap_values_have_one_column_per_feature():
    from shap_analysis import extract_shap_values

    pipe, X, names = _fitted_pipeline()
    values, base = extract_shap_values(pipe, X)
    assert values.shape == (len(X), len(names))
    assert np.isfinite(base)


def test_shap_attributions_sum_to_the_prediction():
    from shap_analysis import extract_shap_values

    pipe, X, names = _fitted_pipeline()
    values, base = extract_shap_values(pipe, X)
    pred = pipe.predict(X)
    assert np.allclose(base + values.sum(axis=1), pred, atol=1e-3)


def test_importance_ranks_the_driving_feature_first():
    from shap_analysis import extract_shap_values, importance_table

    pipe, X, names = _fitted_pipeline()
    values, _ = extract_shap_values(pipe, X)
    table = importance_table(values, X, names)
    assert table.iloc[0]["feature"] == "f2"
    assert table["mean_abs_shap"].is_monotonic_decreasing


def test_signed_correlation_reveals_the_direction():
    from shap_analysis import extract_shap_values, importance_table

    pipe, X, names = _fitted_pipeline()
    values, _ = extract_shap_values(pipe, X)
    table = importance_table(values, X, names).set_index("feature")
    assert table.loc["f2", "mean_shap_signed_corr"] > 0, "higher f2 raises the prediction"


def test_importance_table_handles_a_constant_feature():
    """A zero-variance feature must not produce NaN.

    corrcoef of a constant column is undefined and returns NaN, which used to
    propagate into the exported table; the table now guards against it.
    """
    from shap_analysis import extract_shap_values, importance_table

    pipe, X, names = _fitted_pipeline()
    X = X.copy()
    X[:, 1] = 0.0
    values, _ = extract_shap_values(pipe, X)
    table = importance_table(values, X, names)
    assert np.isfinite(table[["mean_abs_shap", "mean_shap", "std_abs_shap",
                              "mean_shap_signed_corr"]].to_numpy()).all()
    row = table.loc[table["feature"] == "f1"].iloc[0]
    assert row["mean_shap_signed_corr"] == pytest.approx(0.0)
    assert table.iloc[0]["feature"] == "f2"


def test_shap_figures_render(tmp_path):
    import plotting
    from shap_analysis import extract_shap_values, importance_table

    pipe, X, names = _fitted_pipeline()
    values, _ = extract_shap_values(pipe, X)
    table = importance_table(values, X, names)
    assert plotting.plot_shap_importance(table, tmp_path / "a.png", "t").is_file()
    assert plotting.plot_shap_sign(table, tmp_path / "b.png", "t").is_file()
    assert plotting.plot_shap_dependence("f2", X[:, 2], values[:, 2],
                                         tmp_path / "c.png", "t").is_file()
    combined = table.copy()
    combined.insert(0, "model", "random_cv")
    assert plotting.plot_shap_model_comparison(combined, tmp_path / "d.png", "t").is_file()


def test_model_comparison_uses_the_union_of_top_features(tmp_path):
    import plotting

    rows = []
    for model, feats in (("a", ["x", "y", "z"]), ("b", ["y", "x", "w"])):
        for rank, f in enumerate(feats):
            rows.append({"model": model, "feature": f,
                         "mean_abs_shap": 1.0 / (rank + 1)})
    df = pd.DataFrame(rows)
    out = plotting.plot_shap_model_comparison(df, tmp_path / "e.png", "t", top_n=4)
    assert out.is_file()
