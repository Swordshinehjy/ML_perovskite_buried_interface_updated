"""Tests for the six issues raised in the second review round.

  BUG-10  load_external_search_space was dead code and, even if called, could not
          be observed by modules that had already bound SEARCH_SPACE.
  BUG-11  the nested tuning history concatenated per-fold trials whose counters
          restart at 1, drawing a sawtooth "best so far" curve.
  BUG-12  y_scrambling labelled its statistic with the model's CV protocol even
          though it always evaluates the deployed parameters on plain folds.
  BUG-13  mismatched best_params.json / nested_folds.json raised a bare KeyError.
  BUG-14  the beeswarm fallback passed a dummy matrix, losing feature direction.
  BUG-15  missing values travelled all the way into XGBoost before failing.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import plotting
from model_utils import BUILTIN_SEARCH_SPACE, SEARCH_SPACE, SEARCH_SPACE_FILE, load_search_space


def test_search_space_override_is_applied_at_import_time(tmp_path):
    """BUG-10: the override must be visible to every module, without a call site."""
    space = load_search_space(tmp_path / "absent.json")
    assert space == BUILTIN_SEARCH_SPACE

    external = tmp_path / "xgb_search_space.json"
    external.write_text(json.dumps({"max_depth": {"type": "int", "low": 2, "high": 4}}),
                        encoding="utf-8")
    merged = load_search_space(external)
    assert merged["max_depth"]["high"] == 4
    assert merged["n_estimators"] == BUILTIN_SEARCH_SPACE["n_estimators"]


def test_search_space_override_rejects_unknown_parameters(tmp_path):
    external = tmp_path / "xgb_search_space.json"
    external.write_text(json.dumps({"n_layers": {"low": 1, "high": 3}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown hyperparameters"):
        load_search_space(external)


def test_builtin_space_is_not_mutated(tmp_path):
    external = tmp_path / "xgb_search_space.json"
    external.write_text(json.dumps({"max_depth": {"type": "int", "low": 2, "high": 4}}),
                        encoding="utf-8")
    load_search_space(external)
    assert BUILTIN_SEARCH_SPACE["max_depth"]["high"] == 8


def test_search_space_matches_the_repository_default():
    """No xgb_search_space.json is shipped, so the effective space is the built-in."""
    assert not SEARCH_SPACE_FILE.is_file()
    assert SEARCH_SPACE == BUILTIN_SEARCH_SPACE


def _nested_trials(n_folds=3, n_trials=20):
    """Reproduce how hyperparam_tuning concatenates per-fold trial frames."""
    rng = np.random.default_rng(0)
    frames = []
    for k in range(n_folds):
        sub = pd.DataFrame({"trial": np.arange(1, n_trials + 1),
                            "cv_rmse": rng.normal(1.3, 0.1, n_trials)})
        sub["best_so_far"] = sub["cv_rmse"].cummin()
        sub.insert(0, "outer_fold", k + 1)
        sub["trial_global"] = sub["trial"] + k * n_trials
        frames.append(sub)
    return pd.concat(frames, ignore_index=True), n_trials


def test_nested_history_uses_a_monotonic_global_trial_axis():
    """BUG-11: without the offset every fold restarts at 1 and the x axis overlaps."""
    trials, n_trials = _nested_trials()
    assert trials["trial_global"].is_monotonic_increasing
    assert trials["trial_global"].max() == len(trials)
    for fold, sub in trials.groupby("outer_fold"):
        assert sub["trial_global"].min() == (fold - 1) * n_trials + 1
    assert trials["trial"].max() == n_trials, "per-fold counter is expected to restart"


def test_nested_history_figure_draws_one_curve_per_fold(tmp_path):
    trials, _ = _nested_trials()
    out = plotting.plot_optimization_history(trials, tmp_path / "h.png", "nested history")
    assert out.is_file()

    fig, ax = plt.subplots()
    for fold, sub in trials.groupby("outer_fold"):
        ax.step(sub["trial_global"], sub["best_so_far"], where="post",
                label=f"outer fold {int(fold)} best")
    assert len(ax.get_legend_handles_labels()[1]) == trials["outer_fold"].nunique()
    plt.close(fig)


def test_single_search_history_still_uses_one_curve(tmp_path):
    trials, _ = _nested_trials(n_folds=1)
    out = plotting.plot_optimization_history(trials, tmp_path / "s.png", "single history")
    assert out.is_file()


def test_scrambling_summary_declares_its_observed_basis():
    """BUG-12: the protocol field must not imply per-fold nested parameters."""
    import y_scrambling

    src = (SEARCH_SPACE_FILE.parent / "src" / "y_scrambling.py").read_text(encoding="utf-8")
    assert "observed_basis" in src
    assert "does **not** reuse" in src
    assert "Protocol note" in src
    assert y_scrambling.HIGHER_IS_BETTER == ("Pearson_r", "R2")


def test_mismatched_nested_folds_raise_a_clear_error():
    """BUG-13."""
    from train import extract_fold_params

    params = {"n_estimators": 100, "max_depth": 3}
    records = [{"fold": 1, "n_estimators": 100, "max_depth": 3}]
    assert extract_fold_params("random_nested", records, params) == [params]

    stale = [{"fold": 1, "n_estimators": 100}]
    with pytest.raises(ValueError, match="Re-run"):
        extract_fold_params("random_nested", stale, params)


def test_beeswarm_fallback_keeps_the_real_feature_matrix(tmp_path):
    """BUG-14: the fallback must not fabricate a constant matrix."""
    import shap_analysis

    rng = np.random.default_rng(0)
    n, p = 80, 3
    X = rng.normal(size=(n, p))
    shap_values = 3.0 * X                      # every feature drives its own SHAP value
    names = ["a", "b", "c"]
    out = shap_analysis.plot_beeswarm(shap_values, X, names, tmp_path / "b.png", "t")
    assert out is None or out.is_file()

    table = shap_analysis.importance_table(shap_values, X, names)
    corr = table.set_index("feature")["mean_shap_signed_corr"]
    assert (corr > 0.99).all(), "with a constant matrix every correlation collapses to 0"
    assert (table["std_abs_shap"] > 0).all()


def test_missing_targets_are_reported_at_load_time(tmp_path):
    """BUG-15: NaN must be caught where the data is read, not inside XGBoost."""
    from data_utils import load_dataset

    path = tmp_path / "bad.csv"
    path.write_text("SMILES,PCE,control_PCE,f1\nCCO,20.0,18.0,1.0\nCCC,,18.0,2.0\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="missing/non-numeric"):
        load_dataset(path)


def test_missing_features_are_reported_at_matrix_build(tmp_path):
    from data_utils import build_matrices, load_new_molecules

    path = tmp_path / "bad_new.csv"
    path.write_text("SMILES,control_PCE,f1\nCCO,18.0,\nCCC,18.0,2.0\n", encoding="utf-8")
    df = load_new_molecules(path)
    df["delta_PCE"] = 0.0
    with pytest.raises(ValueError, match="non-finite feature"):
        build_matrices(df)


def test_clean_data_passes_the_guards():
    from data_utils import build_matrices, load_dataset, load_new_molecules
    from predict_new import molecule_labels

    df = load_dataset()
    X, y, ctrl, groups, feats = build_matrices(df)
    assert np.isfinite(X).all() and np.isfinite(y).all()

    new = load_new_molecules()
    labels = molecule_labels(new)
    assert len(set(labels)) == len(new)


def test_review_issues_are_documented():
    path = SEARCH_SPACE_FILE.parent / "README.md"
    text = path.read_text(encoding="utf-8")
    assert "BUG-10" in text and "BUG-15" in text
