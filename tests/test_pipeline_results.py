"""Tests for the result structures produced by training, prediction, scrambling
and comparison. These cover the remaining bugs found in review.

BUG-07: y_scrambling wrote statistics under `p_pearson_r` / `null_mean_pearson_r`
        while the metrics dict uses `Pearson_r`; lookups raised KeyError and the
        permutation plot could not find its column.
BUG-08: predict_new labelled molecules by truncated SMILES. Several molecules in
        new-mol.csv share a 12-character prefix, so the pivot used for comparison
        silently collapsed 9 molecules into 5 rows.
BUG-09: the optimism gap was computed as plain - nested for every metric, so for
        RMSE a negative number meant "optimistic" while for r it was positive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import train
from config import PRIMARY_METRICS
from metrics_utils import regression_metrics


def _fake_result(protocol: str = "cv") -> dict:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=40)
    return {
        "model": "random_cv", "label": "Random split + CV", "short": "Random + CV",
        "split": "random", "protocol": protocol,
        "train": regression_metrics(y_true, y_true + rng.normal(0, 0.1, 40)),
        "test": regression_metrics(y_true, y_true + rng.normal(0, 0.4, 40)),
        "cv": regression_metrics(y_true, y_true + rng.normal(0, 0.5, 40)),
        "nested": None, "cv_per_fold": [dict(fold=i + 1, **regression_metrics(
            y_true, y_true + rng.normal(0, 0.5, 40))) for i in range(5)],
        "nested_per_fold": None,
    }


def test_result_keeps_protocol_and_cv_metrics_separate():
    res = _fake_result()
    assert res["protocol"] == "cv"
    assert isinstance(res["cv"], dict) and "RMSE" in res["cv"]


def test_summary_row_reports_the_protocol_string():
    row = train.build_summary_row(_fake_result("nested"))
    assert row["cv"] == "nested"
    for metric in PRIMARY_METRICS:
        assert isinstance(row[f"{metric}_cv"], float)


def test_summary_row_handles_missing_nested_block():
    row = train.build_summary_row(_fake_result("cv"))
    assert np.isnan(row["RMSE_nested"])


def test_scrambling_stat_keys_match_metric_names():
    from y_scrambling import HIGHER_IS_BETTER

    metrics = regression_metrics(np.arange(10.0), np.arange(10.0))
    for metric in HIGHER_IS_BETTER:
        assert metric in metrics, f"{metric} is not a key of regression_metrics"
    assert "RMSE" in metrics


def test_scrambling_plot_reads_the_same_columns_it_writes(tmp_path):
    import plotting

    rng = np.random.default_rng(0)
    metrics = regression_metrics(np.arange(20.0), np.arange(20.0) + rng.normal(0, 0.3, 20))
    null_df = pd.DataFrame({k: rng.normal(0, 0.1, 30) for k in ("Pearson_r", "RMSE", "R2")})
    stats_ = {f"p_{m}": 0.005 for m in ("Pearson_r", "RMSE", "R2")}
    stats_.update({f"z_{m}": 3.0 for m in ("Pearson_r", "RMSE", "R2")})
    out = plotting.plot_y_scrambling(null_df, metrics, stats_, tmp_path / "y.png", "Random + CV")
    assert out.is_file()


def test_molecule_labels_are_unique_even_with_shared_prefixes():
    from predict_new import molecule_labels

    df = pd.DataFrame({"SMILES": [f"O=P(O)(O)CCn1c2ccc({s})cc2" for s in "ClBrI"]})
    labels = molecule_labels(df)
    assert len(set(labels)) == len(df)

    piv = pd.DataFrame({"label": np.repeat(labels, 2), "model": ["a", "b"] * len(labels),
                        "PCE_pred": np.arange(2 * len(labels), dtype=float)})
    wide = piv.pivot_table(index="label", columns="model", values="PCE_pred", aggfunc="mean")
    assert len(wide) == len(labels)


def test_truncated_smiles_would_have_collided():
    """Documents BUG-08: the previous labelling scheme lost rows."""
    from predict_new import molecule_labels

    df = pd.DataFrame({"SMILES": [f"O=P(O)(O)CCn1c2ccc({s})cc2" for s in "ClBrI"]})
    truncated = df["SMILES"].str.slice(0, 12).to_numpy()
    assert len(set(truncated)) < len(df)
    assert len(set(molecule_labels(df))) == len(df)


def _record(cv_metrics: dict, nested_metrics: dict | None, protocol: str) -> dict:
    return {"label": protocol, "split": "random", "protocol": protocol,
            "train": cv_metrics, "test": cv_metrics, "cv": cv_metrics,
            "nested": nested_metrics, "cv_per_fold": [dict(fold=1, **cv_metrics)] * 3,
            "nested_per_fold": [dict(fold=1, **nested_metrics)] * 3 if nested_metrics else None,
            "n_train": 100, "n_test": 20}


def test_optimism_gap_is_positive_when_plain_cv_looks_better():
    import compare_models

    plain = {"RMSE": 1.0, "Pearson_r": 0.60, "R2": 0.30, "MAE": 0.8, "Spearman_rho": 0.5, "n": 100}
    nested = {"RMSE": 1.5, "Pearson_r": 0.20, "R2": 0.05, "MAE": 0.9, "Spearman_rho": 0.1, "n": 100}
    records = {"random_cv": _record(plain, None, "cv"),
               "random_nested": _record(plain, nested, "nested")}
    gaps = compare_models.build_gap_table(records)
    assert len(gaps) == 1
    row = gaps.iloc[0]
    assert row["gap_RMSE"] == pytest.approx(0.5)
    assert row["gap_Pearson_r"] == pytest.approx(0.4)
    assert row["gap_R2"] == pytest.approx(0.25)


def test_optimism_gap_is_negative_when_plain_cv_looks_worse():
    import compare_models

    plain = {"RMSE": 1.5, "Pearson_r": 0.20, "R2": 0.05, "MAE": 0.9, "Spearman_rho": 0.1, "n": 100}
    nested = {"RMSE": 1.0, "Pearson_r": 0.60, "R2": 0.30, "MAE": 0.8, "Spearman_rho": 0.5, "n": 100}
    records = {"random_cv": _record(plain, None, "cv"),
               "random_nested": _record(plain, nested, "nested")}
    row = compare_models.build_gap_table(records).iloc[0]
    assert row["gap_RMSE"] < 0 and row["gap_Pearson_r"] < 0 and row["gap_R2"] < 0


def test_comparison_table_columns():
    import compare_models

    m = {"RMSE": 1.2, "Pearson_r": 0.3, "R2": 0.1, "MAE": 0.9, "Spearman_rho": 0.2, "n": 100}
    records = {"group_cv": _record(m, None, "cv")}
    df = compare_models.build_comparison(records)
    for metric in PRIMARY_METRICS:
        assert f"{metric}_cv" in df.columns
        assert f"{metric}_cv_std" in df.columns
    assert df.loc[0, "cv_protocol"] == "cv"


def test_markdown_fallback_works_without_tabulate():
    import compare_models

    md = compare_models.to_markdown(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    assert md.startswith("| a | b |")
    assert "---" in md
