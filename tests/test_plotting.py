"""Tests for the plotting layer, covering the layout bugs found in review.

BUG-04: METRIC_PANELS hard-coded y limits of (-0.05, 1.05) for r and R2. A model
        with a negative R2 had its bar clipped away entirely.
BUG-05: bar annotations were always placed above the bar, so negative values put
        the label under the axis where it was drawn on top of the tick labels.
BUG-06: the metrics heatmap coloured RMSE and R2 with the same RdYlGn map, so the
        *best* (lowest) RMSE was painted red, the worst green.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import plotting


def _metric_dict(rmse=1.2, r=0.3, r2=0.1):
    return {"RMSE": rmse, "MAE": 0.9, "Pearson_r": r, "R2": r2, "Spearman_rho": 0.2, "n": 100}


def test_autoscale_keeps_negative_values_visible():
    fig, ax = plt.subplots()
    plotting._autoscale(ax, [-0.8, 0.2])
    lo, hi = ax.get_ylim()
    assert lo < -0.8 and hi > 0.2
    plt.close(fig)


def test_autoscale_handles_all_negative_values():
    fig, ax = plt.subplots()
    plotting._autoscale(ax, [-0.8, -0.4])
    lo, hi = ax.get_ylim()
    assert lo < -0.8 and hi >= 0.0
    plt.close(fig)


def test_metric_bars_negative_r2_is_not_clipped(tmp_path):
    groups = {"Train": _metric_dict(r2=0.5), "Test": _metric_dict(rmse=1.5, r=0.1, r2=-0.6)}
    out = plotting.plot_metric_bars(groups, tmp_path / "bars.png", "short title")
    assert out.is_file() and out.stat().st_size > 0


def test_annotation_sits_below_the_bar_when_negative():
    fig, ax = plt.subplots()
    bars = ax.bar([0], [-0.6], 0.5)
    plotting._autoscale(ax, [-0.6])
    plotting._annotate_bars(ax, bars)
    assert len(ax.texts) == 1
    assert ax.texts[0].get_position()[1] < -0.6
    plt.close(fig)


def test_annotation_sits_above_the_bar_when_positive():
    fig, ax = plt.subplots()
    bars = ax.bar([0], [0.6], 0.5)
    plotting._autoscale(ax, [0.6])
    plotting._annotate_bars(ax, bars)
    assert ax.texts[0].get_position()[1] > 0.6
    plt.close(fig)


def test_goodness_matrix_flips_lower_is_better_metrics():
    df = pd.DataFrame({"RMSE_cv": [1.0, 2.0, 3.0], "R2_cv": [0.1, 0.5, 0.9]})
    mat = plotting.goodness_matrix(df, ["RMSE_cv", "R2_cv"])
    assert mat[0, 0] == pytest.approx(1.0), "lowest RMSE must be the best cell"
    assert mat[2, 0] == pytest.approx(0.0), "highest RMSE must be the worst cell"
    assert mat[2, 1] == pytest.approx(1.0), "highest R2 must be the best cell"
    assert mat[0, 1] == pytest.approx(0.0)


def test_goodness_matrix_handles_constant_columns():
    df = pd.DataFrame({"RMSE_cv": [1.0, 1.0], "R2_cv": [0.2, 0.2]})
    mat = plotting.goodness_matrix(df, ["RMSE_cv", "R2_cv"])
    assert np.allclose(mat, 0.5)


def test_heatmap_renders(tmp_path):
    df = pd.DataFrame({"label": ["A", "B"], "RMSE_cv": [1.0, 2.0], "R2_cv": [0.8, 0.1],
                       "RMSE_cv_std": [0.1, 0.2]})
    out = plotting.plot_metrics_heatmap(df, tmp_path / "heat.png", "heatmap")
    assert out.is_file()


def test_legend_is_placed_below_the_axes_not_under_the_title():
    fig, axes = plt.subplots(1, 3)
    axes[0].bar([0], [1], label="Train")
    plotting._legend_below(fig, ncol=2)
    legend = fig.legends[0] if fig.legends else None
    assert legend is not None
    assert legend._loc == 8 or str(legend._loc) == "lower center"
    plt.close(fig)


def test_grouped_metric_bars_skips_missing_series(tmp_path):
    df = pd.DataFrame({"label": ["A", "B"], "RMSE_cv": [1.0, 2.0],
                       "Pearson_r_cv": [0.3, 0.2], "R2_cv": [0.1, 0.05]})
    out = plotting.plot_grouped_metric_bars(
        df, tmp_path / "g.png", "comparison",
        [("{metric}_cv", "CV"), ("{metric}_nested", "Nested")])
    assert out.is_file()


def test_grouped_metric_bars_raises_without_columns(tmp_path):
    df = pd.DataFrame({"label": ["A"]})
    with pytest.raises(ValueError):
        plotting.plot_grouped_metric_bars(df, tmp_path / "g.png", "x", [("{metric}_cv", "CV")])


def test_parity_and_scrambling_figures_render(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    plotting.plot_parity(y, y + rng.normal(0, 0.2, 50), tmp_path / "p.png", "Random + CV")
    assert (tmp_path / "p.png").is_file()

    null = pd.DataFrame({"Pearson_r": rng.normal(0, 0.1, 50), "RMSE": rng.normal(1.4, 0.05, 50),
                         "R2": rng.normal(-0.1, 0.1, 50)})
    observed = {"Pearson_r": 0.35, "RMSE": 1.2, "R2": 0.12}
    stats_ = {"p_Pearson_r": 0.005, "z_Pearson_r": 4.0, "p_RMSE": 0.005,
              "z_RMSE": -4.0, "p_R2": 0.005, "z_R2": 4.0}
    plotting.plot_y_scrambling(null, observed, stats_, tmp_path / "y.png", "Random + CV")
    plotting.plot_scrambling_convergence(null, observed, tmp_path / "c.png", "Random + CV")
    assert (tmp_path / "y.png").is_file() and (tmp_path / "c.png").is_file()
