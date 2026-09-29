"""All plotting functions.

Single responsibility: turn numbers/arrays that already exist into a figure on
disk. No computation, no modelling. One figure per function.

Layout rules applied throughout (these were the source of the earlier overlaps):
  * figure-level legend sits *below* the axes, never under the suptitle;
  * suptitle carries the model name, per-axes titles stay short;
  * no hard-coded y limits, so negative R2 bars are never clipped;
  * bar annotations flip below the bar when the value is negative.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy import stats

from config import CONTROL_COL, DELTA_COL, LOWER_IS_BETTER, PRIMARY_METRICS
from model_utils import LABELS, LOG_PARAMS

FIG_DPI = 300
TRAIN_COLOR = "#3C6E9F"
TEST_COLOR = "#E8871A"
NESTED_COLOR = "#3E8E5A"
ACCENT_COLOR = "#C0392B"
GREY_COLOR = "#B0B0B0"

# (metric key, axis label, hard y-limits or None for autoscaling)
METRIC_PANELS = [
    ("RMSE", "RMSE", None),
    ("Pearson_r", "Pearson r", None),
    ("R2", r"$R^2$", None),
]

# Fraction of the data range kept free above/below bars so annotations fit.
HEADROOM = 0.22


def apply_style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "font.family": "sans-serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _save(fig: plt.Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def _limits(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    lo = float(np.floor(min(y_true.min(), y_pred.min())))
    hi = float(np.ceil(max(y_true.max(), y_pred.max())))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return lo, hi


def _decorate_parity(ax: plt.Axes, lo: float, hi: float, xlabel: str, ylabel: str) -> None:
    ax.plot([lo, hi], [lo, hi], "-", color=ACCENT_COLOR, alpha=0.75, lw=0.9, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.minorticks_on()
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.3f}") -> None:
    """Label bars above the top for positive values and below zero for negative ones."""
    for b in bars:
        h = b.get_height()
        if not np.isfinite(h):
            continue
        offset = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        va = "bottom" if h >= 0 else "top"
        y = h + offset if h >= 0 else h - offset
        ax.text(b.get_x() + b.get_width() / 2, y, fmt.format(h),
                ha="center", va=va, fontsize=7, clip_on=False)


def _autoscale(ax: plt.Axes, values) -> None:
    """Leave room for annotations; keeps negative values fully visible."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    lo, hi = float(v.min()), float(v.max())
    if lo > 0:
        lo = min(lo, 0.0)
    if hi < 0:
        hi = max(hi, 0.0)
    span = hi - lo
    if span < 1e-9:
        span = max(abs(hi), 1.0) * 0.1
    ax.set_ylim(lo - HEADROOM * span, hi + HEADROOM * span)


def _legend_below(fig: plt.Figure, ncol: int, bottom: float = 0.10) -> None:
    """Place a figure-level legend under the axes, clear of the suptitle."""
    handles, labels = [], []
    seen = set()
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in seen:
                seen.add(ll)
                handles.append(hh)
                labels.append(ll)
    if not handles:
        bottom = 0.02
    else:
        fig.legend(handles, labels, loc="lower center", ncol=max(1, ncol),
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, bottom, 1, 0.93))


# Parity scatter plots
def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, out: Path, title: str,
                xlabel: str | None = None, ylabel: str | None = None,
                color: str = TRAIN_COLOR) -> Path:
    """Single parity scatter with the y = x line and a fitted regression line."""
    apply_style()
    xlabel = xlabel or f"Measured {DELTA_COL} (%)"
    ylabel = ylabel or f"Predicted {DELTA_COL} (%)"
    fig, ax = plt.subplots(figsize=(4.2, 4.4), dpi=150)
    ax.scatter(y_true, y_pred, s=16, alpha=0.7, color=color, edgecolor="none")
    lo, hi = _limits(y_true, y_pred)
    _decorate_parity(ax, lo, hi, xlabel, ylabel)
    if np.std(y_true) > 1e-12:
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        xs = np.linspace(lo, hi, 20)
        ax.plot(xs, slope * xs + intercept, "--", color="#2E6B34", lw=1.0,
                label=f"fit slope={slope:.2f}")
    r = float(stats.pearsonr(y_true, y_pred)[0]) if np.std(y_pred) > 1e-12 else 0.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ax.set_title(f"{title}\nn={len(y_true)}   R={r:.3f}   RMSE={rmse:.3f}", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    return _save(fig, out)


def plot_train_test_parity(y_tr, p_tr, y_te, p_te, out: Path, title: str) -> Path:
    """Training and held-out test points on one parity plot."""
    apply_style()
    fig, ax = plt.subplots(figsize=(4.2, 4.4), dpi=150)
    ax.scatter(y_tr, p_tr, s=12, alpha=0.6, color=TRAIN_COLOR, label="Train")
    ax.scatter(y_te, p_te, s=22, alpha=0.9, color=TEST_COLOR, label="Test")
    lo, hi = _limits(np.r_[y_tr, y_te], np.r_[p_tr, p_te])
    _decorate_parity(ax, lo, hi, f"Measured {DELTA_COL} (%)", f"Predicted {DELTA_COL} (%)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out)


def plot_pce_parity(pce_meas: np.ndarray, pce_pred: np.ndarray, out: Path, title: str) -> Path:
    """Parity plot on the reconstructed PCE scale."""
    return plot_parity(pce_meas, pce_pred, out, title,
                       xlabel="Measured PCE (%)",
                       ylabel=f"Predicted PCE (%)  (= {DELTA_COL} + {CONTROL_COL})",
                       color=NESTED_COLOR)


# Metric bar charts
def plot_metric_bars(groups: dict[str, dict], out: Path, title: str) -> Path:
    """RMSE / Pearson r / R2 for one model across evaluation protocols.

    groups maps a short protocol name to its metric dict, e.g.
    {"Train": {...}, "Test": {...}, "KFold CV": {...}, "Nested CV": {...}}.
    """
    apply_style()
    names = list(groups)
    colors = [TRAIN_COLOR, TEST_COLOR, NESTED_COLOR, "#8E6BAF", "#7F7F7F"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), dpi=140)
    x = np.arange(len(names))
    w = 0.8 / max(len(names), 1)

    for ax, (key, label, _) in zip(axes, METRIC_PANELS):
        vals = [groups[n].get(key, np.nan) for n in names]
        for i, (n, v) in enumerate(zip(names, vals)):
            b = ax.bar(x[i], v, w, color=colors[i % len(colors)],
                       label=n if ax is axes[0] else None)
            _annotate_bars(ax, b)
        _autoscale(ax, vals)
        ax.set_ylabel(label)
        ax.set_title(LABELS.get(key, label), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)

    fig.suptitle(title, fontsize=11)
    _legend_below(fig, ncol=len(names))
    return _save(fig, out)


def plot_fold_metrics(folds_df: pd.DataFrame, out: Path, title: str) -> Path:
    """Per-fold RMSE / Pearson r / R2."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), dpi=140)
    x = np.arange(len(folds_df))
    for ax, (key, label, _) in zip(axes, METRIC_PANELS):
        vals = folds_df[key].to_numpy(dtype=float)
        bars = ax.bar(x, vals, 0.65, color=NESTED_COLOR, edgecolor="black", alpha=0.85)
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ax.axhline(m, ls="--", lw=1.0, color=ACCENT_COLOR, label="mean")
        ax.axhspan(m - s, m + s, color=ACCENT_COLOR, alpha=0.10)
        _autoscale(ax, np.r_[vals, m - s, m + s])
        _annotate_bars(ax, bars)
        ax.set_ylabel(label)
        ax.set_title(f"{LABELS.get(key, label)} per fold\nmean {m:.3f} +/- {s:.3f}", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{int(i)}" for i in folds_df["fold"]], fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    return _save(fig, out)


def plot_grouped_metric_bars(summary: pd.DataFrame, out: Path, title: str,
                             columns: list[tuple[str, str]]) -> Path:
    """Compare several models on RMSE / Pearson r / R2.

    columns is a list of (column template, series name); the template's {metric}
    placeholder is filled per panel, e.g. ("{metric}_cv", "CV") -> RMSE_cv.
    Series whose column is absent are skipped.
    """
    apply_style()
    labels = summary["label"].tolist()
    x = np.arange(len(labels))
    present = [(t, s) for t, s in columns if t.format(metric="RMSE") in summary.columns]
    if not present:
        raise ValueError("no metric columns available to plot")
    w = 0.8 / len(present)
    colors = [TRAIN_COLOR, NESTED_COLOR, TEST_COLOR, "#8E6BAF"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), dpi=140)

    for ax, (metric, label, _) in zip(axes, METRIC_PANELS):
        for i, (template, series) in enumerate(present):
            vals = pd.to_numeric(summary[template.format(metric=metric)],
                                 errors="coerce").to_numpy(dtype=float)
            offset = (i - (len(present) - 1) / 2) * w
            bars = ax.bar(x + offset, vals, w, color=colors[i % len(colors)],
                          label=series if ax is axes[0] else None)
            _annotate_bars(ax, bars)
        _autoscale(ax, np.concatenate([
            pd.to_numeric(summary[t.format(metric=metric)], errors="coerce").to_numpy(dtype=float)
            for t, _ in present]))
        ax.set_ylabel(label)
        ax.set_title(LABELS.get(metric, label), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)

    fig.suptitle(title, fontsize=11)
    _legend_below(fig, ncol=len(present))
    return _save(fig, out)


# Tuning diagnostics
def plot_optimization_history(trials_df: pd.DataFrame, out: Path, title: str) -> Path:
    """Search history: per-trial CV RMSE with the best-so-far curve.

    A nested run concatenates a separate search per outer fold. Each fold restarts
    its trial counter at 1 and its best-so-far from scratch, so the folds are drawn
    as separate curves on a shared global trial axis (`trial_global`). Collapsing
    them into one "best so far" line used to produce a sawtooth that dropped back
    down at every fold boundary.
    """
    apply_style()
    xcol = "trial_global" if "trial_global" in trials_df.columns else "trial"
    has_folds = "outer_fold" in trials_df.columns
    folds = sorted(pd.unique(trials_df["outer_fold"])) if has_folds else [None]
    multi = has_folds and len(folds) > 1

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=140)
    cmap = plt.get_cmap("tab10")

    if not multi:
        ax.plot(trials_df[xcol], trials_df["cv_rmse"], "o", ms=3, alpha=0.35,
                color="#9BB7CE", label="trial CV RMSE")
        ax.plot(trials_df[xcol], trials_df["best_so_far"], "-", lw=1.6,
                color=ACCENT_COLOR, label="best so far")
    else:
        for i, fold in enumerate(folds):
            sub = trials_df[trials_df["outer_fold"] == fold]
            color = cmap(i % 10)
            last = sub[xcol].max()
            ax.plot(sub[xcol], sub["cv_rmse"], "o", ms=3, alpha=0.30, color=color)
            ax.step(sub[xcol], sub["best_so_far"], where="post", lw=1.5, color=color,
                    label=f"outer fold {int(fold)} best")
            if i < len(folds) - 1:
                ax.axvline(last + 0.5, ls=":", lw=0.9, color="gray")
        ax.set_xlabel("Trial (searches run independently per outer fold)")

    if not multi:
        ax.set_xlabel("Trial")
    ax.set_ylabel("CV RMSE")
    ax.set_title(title, fontsize=10)
    ax.minorticks_on()
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return _save(fig, out)


def plot_param_importance(importance: dict[str, float], out: Path, title: str) -> Path:
    """Horizontal bar chart of Optuna Fanova hyperparameter importance."""
    apply_style()
    if not importance:
        fig, ax = plt.subplots(figsize=(5.0, 2.6), dpi=140)
        ax.text(0.5, 0.5, "hyperparameter importance unavailable", ha="center", va="center")
        ax.axis("off")
        return _save(fig, out)
    items = sorted(importance.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=140)
    ax.barh(names, vals, color=NESTED_COLOR, edgecolor="black", alpha=0.85)
    ax.set_xlabel("Relative importance (Fanova)")
    ax.set_title(title, fontsize=10)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    return _save(fig, out)


def plot_trial_param_scatter(trials_df: pd.DataFrame, out: Path, title: str) -> Path:
    """One panel per hyperparameter: sampled value vs the resulting CV RMSE."""
    apply_style()
    params = [p for p in trials_df.columns
              if p not in ("trial", "cv_rmse", "best_so_far", "state", "outer_fold")]
    n = len(params)
    ncols = 3
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), dpi=140)
    axes = np.atleast_1d(axes).flatten()
    for i, p in enumerate(params):
        ax = axes[i]
        ax.scatter(trials_df[p], trials_df["cv_rmse"], s=10, alpha=0.5, color=TRAIN_COLOR)
        if p in LOG_PARAMS:
            ax.set_xscale("log")
        ax.set_xlabel(p, fontsize=8)
        ax.set_ylabel("CV RMSE", fontsize=8)
        ax.set_title(p, fontsize=9)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out)


def plot_nested_fold_params(fold_df: pd.DataFrame, out: Path, title: str) -> Path:
    """Stability of the hyperparameters chosen independently in each outer fold."""
    apply_style()
    params = [c for c in fold_df.columns
              if c not in ("fold", "inner_cv_rmse", "n_train", "n_test", "RMSE")]
    n = len(params)
    ncols = 3
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), dpi=140)
    axes = np.atleast_1d(axes).flatten()
    for i, p in enumerate(params):
        ax = axes[i]
        vals = pd.to_numeric(fold_df[p], errors="coerce").to_numpy(dtype=float)
        ax.bar(np.arange(len(vals)), vals, 0.6, color=NESTED_COLOR, edgecolor="black", alpha=0.85)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels([f"F{int(f)}" for f in fold_df["fold"]], fontsize=7)
        if p in LOG_PARAMS:
            ax.set_yscale("log")
        ax.set_title(p, fontsize=9)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out)


def plot_nested_fold_scores(fold_df: pd.DataFrame, out: Path, title: str) -> Path:
    """Per outer fold: inner tuning objective, plus the outer RMSE when available."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=140)
    x = np.arange(len(fold_df))
    has_outer = "RMSE" in fold_df.columns and fold_df["RMSE"].notna().any()
    w = 0.38 if has_outer else 0.6
    shift = w / 2 if has_outer else 0.0

    b1 = ax.bar(x - shift, fold_df["inner_cv_rmse"], w, color=TRAIN_COLOR,
                label="inner CV RMSE (tuning objective)")
    if has_outer:
        b2 = ax.bar(x + shift, fold_df["RMSE"].fillna(0.0), w, color=NESTED_COLOR,
                    label="outer fold RMSE (unbiased)")
        _annotate_bars(ax, b2)
    _annotate_bars(ax, b1)
    _autoscale(ax, np.r_[fold_df["inner_cv_rmse"].to_numpy(dtype=float),
                         fold_df["RMSE"].to_numpy(dtype=float) if has_outer else [0.0]])
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{int(f)}" for f in fold_df["fold"]])
    ax.set_ylabel("RMSE")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out)


# Y-scrambling
def plot_y_scrambling(null_df: pd.DataFrame, observed: dict, stats_: dict, out: Path,
                      title: str) -> Path:
    """Null distributions of Pearson r, RMSE and R2 with the observed value marked."""
    apply_style()
    panels = [
        ("Pearson_r", "Pearson r", "#B9D3EA"),
        ("RMSE", "RMSE", "#F3D9A4"),
        ("R2", r"$R^2$", "#CFE3D0"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), dpi=140)
    for ax, (metric, label, color) in zip(axes, panels):
        null = null_df[metric].to_numpy(dtype=float)
        obs = observed[metric]
        ax.hist(null, bins=30, color=color, edgecolor="#5A5A5A", alpha=0.9,
                label=f"scrambled (n={len(null)})")
        ax.axvline(float(np.mean(null)), ls="--", lw=1.3, color="gray",
                   label=f"null mean {np.mean(null):.3f}")
        ax.axvline(obs, lw=2.0, color=ACCENT_COLOR, label=f"observed {obs:.3f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Y-scrambling: {label}", fontsize=10)
        p = stats_.get(f"p_{metric}", float("nan"))
        z = stats_.get(f"z_{metric}", float("nan"))
        ax.text(0.03, 0.97, f"p = {p:.4f}\nz = {z:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, out)


def plot_scrambling_convergence(null_df: pd.DataFrame, observed: dict, out: Path,
                                title: str) -> Path:
    """Cumulative p-value and null mean as the permutation count grows."""
    apply_style()
    n = len(null_df)
    cum_p, cum_mean = [], []
    for k in range(1, n + 1):
        sub = null_df["Pearson_r"].to_numpy(dtype=float)[:k]
        cum_p.append((1 + np.sum(sub >= observed["Pearson_r"])) / (k + 1))
        cum_mean.append(float(np.mean(sub)))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), dpi=140)
    axes[0].plot(np.arange(1, n + 1), cum_p, lw=1.3, color=ACCENT_COLOR)
    axes[0].set_xlabel("Number of permutations")
    axes[0].set_ylabel("permutation p (Pearson r)")
    axes[0].set_title("p-value stability", fontsize=10)
    axes[1].plot(np.arange(1, n + 1), cum_mean, lw=1.3, color=TRAIN_COLOR)
    axes[1].axhline(0.0, ls=":", color="gray", lw=1.0)
    axes[1].set_xlabel("Number of permutations")
    axes[1].set_ylabel("null mean Pearson r")
    axes[1].set_title("null distribution stability", fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, out)


# New molecule predictions
def plot_new_predictions(df: pd.DataFrame, out: Path, title: str) -> Path:
    """Predicted delta PCE per molecule, and predicted PCE against the control."""
    apply_style()
    if "label" in df.columns:
        names = df["label"].astype(str).tolist()
    else:
        names = [str(i + 1) for i in range(len(df))]
    x = np.arange(len(df))
    fig, axes = plt.subplots(2, 1, figsize=(max(7.0, 0.55 * len(df)), 7.0), dpi=140)

    axes[0].bar(x, df["delta_PCE_pred"], 0.65, color=TRAIN_COLOR, edgecolor="black", alpha=0.85)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylabel(f"Predicted {DELTA_COL} (%)")
    axes[0].set_title(f"Predicted {DELTA_COL} per new molecule", fontsize=10)
    axes[0].set_xticks(x)

    w = 0.38
    axes[1].bar(x - w / 2, df[CONTROL_COL], w, color=GREY_COLOR, label=f"{CONTROL_COL} (baseline)")
    axes[1].bar(x + w / 2, df["PCE_pred"], w, color=NESTED_COLOR, label="Predicted PCE")
    axes[1].set_ylabel("PCE (%)")
    axes[1].set_title(f"Predicted PCE vs control ({CONTROL_COL})", fontsize=10)
    axes[1].set_xticks(x)
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.minorticks_on()

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out)


def plot_prediction_comparison(df: pd.DataFrame, out: Path, title: str,
                               value_col: str = "PCE_pred") -> Path:
    """Predictions of all models for the same set of new molecules."""
    apply_style()
    piv = df.pivot_table(index="label", columns="model", values=value_col, aggfunc="mean")
    piv = piv.reindex(columns=[c for c in df["model"].unique() if c in piv.columns])
    fig, ax = plt.subplots(figsize=(max(8.0, 0.5 * len(piv)), 4.6), dpi=140)
    x = np.arange(len(piv))
    n_series = len(piv.columns)
    w = 0.8 / n_series
    cmap = plt.get_cmap("tab10")
    for i, col in enumerate(piv.columns):
        offset = (i - (n_series - 1) / 2) * w
        ax.bar(x + offset, piv[col].to_numpy(dtype=float), w,
               color=cmap(i % 10), label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index.astype(str), rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(value_col.replace("_", " "))
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return _save(fig, out)


# Model comparison
def plot_optimism_gap(gap_df: pd.DataFrame, out: Path, title: str) -> Path:
    """Optimism of the plain-CV estimate relative to nested CV, per split."""
    apply_style()
    fig, axes = plt.subplots(1, len(PRIMARY_METRICS),
                             figsize=(4.4 * len(PRIMARY_METRICS), 4.0), dpi=140)
    x = np.arange(len(gap_df))
    for ax, metric in zip(np.atleast_1d(axes), PRIMARY_METRICS):
        col = f"gap_{metric}"
        vals = pd.to_numeric(gap_df.get(col), errors="coerce").to_numpy(dtype=float)
        bars = ax.bar(x, vals, 0.5, color=ACCENT_COLOR, edgecolor="black", alpha=0.85)
        _autoscale(ax, np.r_[vals, 0.0])
        _annotate_bars(ax, bars)
        ax.axhline(0.0, color="black", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(gap_df["split"].tolist())
        ax.set_ylabel(f"{LABELS.get(metric, metric)} optimism")
        ax.set_title(f"{LABELS.get(metric, metric)}: plain CV vs nested CV", fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, out)


def goodness_matrix(summary: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Per-column min-max score in [0, 1] where 1 always means "better".

    The raw heatmap used one colormap for RMSE (lower is better) and R2 (higher is
    better) at the same time, which painted the best RMSE red. Flipping the sign
    of lower-is-better metrics fixes the direction.
    """
    cols_out = []
    for col in cols:
        metric = next((m for m in PRIMARY_METRICS if col.startswith(f"{m}_")), None)
        v = pd.to_numeric(summary[col], errors="coerce").to_numpy(dtype=float)
        sign = -1.0 if LOWER_IS_BETTER.get(metric, False) else 1.0
        s = sign * v
        finite = s[np.isfinite(s)]
        if finite.size == 0 or np.nanmax(finite) - np.nanmin(finite) < 1e-12:
            cols_out.append(np.full(len(s), 0.5))
        else:
            cols_out.append((s - np.nanmin(finite)) / (np.nanmax(finite) - np.nanmin(finite)))
    return np.array(cols_out).T if cols_out else np.zeros((len(summary), 0))


def plot_metrics_heatmap(summary: pd.DataFrame, out: Path, title: str) -> Path:
    """Heatmap of models x (protocol, metric); green always means better."""
    apply_style()
    cols = [c for c in summary.columns
            if any(c.startswith(f"{m}_") for m in PRIMARY_METRICS) and not c.endswith("_std")]
    if not cols:
        raise ValueError("no metric columns available for the heatmap")
    mat = goodness_matrix(summary, cols)
    values = summary[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.85 * len(cols)), 0.6 * len(summary) + 2.2), dpi=140)
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(summary)))
    ax.set_yticklabels(summary["label"].tolist(), fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("green = better (scaled per column)", fontsize=8, loc="left")
    fig.suptitle(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="relative goodness")
    fig.tight_layout()
    return _save(fig, out)


# SHAP
def plot_shap_importance(importance: pd.DataFrame, out: Path, title: str,
                         top_n: int = 15) -> Path:
    """Horizontal bar chart of mean |SHAP| per feature."""
    apply_style()
    sub = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(sub) + 1.8), dpi=140)
    ax.barh(sub["feature"], sub["mean_abs_shap"], color=NESTED_COLOR,
            edgecolor="black", alpha=0.85)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title, fontsize=10)
    for i, v in enumerate(sub["mean_abs_shap"]):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=7)
    fig.tight_layout()
    return _save(fig, out)


def plot_shap_sign(importance: pd.DataFrame, out: Path, title: str, top_n: int = 15) -> Path:
    """Signed mean SHAP per feature: which way each feature pushes the prediction."""
    apply_style()
    sub = importance.head(top_n).iloc[::-1]
    corr = sub["mean_shap_signed_corr"].to_numpy(dtype=float)
    colors = [TEST_COLOR if c >= 0 else TRAIN_COLOR for c in corr]
    fig, ax = plt.subplots(figsize=(7.0, 0.32 * len(sub) + 1.8), dpi=140)
    ax.barh(sub["feature"], sub["mean_shap"], color=colors, edgecolor="black", alpha=0.85)
    ax.axvline(0.0, color="black", lw=0.9)
    ax.set_xlabel("mean SHAP value (signed)")
    ax.set_title(title, fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TEST_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=TRAIN_COLOR)]
    ax.legend(handles, ["feature correlates positively with delta PCE (>= 0)",
                        "correlates negatively (< 0)"], fontsize=7, loc="lower right")
    fig.tight_layout()
    return _save(fig, out)


def plot_shap_dependence(feature: str, values: np.ndarray, shap_vals: np.ndarray,
                         out: Path, title: str) -> Path:
    """SHAP dependence scatter for one feature."""
    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=140)
    ax.scatter(values, shap_vals, s=16, alpha=0.7, color=TRAIN_COLOR, edgecolor="none")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP value")
    ax.set_title(title, fontsize=10)
    ax.minorticks_on()
    fig.tight_layout()
    return _save(fig, out)


def plot_shap_model_comparison(importance: pd.DataFrame, out: Path, title: str,
                               top_n: int = 10) -> Path:
    """Grouped bars of mean |SHAP| for the union of each model's top features."""
    apply_style()
    piv = importance.pivot_table(index="feature", columns="model",
                                 values="mean_abs_shap", aggfunc="mean")
    order = (importance.groupby("feature")["mean_abs_shap"].max()
             .sort_values(ascending=False).head(top_n).index)
    piv = piv.loc[[f for f in order if f in piv.index]]
    x = np.arange(len(piv))
    n_series = len(piv.columns)
    w = 0.8 / n_series
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(max(8.0, 0.6 * len(piv)), 4.8), dpi=140)
    for i, col in enumerate(piv.columns):
        offset = (i - (n_series - 1) / 2) * w
        ax.bar(x + offset, piv[col].to_numpy(dtype=float), w,
               color=cmap(i % 10), label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index.astype(str), rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("mean |SHAP value|")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return _save(fig, out)
