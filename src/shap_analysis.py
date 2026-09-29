"""Entry script 4/6: SHAP interpretation of the trained models.

Single responsibility: explain *why* each model predicts what it predicts.
No training, no tuning, no metric computation.

For every model it computes TreeExplainer SHAP values on the full dataset using
the deployment pipeline (scaler + booster), then writes

    outputs/<model>/shap/shap_importance.csv   mean |SHAP| and signed mean SHAP
    outputs/<model>/shap/shap_values.csv       per-row, per-feature SHAP values
    outputs/<model>/shap/shap_summary.json     top features and the SHAP baseline
    outputs/<model>/figures/shap_*.png         bar / sign / beeswarm / dependence

and a cross-model comparison under outputs/_comparison/.

Usage:
    conda activate rdkit
    python src/shap_analysis.py
    python src/shap_analysis.py --models group_nested --top-n 12
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import plotting
from data_utils import build_matrices, load_dataset
from log_utils import install_excepthook, setup_logger
from metrics_utils import to_native
from model_utils import load_deployment_model


def extract_shap_values(pipeline, X: np.ndarray) -> tuple[np.ndarray, float]:
    """SHAP values for a fitted scaler+model pipeline.

    The tree splits live in scaled space, so the scaled matrix must be passed to
    the booster; feature attribution is unaffected because scaling is a monotone
    per-feature transform.
    """
    X_scaled = pipeline.named_steps["scaler"].transform(X)
    booster = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(booster)
    raw = explainer(X_scaled)
    values = np.asarray(getattr(raw, "values", raw), dtype=float)
    if values.ndim == 3:                     # some estimators return (n, f, output)
        values = values[:, :, 0]
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    return values, base


def importance_table(shap_values: np.ndarray, X: np.ndarray,
                     feature_names: list[str]) -> pd.DataFrame:
    """Per-feature SHAP summary ranked by mean |SHAP|."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    rows = []
    for j, name in enumerate(feature_names):
        xj = X[:, j]
        if np.std(xj) < 1e-12 or np.std(shap_values[:, j]) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(xj, shap_values[:, j])[0, 1])
        rows.append({
            "feature": name,
            "mean_abs_shap": float(mean_abs[j]),
            "mean_shap": float(mean_signed[j]),
            "std_abs_shap": float(np.abs(shap_values[:, j]).std(ddof=1)),
            "mean_shap_signed_corr": corr,
        })
    return (pd.DataFrame(rows)
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))


def analyse_model(key: str, X: np.ndarray, feature_names: list[str],
                  top_n: int, logger) -> pd.DataFrame | None:
    """Compute and persist SHAP artefacts for one model."""
    mpath = cfg.model_path(key)
    if not mpath.is_file():
        logger.warning("[%s] %s not found; run src/train.py first", key, mpath.name)
        return None

    pipe, bundle = load_deployment_model(mpath, cfg.model_bundle_path(key))
    if list(bundle.get("feature_names", [])) != list(feature_names):
        logger.warning("[%s] feature names differ from the current dataset; "
                       "using the names stored in the model bundle", key)
        feature_names = list(bundle["feature_names"])

    shap_values, base = extract_shap_values(pipe, X)
    importance = importance_table(shap_values, X, feature_names)

    cfg.ensure_model_dirs(key)
    shap_dir = cfg.shap_dir(key)
    pd.DataFrame(shap_values, columns=feature_names).to_csv(cfg.shap_values_path(key), index=False)
    importance.to_csv(cfg.shap_importance_path(key), index=False)
    cfg.shap_summary_path(key).write_text(
        json.dumps(to_native({
            "model": key,
            "label": bundle.get("label", key),
            "n_rows": int(shap_values.shape[0]),
            "n_features": int(shap_values.shape[1]),
            "expected_value": base,
            "top_features": importance.head(top_n)["feature"].tolist(),
            "top_mean_abs_shap": importance.head(top_n)["mean_abs_shap"].tolist(),
        }), indent=2), encoding="utf-8")

    fig_dir = cfg.figure_dir(key)
    short = cfg.SPEC_BY_KEY[key].short
    plotting.plot_shap_importance(importance, fig_dir / "shap_importance.png",
                                  f"{short}: SHAP feature importance")
    plotting.plot_shap_sign(importance, fig_dir / "shap_sign.png",
                            f"{short}: signed mean SHAP")
    plot_beeswarm(shap_values, X, feature_names, fig_dir / "shap_beeswarm.png",
                  f"{short}: SHAP beeswarm")
    for j, name in enumerate(importance.head(2)["feature"]):
        idx = feature_names.index(name)
        plotting.plot_shap_dependence(
            name, X[:, idx], shap_values[:, idx],
            fig_dir / f"shap_dependence_{name}.png",
            f"{short}: SHAP dependence on {name}")

    top = importance.head(3)
    logger.info("[%s] top SHAP features: %s", key,
                ", ".join(f"{r.feature}={r.mean_abs_shap:.3f}" for r in top.itertuples()))

    out = importance.copy()
    out.insert(0, "model", key)
    out.insert(1, "model_label", bundle.get("label", key))
    return out


def plot_beeswarm(shap_values: np.ndarray, X: np.ndarray, feature_names: list[str],
                  out: Path, title: str) -> Path | None:
    """Beeswarm summary rendered by shap itself, saved headlessly."""
    plotting.apply_style()
    try:
        shap.summary_plot(shap_values, pd.DataFrame(X, columns=feature_names),
                          feature_names=feature_names, show=False,
                          max_display=cfg.SHAP_MAX_DISPLAY)
        fig = plt.gcf()
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        return plotting._save(fig, out)
    except Exception as exc:
        plt.close("all")
        logging.getLogger(__name__).warning("beeswarm plot failed (%s); falling back to bar chart",
                                            exc)
        return plotting.plot_shap_importance(
            importance_table(shap_values, X, feature_names), out, title)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP analysis for the four models")
    p.add_argument("--models", nargs="*", default=None, choices=cfg.MODEL_KEYS)
    p.add_argument("--top-n", type=int, default=cfg.TOP_N_SHAP)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logger = setup_logger(cfg.OUTPUT_DIR / "shap_analysis.log")
    install_excepthook(logger)

    df = load_dataset()
    X, y, ctrl, groups, feats = build_matrices(df)
    logger.info("SHAP on %d rows x %d features", X.shape[0], X.shape[1])

    keys = args.models or cfg.MODEL_KEYS
    frames = []
    for key in keys:
        try:
            out = analyse_model(key, X, feats, args.top_n, logger)
            if out is not None:
                frames.append(out)
        except Exception:
            logger.exception("[FAILED] %s", key)
            plt.close("all")

    if not frames:
        logger.error("no model produced SHAP output")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    comp_dir = cfg.COMPARISON_DIR
    comp_dir.mkdir(parents=True, exist_ok=True)
    (comp_dir / "figures").mkdir(parents=True, exist_ok=True)
    combined.to_csv(comp_dir / "shap_importance_all_models.csv", index=False)

    plotting.plot_shap_model_comparison(
        combined, comp_dir / "figures" / "shap_top_features_comparison.png",
        "SHAP feature importance across the four models", top_n=args.top_n)

    # Consensus ranking: mean rank of |SHAP| across models (1 = most important).
    combined["rank"] = combined.groupby("model")["mean_abs_shap"].rank(ascending=False)
    consensus = (combined.groupby("feature")["rank"].mean()
                 .sort_values().rename("mean_rank").reset_index())
    consensus.to_csv(comp_dir / "shap_consensus_ranking.csv", index=False)

    logger.info("cross-model consensus ranking (1 = most important):")
    for r in consensus.head(args.top_n).itertuples():
        logger.info("    %2d. %-24s mean rank %.2f", r.Index + 1, r.feature, r.mean_rank)
    logger.info("SHAP output -> %s", comp_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
