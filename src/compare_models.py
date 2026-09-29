"""Entry script 6/6: compare the four models.

Aggregation only: reads outputs/<model>/metrics.json and writes

    outputs/_comparison/model_comparison.csv           metrics of all four models
    outputs/_comparison/optimism_gap.csv               plain CV vs nested CV
    outputs/_comparison/comparison_report.md           tables ready for a report
    outputs/_comparison/figures/metrics_comparison.png RMSE / r / R2 grouped bars
    outputs/_comparison/figures/optimism_gap.png       optimism of plain CV
    outputs/_comparison/figures/metrics_heatmap.png    green = better heatmap

No training, no tuning, no prediction.

Usage:
    conda activate rdkit
    python src/compare_models.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import plotting
from log_utils import install_excepthook, setup_logger


def to_markdown(df: pd.DataFrame) -> str:
    """DataFrame to a Markdown table, with a built-in fallback when tabulate is absent."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        head = "| " + " | ".join(map(str, df.columns)) + " |"
        sep = "| " + " | ".join("---" for _ in df.columns) + " |"
        body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
                for row in df.itertuples(index=False)]
        return "\n".join([head, sep] + body)


def load_metrics(key: str) -> dict | None:
    p = cfg.metrics_path(key)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def build_comparison(records: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key in cfg.MODEL_KEYS:
        m = records.get(key)
        if m is None:
            continue
        row = {"model": key, "label": m["label"], "split": m["split"],
               "cv_protocol": m["protocol"], "n_train": m["n_train"], "n_test": m["n_test"]}
        for tag, block in (("train", m["train"]), ("test", m["test"]),
                           ("cv", m["cv"]), ("nested", m["nested"])):
            if not block:
                continue
            for metric in cfg.PRIMARY_METRICS:
                row[f"{metric}_{tag}"] = round(float(block[metric]), 4)
        for tag, folds_key in (("cv", "cv_per_fold"), ("nested", "nested_per_fold")):
            folds = m.get(folds_key)
            if folds:
                fdf = pd.DataFrame(folds)
                for metric in cfg.PRIMARY_METRICS:
                    row[f"{metric}_{tag}_std"] = round(float(fdf[metric].std(ddof=1)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def build_gap_table(records: dict[str, dict]) -> pd.DataFrame:
    """Optimism of the plain-CV estimate, using nested CV as the unbiased reference.

    The gap is sign-corrected so that a positive value always means "plain CV
    looks better": nested - plain for RMSE, plain - nested for r and R2.
    """
    rows = []
    for split, pair in cfg.PAIR_BY_SPLIT.items():
        plain, nested = records.get(pair[cfg.CV_PLAIN]), records.get(pair[cfg.CV_NESTED])
        if not plain or not nested or not nested.get("nested"):
            continue
        row = {"split": split}
        for metric in cfg.PRIMARY_METRICS:
            p = float(plain["cv"][metric])
            n = float(nested["nested"][metric])
            diff = (n - p) if cfg.LOWER_IS_BETTER.get(metric, False) else (p - n)
            row[f"gap_{metric}"] = round(diff, 4)
            row[f"plain_{metric}"] = round(p, 4)
            row[f"nested_{metric}"] = round(n, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, gaps: pd.DataFrame, out_md: Path) -> None:
    lines = ["# Comparison of the four XGBoost models", ""]
    lines += [
        f"- Target: `{cfg.DELTA_COL} = {cfg.TARGET_COL} - {cfg.CONTROL_COL}`; "
        f"reconstruction `{cfg.RECONSTRUCTION}`",
        "- `*_cv`     : plain CV out-of-fold estimate, tuned on the same folds, **optimistic**",
        "- `*_nested` : nested CV out-of-fold estimate, outer folds never used for tuning, **unbiased**",
        "- `*_test`   : held-out rows removed before training, identical protocol for all models",
        "",
    ]

    cols = ["label"] + [f"{m}_{t}" for m in cfg.PRIMARY_METRICS for t in ("cv", "nested", "test")]
    cols = [c for c in cols if c in summary.columns]
    lines += ["## 1. Primary metrics (RMSE / Pearson r / R2)", "", to_markdown(summary[cols]), ""]

    std_cols = [c for c in summary.columns if c.endswith("_std")]
    if std_cols:
        lines += ["## 2. Between-fold standard deviation (stability)", "",
                  to_markdown(summary[["label"] + std_cols]), ""]

    if not gaps.empty:
        lines += ["## 3. Optimism of plain CV (nested CV as reference)", "", to_markdown(gaps), ""]
        lines += ["The gap is sign-corrected: positive always means plain CV looks better "
                  "(nested - plain for RMSE, plain - nested for r and R2)."]

    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare the four models")
    p.add_argument("--models", nargs="*", default=None, choices=cfg.MODEL_KEYS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logger = setup_logger(cfg.OUTPUT_DIR / "compare_models.log")
    install_excepthook(logger)

    keys = args.models or cfg.MODEL_KEYS
    records = {k: load_metrics(k) for k in keys}
    missing = [k for k, v in records.items() if v is None]
    if missing:
        logger.warning("metrics.json missing for %s; run src/train.py first", missing)
    records = {k: v for k, v in records.items() if v is not None}
    if not records:
        logger.error("no metrics.json found, nothing to compare")
        return 1

    cfg.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = cfg.COMPARISON_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = build_comparison(records)
    summary.to_csv(cfg.COMPARISON_DIR / "model_comparison.csv", index=False)

    logger.info("primary metrics (cv / nested are CV estimates, test is the held-out set)")
    show = [c for c in summary.columns if c not in ("model", "split", "cv_protocol")]
    for line in summary[show].to_string(index=False).splitlines():
        logger.info(line)

    series = [("{metric}_cv", "CV (plain)"), ("{metric}_nested", "Nested CV"),
              ("{metric}_test", "Test")]
    plotting.plot_grouped_metric_bars(
        summary, fig_dir / "metrics_comparison.png",
        "RMSE / Pearson r / R2 by evaluation protocol", series)

    gaps = build_gap_table(records)
    if not gaps.empty:
        gaps.to_csv(cfg.COMPARISON_DIR / "optimism_gap.csv", index=False)
        plotting.plot_optimism_gap(gaps, fig_dir / "optimism_gap.png",
                                   "Optimism of plain CV (positive = plain CV looks better)")
        logger.info("optimism of plain CV (positive = plain CV looks better)")
        for line in gaps.to_string(index=False).splitlines():
            logger.info(line)

    plotting.plot_metrics_heatmap(summary, fig_dir / "metrics_heatmap.png",
                                  "Metrics heatmap: models x (protocol, metric)")
    write_report(summary, gaps, cfg.COMPARISON_DIR / "comparison_report.md")
    logger.info("comparison -> %s", cfg.COMPARISON_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
