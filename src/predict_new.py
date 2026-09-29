"""Entry script 3/6: predict new molecules with the trained models.

Loads outputs/<model>/model.json + model_bundle.json (refitted on all rows),
predicts the screening table, writes the delta PCE and the reconstructed PCE,
and draws per-model and cross-model figures. No training, no tuning.

Usage:
    conda activate rdkit
    python src/predict_new.py
    python src/predict_new.py --input data/new.csv --models group_nested
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
import plotting
from data_utils import align_features, load_new_molecules
from log_utils import install_excepthook, setup_logger
from model_utils import load_deployment_model


def molecule_labels(df_new: pd.DataFrame) -> np.ndarray:
    """Short, unique labels M01, M02, ... for the x axis.

    Truncated SMILES must not be used: several molecules share the same prefix,
    which would collapse rows when the table is pivoted for comparison.
    """
    n = len(df_new)
    width = max(2, len(str(n)))
    return np.array([f"M{i + 1:0{width}d}" for i in range(n)])


def predict_one(key: str, df_new: pd.DataFrame, logger) -> pd.DataFrame | None:
    mpath = cfg.model_path(key)
    bpath = cfg.model_bundle_path(key)
    if not mpath.is_file() or not bpath.is_file():
        logger.warning("[%s] %s not found; run src/train.py first", key, mpath.name)
        return None

    pipe, bundle = load_deployment_model(mpath, bpath)
    X_new = align_features(df_new, list(bundle["feature_names"]))

    delta_pred = np.asarray(pipe.predict(X_new), dtype=float)
    if cfg.CONTROL_COL in df_new.columns:
        ctrl = df_new[cfg.CONTROL_COL].to_numpy(dtype=float)
    else:
        logger.warning("[%s] screening table has no %s column; PCE falls back to delta PCE",
                       key, cfg.CONTROL_COL)
        ctrl = np.zeros_like(delta_pred)

    out = pd.DataFrame({
        "SMILES": df_new[cfg.SMILES_COL].to_numpy(),
        "label": (df_new["label"].astype(str).to_numpy() if "label" in df_new.columns
                  else molecule_labels(df_new)),
        "model": key,
        "model_label": bundle.get("label", key),
        cfg.CONTROL_COL: ctrl,
        "delta_PCE_pred": delta_pred,
        "PCE_pred": delta_pred + ctrl,
    })
    out.to_csv(cfg.predictions_path(key), index=False)

    logger.info("[%s] delta PCE range %.3f to %.3f (mean %.3f)",
                key, delta_pred.min(), delta_pred.max(), delta_pred.mean())
    plotting.plot_new_predictions(
        out, cfg.figure_dir(key) / "new_molecules.png",
        f"{cfg.SPEC_BY_KEY[key].short}: screening predictions")
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screen new molecules with the four models")
    p.add_argument("--input", default=str(cfg.NEW_MOL_CSV))
    p.add_argument("--models", nargs="*", default=None, choices=cfg.MODEL_KEYS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logger = setup_logger(cfg.OUTPUT_DIR / "predict_new.log")
    install_excepthook(logger)

    df_new = load_new_molecules(args.input)
    logger.info("screening table: %d molecules (%d unique)",
                len(df_new), df_new[cfg.GROUP_COL].nunique())

    keys = args.models or cfg.MODEL_KEYS
    frames = []
    for key in keys:
        cfg.ensure_model_dirs(key)
        try:
            out = predict_one(key, df_new, logger)
            if out is not None:
                frames.append(out)
        except Exception:
            logger.exception("[FAILED] %s", key)

    if not frames:
        logger.error("no model produced predictions")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    comp_dir = cfg.COMPARISON_DIR
    comp_dir.mkdir(parents=True, exist_ok=True)
    (comp_dir / "figures").mkdir(parents=True, exist_ok=True)
    combined.to_csv(comp_dir / "new_molecule_predictions_all_models.csv", index=False)

    wide = combined.pivot_table(index="label", columns="model", values="PCE_pred",
                                aggfunc="mean").reset_index()
    wide.columns.name = None
    wide = wide.reindex(columns=["label"] + [k for k in keys if k in wide.columns])
    wide.to_csv(comp_dir / "new_molecule_PCE_pred_wide.csv", index=False)

    plotting.plot_prediction_comparison(
        combined, comp_dir / "figures" / "new_molecules_all_models.png",
        "Predicted PCE of the screened molecules", value_col="PCE_pred")
    plotting.plot_prediction_comparison(
        combined, comp_dir / "figures" / "new_molecules_delta_all_models.png",
        "Predicted delta PCE of the screened molecules", value_col="delta_PCE_pred")

    for r in combined.drop_duplicates("label").itertuples():
        logger.info("    %s  %s", r.label, r.SMILES)
    logger.info("delta PCE predictions by model:")
    delta_wide = combined.pivot_table(index="label", columns="model",
                                      values="delta_PCE_pred", aggfunc="mean")
    delta_wide = delta_wide.reindex(columns=[k for k in keys if k in delta_wide.columns])
    for line in delta_wide.round(3).to_string().splitlines():
        logger.info(line)
    logger.info("saved -> %s", comp_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
