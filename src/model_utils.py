"""XGBoost estimator factory: search space, parameter sampling, model and pipeline.

Single responsibility: answer "how do I get an estimator from a parameter dict".
Tuning and training share this module so both agree on the parameter definition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import SEED
from metrics_utils import to_native

SEARCH_SPACE_FILE = Path(__file__).resolve().parent.parent / "xgb_search_space.json"

BUILTIN_SEARCH_SPACE: dict[str, dict] = {
    "n_estimators":     {"type": "int",   "low": 100,  "high": 800, "step": 50},
    "max_depth":        {"type": "int",   "low": 2,    "high": 8},
    "learning_rate":    {"type": "float", "low": 1e-3, "high": 2e-1, "log": True},
    "subsample":        {"type": "float", "low": 0.5,  "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.5,  "high": 1.0},
    "min_child_weight": {"type": "int",   "low": 1,    "high": 10},
    "gamma":            {"type": "float", "low": 1e-4, "high": 1e0, "log": True},
    "reg_alpha":        {"type": "float", "low": 1e-4, "high": 1e1, "log": True},
    "reg_lambda":       {"type": "float", "low": 1e-4, "high": 1e1, "log": True},
}

ESTIMATOR_NAME = "XGBRegressor"
OBJECTIVE = "reg:squarederror"

# Parameters best shown on a log axis in tuning plots.
LOG_PARAMS = {"learning_rate", "gamma", "reg_alpha", "reg_lambda"}

LABELS = {
    "RMSE": "RMSE",
    "MAE": "MAE",
    "Pearson_r": "Pearson r",
    "R2": "$R^2$",
    "Spearman_rho": r"Spearman $\rho$",
}


def is_int_param(name: str) -> bool:
    spec = SEARCH_SPACE[name]
    return spec.get("type") == "int" or bool(spec.get("step"))


def load_search_space(path: Path | None = None) -> dict[str, dict]:
    """Built-in ranges, overridden per key by xgb_search_space.json if present.

    Called once at import time (see SEARCH_SPACE below). Overriding the module
    global later would not be visible to modules that already did
    ``from model_utils import SEARCH_SPACE``, so the override must happen before
    the name is ever bound elsewhere.
    """
    path = Path(path) if path is not None else SEARCH_SPACE_FILE
    space = {name: dict(spec) for name, spec in BUILTIN_SEARCH_SPACE.items()}
    if not path.is_file():
        return space
    external = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(external) - set(space))
    if unknown:
        raise ValueError(f"{path.name} declares unknown hyperparameters: {unknown}")
    for name, spec in external.items():
        merged = dict(space[name])
        merged.update(spec)
        space[name] = merged
    return space


SEARCH_SPACE: dict[str, dict] = load_search_space()


def _suggest_int(trial: optuna.Trial, name: str, spec: dict) -> int:
    lo, hi = int(spec["low"]), int(spec["high"])
    if lo == hi:
        return lo
    step = spec.get("step")
    return trial.suggest_int(name, lo, hi, step=int(step)) if step else trial.suggest_int(name, lo, hi)


def _suggest_float(trial: optuna.Trial, name: str, spec: dict) -> float:
    lo, hi = float(spec["low"]), float(spec["high"])
    if lo == hi:
        return lo
    return trial.suggest_float(name, lo, hi, log=True) if spec.get("log") else trial.suggest_float(name, lo, hi)


def suggest_params(trial: optuna.Trial) -> dict:
    """Sample one hyperparameter set for a trial."""
    return {
        name: (_suggest_int(trial, name, spec) if is_int_param(name)
               else _suggest_float(trial, name, spec))
        for name, spec in SEARCH_SPACE.items()
    }


def build_xgb(params: dict, seed: int = SEED) -> XGBRegressor:
    """Build an XGBRegressor from a complete parameter dict."""
    return XGBRegressor(
        objective=OBJECTIVE,
        random_state=seed,
        n_jobs=1,               # fold-level parallelism is controlled upstream
        verbosity=0,
        **params,
    )


def build_from_trial(trial: optuna.Trial, seed: int = SEED) -> XGBRegressor:
    return build_xgb(suggest_params(trial), seed=seed)


def build_pipeline(model: XGBRegressor) -> Pipeline:
    """Scaler inside the pipeline so it is fitted on the training fold only."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def build_pipeline_from_params(params: dict, seed: int = SEED) -> Pipeline:
    return build_pipeline(build_xgb(params, seed=seed))


def save_deployment_model(pipe: Pipeline, meta: dict, model_file: Path,
                          bundle_file: Path) -> None:
    """Persist a fitted scaler+booster pipeline in a code-free format.

    The booster is written with XGBoost's own native JSON serialisation and the
    fitted StandardScaler statistics plus the metadata go into a plain JSON
    sidecar. Neither file contains pickled objects, so loading them cannot
    execute arbitrary code (unlike pickle/joblib artefacts).
    """
    pipe.named_steps["model"].save_model(model_file)
    scaler = pipe.named_steps["scaler"]
    payload = {
        "format": "xgboost-json+scaler",
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
        },
        **to_native(meta),
    }
    bundle_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_deployment_model(model_file: Path, bundle_file: Path) -> tuple[Pipeline, dict]:
    """Rebuild the scaler+booster pipeline from the safe JSON artefacts."""
    meta = json.loads(bundle_file.read_text(encoding="utf-8"))
    model = XGBRegressor()
    model.load_model(model_file)
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(meta["scaler"]["mean"], dtype=float)
    scaler.scale_ = np.asarray(meta["scaler"]["scale"], dtype=float)
    scaler.var_ = np.asarray(meta["scaler"]["var"], dtype=float)
    scaler.n_features_in_ = len(scaler.mean_)
    return Pipeline([("scaler", scaler), ("model", model)]), meta


def build_distributions(space: dict | None = None) -> dict:
    """Translate the search space into Optuna distributions.

    Needed to rebuild a study from archived trials, which is how hyperparameter
    importance is computed without keeping the live study object around.
    """
    space = space if space is not None else SEARCH_SPACE
    dists: dict = {}
    for name, spec in space.items():
        lo, hi = spec["low"], spec["high"]
        if spec.get("type") == "int" or spec.get("step"):
            dists[name] = optuna.distributions.IntDistribution(
                low=int(lo), high=int(hi), step=int(spec.get("step", 1)))
        else:
            dists[name] = optuna.distributions.FloatDistribution(
                low=float(lo), high=float(hi), log=bool(spec.get("log")))
    return dists
