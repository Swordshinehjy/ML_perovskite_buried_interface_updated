"""Optuna hyperparameter search plus reading and writing the parameter json.

Single responsibility: answer "how do I search for hyperparameters on this data"
and "how are parameters persisted". Shared by the tuning entry point and by the
nested-CV path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from config import SEED
from metrics_utils import regression_metrics, to_native
from model_utils import OBJECTIVE, SEARCH_SPACE, build_pipeline, build_xgb, suggest_params

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _fold_rmse(params: dict, X: np.ndarray, y: np.ndarray, folds) -> float:
    """Mean per-fold RMSE for a fixed parameter set (same convention as cross_val_score)."""
    scores = []
    for tr, te in folds:
        pipe = build_pipeline(build_xgb(params))
        pipe.fit(X[tr], y[tr])
        scores.append(regression_metrics(y[te], pipe.predict(X[te]))["RMSE"])
    return float(np.mean(scores))


def run_optuna_search(X: np.ndarray, y: np.ndarray, folds, n_trials: int, seed: int = SEED,
                      study_name: str | None = None, progress=None) -> optuna.Study:
    """Run one complete Optuna search; objective is the mean in-fold RMSE."""
    def objective(trial: optuna.Trial) -> float:
        return _fold_rmse(suggest_params(trial), X, y, folds)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed),
                                study_name=study_name)
    callbacks = []
    if progress is not None:
        def _cb(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            progress(trial.number + 1, n_trials, study_.best_value)
        callbacks.append(_cb)

    study.optimize(objective, n_trials=n_trials, callbacks=callbacks, show_progress_bar=False)
    return study


def trials_to_frame(study: optuna.Study) -> pd.DataFrame:
    """Flatten a study into one row per trial for archiving and plotting."""
    rows = []
    best = np.inf
    for t in study.trials:
        val = t.value
        if val is not None:
            best = min(best, val)
        row = {"trial": t.number + 1, "cv_rmse": val,
               "best_so_far": best if val is not None else np.nan, "state": t.state.name}
        row.update(t.params)
        rows.append(row)
    return pd.DataFrame(rows)


def param_importance_from_trials(trials: pd.DataFrame) -> dict[str, float]:
    """Rebuild a study from archived trials and ask Optuna's Fanova evaluator.

    Distributions must be supplied explicitly, otherwise Optuna rejects the
    reconstructed trials and no importance can be computed.
    """
    from optuna import create_study
    from optuna.importance import get_param_importances
    from optuna.trial import TrialState, create_trial

    from model_utils import build_distributions

    param_cols = [c for c in trials.columns
                  if c not in ("trial", "cv_rmse", "best_so_far", "state", "outer_fold")]
    dists = build_distributions()
    study = create_study(direction="minimize")
    for _, row in trials.iterrows():
        if pd.isna(row["cv_rmse"]):
            continue
        study.add_trial(create_trial(
            params={c: row[c] for c in param_cols if c in dists},
            distributions=dists,
            value=float(row["cv_rmse"]),
            state=TrialState.COMPLETE,
        ))
    if len(study.trials) < 2:
        return {}
    return dict(get_param_importances(study))


def save_params(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_native(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def load_params(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"hyperparameter file not found: {path}; run src/hyperparam_tuning.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def make_params_payload(model_key: str, params: dict, *, split: str, cv: str,
                        best_cv_rmse: float, n_trials: int, seed: int,
                        selection: str) -> dict:
    """Unified parameter json layout; train.py only reads the 'params' entry."""
    return {
        "model": model_key,
        "estimator": "XGBRegressor",
        "objective": OBJECTIVE,
        "split": split,
        "cv": cv,
        "param_selection": selection,
        "best_cv_rmse": float(best_cv_rmse),
        "n_trials": int(n_trials),
        "seed": int(seed),
        "search_space": SEARCH_SPACE,
        "params": dict(params),
    }
