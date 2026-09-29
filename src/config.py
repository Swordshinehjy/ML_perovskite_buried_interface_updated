"""Global configuration: paths, seeds, modelling target and the four model specs.

Single responsibility: declare *what* things are (constants and data classes).
Every other module imports its settings from here so that no constant is
hard-coded twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
OUTPUT_DIR = ROOT / "outputs"
COMPARISON_DIR = OUTPUT_DIR / "_comparison"

DATASET_CSV = ROOT / "dataset.csv"
NEW_MOL_CSV = ROOT / "new-mol.csv"

SMILES_COL = "SMILES"
TARGET_COL = "PCE"
CONTROL_COL = "control_PCE"
GROUP_COL = "canonical_smiles"
DELTA_COL = "delta_PCE"

# The control efficiency is both the reconstruction baseline and a feature.
KEEP_CONTROL_AS_FEATURE = True
RECONSTRUCTION = f"PCE = {DELTA_COL} + {CONTROL_COL}"

SEED = 42
TEST_SIZE = 0.15
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 5

N_TRIALS = 150
N_TRIALS_INNER = 60
N_SCRAMBLES = 200

TOP_N_SHAP = 10
SHAP_MAX_DISPLAY = 15

SPLIT_RANDOM = "random"
SPLIT_GROUP = "group"
CV_PLAIN = "cv"
CV_NESTED = "nested"

PRIMARY_METRICS = ("RMSE", "Pearson_r", "R2")
LOWER_IS_BETTER = {
    "RMSE": True,
    "MAE": True,
    "Pearson_r": False,
    "R2": False,
    "Spearman_rho": False,
}


def validate_split(split: str) -> str:
    """Reject unknown split strategies instead of silently falling back to random."""
    if split not in (SPLIT_RANDOM, SPLIT_GROUP):
        raise ValueError(f"unknown split strategy {split!r}; expected 'random' or 'group'")
    return split


def validate_cv(cv: str) -> str:
    """Reject unknown cross-validation protocols."""
    if cv not in (CV_PLAIN, CV_NESTED):
        raise ValueError(f"unknown cv protocol {cv!r}; expected 'cv' or 'nested'")
    return cv


@dataclass(frozen=True)
class ModelSpec:
    """One model = one data splitting strategy + one cross-validation protocol."""

    key: str
    split: str
    cv: str
    label: str
    short: str

    @property
    def is_group(self) -> bool:
        return self.split == SPLIT_GROUP

    @property
    def is_nested(self) -> bool:
        return self.cv == CV_NESTED

    @property
    def fold_name(self) -> str:
        return "GroupKFold" if self.is_group else "KFold"

    def __post_init__(self) -> None:
        validate_split(self.split)
        validate_cv(self.cv)


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("random_cv", SPLIT_RANDOM, CV_PLAIN, "Random split + CV", "Random + CV"),
    ModelSpec("random_nested", SPLIT_RANDOM, CV_NESTED, "Random split + NestedCV", "Random + Nested"),
    ModelSpec("group_cv", SPLIT_GROUP, CV_PLAIN, "Group split + CV", "Group + CV"),
    ModelSpec("group_nested", SPLIT_GROUP, CV_NESTED, "Group split + NestedCV", "Group + Nested"),
)

MODEL_KEYS: list[str] = [m.key for m in MODEL_SPECS]
SPEC_BY_KEY: dict[str, ModelSpec] = {m.key: m for m in MODEL_SPECS}

# Under the same split, plain CV and nested CV are counterparts; the pair is used
# to quantify how optimistic the plain-CV estimate is.
PAIR_BY_SPLIT: dict[str, dict[str, str]] = {
    SPLIT_RANDOM: {CV_PLAIN: "random_cv", CV_NESTED: "random_nested"},
    SPLIT_GROUP: {CV_PLAIN: "group_cv", CV_NESTED: "group_nested"},
}


def model_dir(key: str) -> Path:
    return OUTPUT_DIR / key


def figure_dir(key: str) -> Path:
    return model_dir(key) / "figures"


def params_path(key: str) -> Path:
    """Hyperparameter json written by hyperparam_tuning.py, read by train.py."""
    return model_dir(key) / "best_params.json"


def nested_folds_path(key: str) -> Path:
    """Per-outer-fold hyperparameters selected inside nested CV (nested models only)."""
    return model_dir(key) / "nested_folds.json"


def tuning_trials_path(key: str) -> Path:
    return model_dir(key) / "tuning_trials.csv"


def metrics_path(key: str) -> Path:
    return model_dir(key) / "metrics.json"


def folds_path(key: str) -> Path:
    return model_dir(key) / "cv_folds.csv"


def model_path(key: str) -> Path:
    """Deployment model refitted on all rows (XGBoost native JSON, code-free)."""
    return model_dir(key) / "model.json"


def model_bundle_path(key: str) -> Path:
    """JSON sidecar with the fitted scaler statistics and model metadata."""
    return model_dir(key) / "model_bundle.json"


def predictions_path(key: str) -> Path:
    return model_dir(key) / "new_molecule_predictions.csv"


def scramble_csv_path(key: str) -> Path:
    return model_dir(key) / "y_scrambling.csv"


def scramble_summary_path(key: str) -> Path:
    return model_dir(key) / "y_scrambling_summary.json"


def shap_dir(key: str) -> Path:
    return model_dir(key) / "shap"


def shap_importance_path(key: str) -> Path:
    return shap_dir(key) / "shap_importance.csv"


def shap_values_path(key: str) -> Path:
    return shap_dir(key) / "shap_values.csv"


def shap_summary_path(key: str) -> Path:
    return shap_dir(key) / "shap_summary.json"


def ensure_model_dirs(key: str) -> Path:
    d = model_dir(key)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    (d / "shap").mkdir(parents=True, exist_ok=True)
    return d
