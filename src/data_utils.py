"""Data loading, canonical-SMILES grouping, train/test splitting and fold builders.

Single responsibility: turn a csv file into (X, y, groups, folds). No models, no metrics.

Why grouping matters: the same molecule (identical canonical SMILES) can have
several replicate measurements. A random split may place replicates on both sides
of the split, letting the model "read the answer" from its twin. Group splitting
guarantees that every replicate of a molecule lands on the same side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split

from config import (
    CONTROL_COL,
    DATASET_CSV,
    DELTA_COL,
    GROUP_COL,
    KEEP_CONTROL_AS_FEATURE,
    NEW_MOL_CSV,
    N_SPLITS_INNER,
    SEED,
    SMILES_COL,
    SPLIT_GROUP,
    SPLIT_RANDOM,
    TARGET_COL,
    TEST_SIZE,
    validate_split,
)

RDLogger.DisableLog("rdApp.*")

Folds = list[tuple[np.ndarray, np.ndarray]]


def canonical_smiles(smiles: str) -> str | None:
    """Normalise any SMILES to its RDKit canonical form (None if unparsable)."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """Read dataset.csv and attach the group column plus the regression target."""
    df = pd.read_csv(path or DATASET_CSV)
    for col in (TARGET_COL, CONTROL_COL, SMILES_COL):
        if col not in df.columns:
            raise ValueError(f"dataset is missing required column {col!r}")

    df[GROUP_COL] = df[SMILES_COL].map(canonical_smiles)
    unparsable = df[GROUP_COL].isna()
    if unparsable.any():
        # Fall back to the raw string so the row is still grouped, not dropped.
        df.loc[unparsable, GROUP_COL] = df.loc[unparsable, SMILES_COL].astype(str)

    pce = pd.to_numeric(df[TARGET_COL], errors="coerce")
    ctrl = pd.to_numeric(df[CONTROL_COL], errors="coerce")
    _require_finite("dataset", {TARGET_COL: pce, CONTROL_COL: ctrl})

    df[DELTA_COL] = pce - ctrl
    return df


def _require_finite(name: str, columns: dict[str, pd.Series]) -> None:
    """Fail early with a readable message instead of letting NaN reach XGBoost.

    XGBoost accepts NaN silently and only fails much later with a cryptic
    message, so the boundary is checked here.
    """
    for col, series in columns.items():
        mask = series.isna()
        if mask.any():
            rows = series.index[mask].tolist()[:10]
            raise ValueError(
                f"{name}: column {col!r} has {int(mask.sum())} missing/non-numeric "
                f"value(s); first affected row index(es): {rows}")


def _require_finite_matrix(name: str, X: np.ndarray, feature_names: list[str]) -> None:
    """Same guard for the design matrix, including infinities."""
    mask = ~np.isfinite(X)
    if mask.any():
        cols = sorted({feature_names[j] for j in np.where(mask.any(axis=0))[0]})
        raise ValueError(f"{name}: non-finite feature values in {cols}")


def load_new_molecules(path: str | None = None) -> pd.DataFrame:
    """Read the screening table (no target column) and attach the group column."""
    df = pd.read_csv(path or NEW_MOL_CSV)
    df[GROUP_COL] = df[SMILES_COL].map(canonical_smiles)
    unparsable = df[GROUP_COL].isna()
    if unparsable.any():
        df.loc[unparsable, GROUP_COL] = df.loc[unparsable, SMILES_COL].astype(str)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All numeric columns except SMILES, the raw target, the group and the target."""
    excluded = {SMILES_COL, TARGET_COL, GROUP_COL, DELTA_COL}
    cols = [c for c in df.columns if c not in excluded]
    if not KEEP_CONTROL_AS_FEATURE:
        cols = [c for c in cols if c != CONTROL_COL]
    return cols


def build_matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (X, y_delta, control, groups, feature_names)."""
    feats = feature_columns(df)
    X = df[feats].to_numpy(dtype=float)
    _require_finite_matrix("dataset", X, feats)
    y = df[DELTA_COL].to_numpy(dtype=float)
    _require_finite("dataset", {DELTA_COL: pd.Series(y)})
    ctrl = df[CONTROL_COL].to_numpy(dtype=float)
    groups = df[GROUP_COL].to_numpy(dtype=object)
    return X, y, ctrl, groups, feats


def align_features(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Build the design matrix for new molecules in the exact training column order."""
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"screening table is missing feature columns: {missing}")
    X = df[feature_names].to_numpy(dtype=float)
    _require_finite_matrix("screening table", X, feature_names)
    return X


def split_train_test(
    n: int,
    split: str,
    groups: np.ndarray | None = None,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices according to the strategy.

    random : plain random split; replicates of one molecule may land on both sides.
    group  : GroupShuffleSplit; every replicate of a molecule stays on one side.
    """
    validate_split(split)
    idx = np.arange(n)
    if split == SPLIT_GROUP:
        if groups is None:
            raise ValueError("group splitting requires groups")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(idx, groups=groups))
    else:
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
    return np.sort(tr), np.sort(te)


def make_folds(
    split: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = N_SPLITS_INNER,
    seed: int = SEED,
) -> Folds:
    """Build cross-validation folds matching the splitting strategy.

    random -> KFold(shuffle=True), row level.
    group  -> GroupKFold, so replicates of one molecule share a fold. This is the
              counterpart of the group split and is what keeps the same molecule
              out of the training and validation sides simultaneously.
    """
    validate_split(split)
    if split == SPLIT_GROUP:
        if groups is None:
            raise ValueError("group cross-validation requires groups")
        cv = GroupKFold(n_splits=n_splits)
        return list(cv.split(X, y, groups))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(X, y))


def count_unique_molecules(groups: np.ndarray) -> int:
    return int(pd.Series(groups).nunique())


def describe_split(split: str, groups: np.ndarray | None = None) -> str:
    validate_split(split)
    if split == SPLIT_GROUP and groups is not None:
        return f"group split ({count_unique_molecules(groups)} unique molecules)"
    return "random split"


def assert_groups_disjoint(groups: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:
    """Raise if any group appears in both index sets (used by tests and sanity checks)."""
    overlap = set(np.asarray(groups)[a]) & set(np.asarray(groups)[b])
    if overlap:
        raise AssertionError(f"{len(overlap)} molecule(s) appear on both sides of the split")
