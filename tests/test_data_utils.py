"""Tests for grouping, splitting and fold construction.

BUG-02 (reviewed): `split_train_test` and `make_folds` branched on
`split == "group"` and otherwise did a random split, so a typo such as "grp"
silently produced a *random* split. A study that believed it was grouped would
then leak replicates across the split with no warning at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import SPLIT_GROUP, SPLIT_RANDOM
from data_utils import (
    assert_groups_disjoint,
    build_matrices,
    canonical_smiles,
    feature_columns,
    load_dataset,
    make_folds,
    split_train_test,
)


def test_canonical_smiles_is_invariant_to_input_writing():
    assert canonical_smiles("C(C)(=O)O") == canonical_smiles("CC(=O)O")
    assert canonical_smiles("not-a-molecule") is None


def test_unknown_split_strategy_raises():
    y = np.zeros(10)
    X = np.zeros((10, 2))
    groups = np.arange(10)
    with pytest.raises(ValueError):
        split_train_test(10, "grp", groups)
    with pytest.raises(ValueError):
        make_folds("grp", X, y, groups)


def test_group_split_requires_groups():
    with pytest.raises(ValueError):
        split_train_test(10, SPLIT_GROUP, None)
    with pytest.raises(ValueError):
        make_folds(SPLIT_GROUP, np.zeros((10, 2)), np.zeros(10), None)


def test_group_split_keeps_replicates_on_one_side(synthetic_matrices):
    X, y, ctrl, groups, feats = synthetic_matrices
    tr, te = split_train_test(len(y), SPLIT_GROUP, groups, test_size=0.4, seed=0)
    assert_groups_disjoint(groups, tr, te)


def test_random_split_does_leak_replicates(synthetic_matrices):
    """Documents why grouping exists: the random split puts twins on both sides."""
    X, y, ctrl, groups, feats = synthetic_matrices
    leaked = False
    for seed in range(20):
        tr, te = split_train_test(len(y), SPLIT_RANDOM, groups, test_size=0.4, seed=seed)
        if set(groups[tr]) & set(groups[te]):
            leaked = True
            break
    assert leaked, "expected the random split to separate replicates at least sometimes"


def test_group_kfold_keeps_molecules_in_one_fold(synthetic_matrices):
    X, y, ctrl, groups, feats = synthetic_matrices
    for tr, te in make_folds(SPLIT_GROUP, X, y, groups, n_splits=3, seed=0):
        assert_groups_disjoint(groups, tr, te)


def test_folds_cover_every_row_exactly_once(synthetic_matrices):
    X, y, ctrl, groups, feats = synthetic_matrices
    for split in (SPLIT_RANDOM, SPLIT_GROUP):
        seen = []
        for tr, te in make_folds(split, X, y, groups, n_splits=3, seed=0):
            seen.append(te)
        pooled = np.sort(np.concatenate(seen))
        assert np.array_equal(pooled, np.arange(len(y)))


def test_feature_columns_exclude_identifiers_and_targets():
    df = pd.DataFrame({"SMILES": ["CCO"], "PCE": [1.0], "control_PCE": [0.5],
                       "delta_PCE": [0.5], "canonical_smiles": ["CCO"], "f1": [1.0]})
    assert feature_columns(df) == ["control_PCE", "f1"]


def test_build_matrices_shapes_and_delta(synthetic_frame):
    df = synthetic_frame.copy()
    df["canonical_smiles"] = df["SMILES"]
    df["delta_PCE"] = df["PCE"] - df["control_PCE"]
    X, y, ctrl, groups, feats = build_matrices(df)
    assert X.shape[0] == len(df)
    assert X.shape[1] == len(feats)
    assert np.allclose(y, df["PCE"].to_numpy() - df["control_PCE"].to_numpy())


def test_load_dataset_on_real_file():
    df = load_dataset()
    assert "canonical_smiles" in df.columns and "delta_PCE" in df.columns
    assert df["canonical_smiles"].notna().all()
