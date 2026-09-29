"""Shared pytest fixtures and sys.path setup for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    """Small dataset with deliberate replicate molecules.

    Molecules A/B/C/D each appear twice and molecule E once, so a group split
    must keep both replicates of A (and B, C, D) on the same side.
    """
    rng = np.random.default_rng(0)
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "O=P(O)(O)CCn1cccc1"]
    rows = []
    for i, s in enumerate(smiles):
        n_rep = 1 if s == "O=P(O)(O)CCn1cccc1" else 2
        for r in range(n_rep):
            x1 = rng.normal(i, 0.3)
            x2 = rng.normal(2 * i, 0.5)
            control = 18.0 + 0.5 * i
            rows.append({
                "SMILES": s,
                "f1": x1,
                "f2": x2,
                "control_PCE": control,
                "PCE": control + 0.8 * x1 - 0.3 * x2 + rng.normal(0, 0.05),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_matrices(synthetic_frame):
    from data_utils import build_matrices, load_dataset
    df = synthetic_frame.copy()
    df["canonical_smiles"] = df["SMILES"]
    df["delta_PCE"] = df["PCE"] - df["control_PCE"]
    return build_matrices(df)
