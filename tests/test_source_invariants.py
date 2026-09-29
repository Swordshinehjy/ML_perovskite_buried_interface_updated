"""Static checks over the source tree.

BUG-01 (reviewed): train.py built its result with a duplicated dict key::

    result = {
        "cv": spec.cv,        # protocol name
        ...
        "cv": cv_metrics,     # silently overwrote the protocol name
    }

The protocol therefore disappeared from metrics.json, the training summary
printed a whole metrics dict into the "cv" column and compare_models mislabelled
every protocol. A dict literal with repeated constant keys is always a mistake,
so the whole source tree is scanned for it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


def _modules():
    return sorted(SRC.glob("*.py"))


def _duplicate_keys(node: ast.Dict) -> list[str]:
    seen, dupes = set(), []
    for k in node.keys:
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            if k.value in seen:
                dupes.append(k.value)
            seen.add(k.value)
    return dupes


def test_no_duplicate_keys_in_dict_literals():
    offenders = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in _duplicate_keys(node):
                    offenders.append(f"{path.name}:{node.lineno}: duplicate key {key!r}")
    assert not offenders, "duplicated dict keys found:\n" + "\n".join(offenders)


def test_no_decorative_separator_comment_lines():
    """Comment separators such as `# =====` or `# -----` are not allowed."""
    offenders = []
    for path in _modules():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            body = stripped.lstrip("#").strip()
            if len(body) >= 4 and len(set(body)) == 1 and body[0] in "=-":
                offenders.append(f"{path.name}:{i}: {stripped}")
    assert not offenders, "decorative separator comments found:\n" + "\n".join(offenders)


def test_every_module_is_importable():
    import importlib

    for path in _modules():
        importlib.import_module(path.stem)


@pytest.mark.parametrize("name", ["config", "plotting", "train", "compare_models"])
def test_module_has_english_docstring(name):
    import importlib

    doc = importlib.import_module(name).__doc__ or ""
    assert doc.strip(), f"{name} has no module docstring"


def test_cli_exposes_every_stage():
    import cli

    assert set(cli.STAGES) == {"tune", "train", "predict", "shap", "scramble", "compare"}
