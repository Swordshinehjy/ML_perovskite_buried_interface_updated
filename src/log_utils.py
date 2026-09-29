"""Logging helpers: one logger per script, writing to file and console.

Single responsibility: assemble logging handlers, no business logic.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _force_utf8_console() -> None:
    """Windows consoles may default to GBK, which mangles non-ASCII log text."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def setup_logger(log_path: Path, level: int = logging.INFO, to_console: bool = True) -> logging.Logger:
    """Configure and return the root logger (file + optional console).

    Existing handlers are removed first so repeated runs do not duplicate output.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _force_utf8_console()
    logger = logging.getLogger()
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(_FMT)
    logger.addHandler(fh)

    if to_console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(_FMT)
        logger.addHandler(sh)

    for name in ("optuna", "matplotlib", "xgboost", "lightgbm", "shap"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logger


def install_excepthook(logger: logging.Logger) -> None:
    """Route uncaught exceptions into the log as well."""
    def _hook(exc_type, exc_value, exc_tb):
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = _hook
