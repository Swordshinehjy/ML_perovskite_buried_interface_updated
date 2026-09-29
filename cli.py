"""Command line entry point for the buried-interface ML pipeline.

Usage
-----
    conda activate rdkit

    # run the whole pipeline end to end (tune -> train -> predict -> shap
    # -> y-scrambling -> compare)
    python cli.py all
    python cli.py all --trials 150 --inner-trials 60 --n-scrambles 200

    # or run a single stage
    python cli.py tune
    python cli.py train
    python cli.py predict
    python cli.py shap
    python cli.py scramble
    python cli.py compare

    # restrict to selected models and override a few knobs
    python cli.py train --models random_cv group_cv
    python cli.py tune  --models group_nested --trials 200 --seed 7

Stages and their artefacts
--------------------------
    tune     outputs/<model>/best_params.json (+ nested_folds.json, tuning figures)
    train    outputs/<model>/metrics.json, model.json + model_bundle.json, figures
    predict  outputs/<model>/new_molecule_predictions.csv, screening figures
    shap     outputs/<model>/shap/*, SHAP figures, cross-model consensus ranking
    scramble outputs/<model>/y_scrambling.csv, permutation figures
    compare  outputs/_comparison/* (metrics tables, report, comparison figures)

Models are `random_cv`, `random_nested`, `group_cv`, `group_nested`
(split strategy x cross-validation protocol). Everything is reproducible: the
seed defaults to config.SEED and every stage writes a log to outputs/.

Examples
--------
    python cli.py all --models group_cv --n-scrambles 50   # quick smoke run
    python cli.py all --trials 300                         # more thorough search
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import config as cfg

STAGES = ("tune", "train", "predict", "shap", "scramble", "compare")


def _run(stage: str, args: argparse.Namespace) -> int:
    if stage == "tune":
        from hyperparam_tuning import main
        return main(["--models", *args.models] + _flag("--trials", args.trials)
                    + _flag("--inner-trials", args.inner_trials) + _flag("--seed", args.seed))
    if stage == "train":
        from train import main
        return main(["--models", *args.models] + _flag("--seed", args.seed))
    if stage == "predict":
        from predict_new import main
        return main(["--models", *args.models] + _flag("--input", args.input))
    if stage == "shap":
        from shap_analysis import main
        return main(["--models", *args.models] + _flag("--top-n", args.top_n))
    if stage == "scramble":
        from y_scrambling import main
        return main(["--models", *args.models] + _flag("--n-scrambles", args.n_scrambles)
                    + _flag("--seed", args.seed))
    if stage == "compare":
        from compare_models import main
        return main(["--models", *args.models])
    raise ValueError(f"unknown stage {stage!r}")


def _flag(name: str, value) -> list[str]:
    return [] if value is None else [name, str(value)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="Buried-interface XGBoost pipeline: tune, train, predict, explain, validate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `python cli.py all` for the complete workflow.",
    )
    p.add_argument("stage", choices=[*STAGES, "all"], help="pipeline stage to run")
    p.add_argument("--models", nargs="*", default=[], choices=cfg.MODEL_KEYS,
                   help="restrict to these models (default: all four)")
    p.add_argument("--trials", type=int, default=cfg.N_TRIALS, help="Optuna trials per search")
    p.add_argument("--inner-trials", type=int, default=cfg.N_TRIALS_INNER,
                   help="Optuna trials inside each nested-CV outer fold")
    p.add_argument("--n-scrambles", type=int, default=cfg.N_SCRAMBLES,
                   help="number of y-scrambling permutations")
    p.add_argument("--top-n", type=int, default=cfg.TOP_N_SHAP,
                   help="number of top SHAP features to report")
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--input", default=None, help="alternative screening table for `predict`")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.models:
        args.models = list(cfg.MODEL_KEYS)
    stages = STAGES if args.stage == "all" else (args.stage,)

    print(f"models: {', '.join(args.models)}")
    for stage in stages:
        print(f"\n[stage] {stage}")
        code = _run(stage, args)
        if code:
            print(f"[stage] {stage} finished with code {code}", file=sys.stderr)
            return code
    print("\nall stages finished; results in", cfg.OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
