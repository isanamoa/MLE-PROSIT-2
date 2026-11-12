"""Single source of configuration for every notebook and the scoring service.

Edit values here (or override them at the top of a notebook with ``CFG.update(...)``).
Paths are resolved relative to the repository root, so the project runs the same way
locally, in Colab (after ``git clone``) or on a teammate's machine.
"""
from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # repo root (src/prosit2/config.py -> ../..)

CFG = dict(
    # ---- paths (relative to repo root unless absolute; env vars override for Colab/Drive) ----
    DATA_DIR=os.environ.get('PROSIT2_DATA_DIR', str(ROOT / 'data')),
    OUT_DIR=os.environ.get('PROSIT2_OUT_DIR', str(ROOT / 'outputs')),
    TRAIN_FILE='Train.csv',
    TEST_FILE='Test.csv',
    ECON_FILE='economic_indicators.csv',
    FINDEX_FILE='micro_gha.csv',  # Global Findex 2021 Ghana microdata (World Bank Microdata Library, catalog 4646)
    # ---- reproducibility / design ----
    SEED=42,
    N_FOLDS=5,
    TEST_FRAC=0.20,          # most recent 20% of loans = untouched temporal test
    GRACE_DAYS=0,            # days after due_date before an outcome counts as known (history features)
    MIN_CAT_FREQ=100,        # rarer categories are pooled into an 'infrequent' bucket
    SAMPLING_RATIO=0.2,      # minority:majority after resampling (1:1 is not assumed to be better)
    TOP_Q=0.10,              # alert budget used for strategy-comparable metrics
    MACRO_LAG_YEARS=1,       # macro indicators joined on (country, year - lag) to avoid look-ahead
    USE_MACRO_FEATURES=False,  # macro is joined and analysed, but NOT a model input: Train covers ~2 macro years,
                               # so the columns act as a year flag and extrapolate on later loans (see 02, E5)
    # ---- compute budget ----
    FAST=os.environ.get('PROSIT2_FAST', '0') == '1',  # True = tiny settings for a smoke test (minutes, not hours)
    DEBUG_ROWS=None,         # e.g. 15000 to subsample Train for a rehearsal
    N_ITER_TUNE=15,
    MAX_SVC_ROWS=15000,
    BOOT_B=1000,
    # ---- cost matrix ASSUMPTIONS (justify for the Ghanaian market in the report) ----
    CURRENCY='KES',          # currency of the amounts in Train/Test.csv (Kenyan loans)
    FX_TO_GHS=None,          # set e.g. 0.0x (GHS per KES, with source and date) to report in GHS
    RECOVERY_RATE=0.20,      # share of principal recovered after default (placeholder)
    COLLECTION_COST_RATE=0.08,  # collection/servicing cost as share of principal (placeholder)
    MIN_APPROVAL_RATE=0.80,  # business constraint: approve at least this share of applicants
    NOISE_FLOOR=0.0263,      # ProSit leaderboard noise floor for F1
    FINAL_MODEL_OVERRIDE=None,  # e.g. 'lgbm' to force a model; None = rule-based (1-SE parsimony)
    REVIEW_FRACTION=0.5,     # PD in [REVIEW_FRACTION*thr, thr) -> approve with reduced limit + manual review
)


def apply_fast_mode(cfg: dict = CFG) -> dict:
    """Shrink compute settings when FAST=True (idempotent). FAST runs write to ``outputs_fast/`` so they
    can never be mistaken for, or resumed into, the full run in ``outputs/``."""
    if cfg['FAST']:
        cfg.update(N_ITER_TUNE=3, MAX_SVC_ROWS=3000, BOOT_B=200, N_FOLDS=3)
        out = str(cfg['OUT_DIR']).rstrip('/\\')
        if not out.endswith('_fast'):
            cfg['OUT_DIR'] = out + '_fast'
    return cfg


def F(full, fast, cfg: dict = CFG):
    """Pick a compute-sized setting: ``fast`` when FAST mode is on, else ``full``."""
    return fast if cfg['FAST'] else full


def money(cfg: dict = CFG):
    """Return (multiplier, label) for reporting amounts: GHS when an FX rate is set, else source currency."""
    fx = cfg.get('FX_TO_GHS')
    return (float(fx), 'GHS') if fx else (1.0, cfg['CURRENCY'])


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


_LAYOUT = {
    'ckpt_base': 'checkpoints/baseline_models', 'ckpt_imb': 'checkpoints/imbalance_experiments',
    'ckpt_opt': 'checkpoints/optimised_models', 'ckpt_best': 'checkpoints/best_models',
    'ckpt_abl': 'checkpoints/ablation_models', 'ckpt_reg': 'checkpoints/regression_models',
    'artifacts': 'artifacts',                      # hand-off files between notebooks
    'results': 'results', 'hpo': 'results/hyperparameter_results',
    'plots_eda': 'plots/eda', 'plots_cmp': 'plots/model_comparisons', 'plots_eval': 'plots/evaluation',
    'plots_fair': 'plots/fairness', 'plots_expl': 'plots/explainability', 'plots_tr': 'plots/transfer',
    'reports': 'reports',
}


def output_dirs(cfg: dict = CFG) -> dict:
    out = Path(cfg['OUT_DIR'])
    dirs = {k: out / v for k, v in _LAYOUT.items()}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def environment_record(cfg: dict = CFG) -> dict:
    """Library versions + config, written to outputs/reports/environment.json for the reproducibility statement."""
    import importlib
    env = dict(python=platform.python_version(), platform=platform.platform(), seed=cfg['SEED'])
    for lib in ['numpy', 'pandas', 'scipy', 'sklearn', 'imblearn', 'lightgbm', 'xgboost', 'shap', 'fairlearn']:
        try:
            env[lib] = importlib.import_module(lib).__version__
        except Exception:  # optional library missing
            env[lib] = None
    env['config'] = cfg
    return env


def write_environment(dirs: dict, cfg: dict = CFG, name: str = 'environment.json') -> dict:
    env = environment_record(cfg)
    (dirs['reports'] / name).write_text(json.dumps(env, indent=2, default=str))
    return env


def savefig(fig, dirs: dict, key: str, name: str, show: bool = True):
    import matplotlib.pyplot as plt
    fig.savefig(dirs[key] / f'{name}.png', dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
