"""Data loading, cleaning, the temporal dev/test split and grouped folds.

Nothing in this module is *fitted*, so calling it from every notebook cannot leak information
across the train/test boundary. Every notebook calls :func:`prepare_dataset` and gets the
same rows, in the same order, with the same split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import CFG
from .features import CAT_COLS, add_calendar, add_history, add_macro, add_terms, feature_groups

EXPECTED_COLS = ['ID', 'customer_id', 'country_id', 'tbl_loan_id', 'Total_Amount', 'Total_Amount_to_Repay',
                 'loan_type', 'disbursement_date', 'duration', 'lender_id', 'New_versus_Repeat',
                 'Amount_Funded_By_Lender', 'Lender_portion_Funded', 'due_date', 'Lender_portion_to_be_repaid',
                 'target']


# ----------------------------------------------------------------------------- raw files
def read_table(path) -> pd.DataFrame:
    """Read a Zindi CSV; rename ``loan_type_id`` -> ``loan_type`` and parse the two date columns."""
    df = pd.read_csv(path).rename(columns={'loan_type_id': 'loan_type'})
    for c in ['disbursement_date', 'due_date']:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df


def load_raw(cfg: dict = CFG):
    data = Path(cfg['DATA_DIR'])
    p_train, p_test = data / cfg['TRAIN_FILE'], data / cfg['TEST_FILE']
    for p in (p_train, p_test):
        if not p.exists():
            raise FileNotFoundError(f'Missing {p}. Download the Zindi files into {data} (see README).')
    train_raw, test_raw = read_table(p_train), read_table(p_test)
    if cfg.get('DEBUG_ROWS'):
        train_raw = train_raw.sample(cfg['DEBUG_ROWS'], random_state=cfg['SEED'])
    missing = [c for c in EXPECTED_COLS if c not in train_raw.columns]
    if missing:
        raise ValueError(f'Train.csv lacks columns from the data dictionary: {missing}')
    return train_raw, test_raw


def clean_labelled(train_raw: pd.DataFrame) -> pd.DataFrame:
    """Exact duplicates removed; rows missing key fields dropped; categoricals as strings; sorted by date."""
    lab = train_raw.drop_duplicates().copy()
    lab = lab.dropna(subset=['customer_id', 'disbursement_date', 'due_date', 'target', 'Total_Amount'])
    lab['target'] = lab['target'].astype(int)
    for c in CAT_COLS:
        lab[c] = lab[c].fillna('missing').astype(str)
    return lab.sort_values('disbursement_date', kind='mergesort').reset_index(drop=True)


def clean_unlabelled(test_raw: pd.DataFrame) -> pd.DataFrame:
    tst = test_raw.dropna(subset=['customer_id', 'disbursement_date']).copy()
    implied_due = tst['disbursement_date'] + pd.to_timedelta(tst['duration'], unit='D')
    tst['due_date'] = tst['due_date'].fillna(implied_due) if 'due_date' in tst else implied_due
    for c in CAT_COLS:
        tst[c] = tst[c].fillna('missing').astype(str)
    return tst.reset_index(drop=True)


# ----------------------------------------------------------------------------- macro indicators
def prepare_econ(econ: pd.DataFrame) -> pd.DataFrame | None:
    """Turn the macro table into one row per (country, year) with ``macro_*`` columns.

    Handles both layouts:
      * **wide** (the file shipped with the challenge): ``Country, Indicator, YR2001 ... YR2023``
      * **long**: ``country, year, <indicator columns>``
    """
    cols = {c.lower(): c for c in econ.columns}
    ccol = next((cols[c] for c in cols if 'country' in c), None)
    if ccol is None:
        return None
    year_cols = [c for c in econ.columns if str(c).upper().startswith('YR') and str(c)[2:].isdigit()]
    icol = next((cols[c] for c in cols if 'indicator' in c), None)
    if year_cols and icol is not None:                                     # wide layout -> long
        long = econ.melt(id_vars=[ccol, icol], value_vars=year_cols, var_name='_yr', value_name='_v')
        long['_year'] = long['_yr'].str[2:].astype(int)
        e = long.pivot_table(index=[ccol, '_year'], columns=icol, values='_v', aggfunc='first').reset_index()
        e.columns.name = None
    else:                                                                  # long layout
        ycol = next((cols[c] for c in cols if 'year' in c), None)
        if ycol is None:
            return None
        e = econ.rename(columns={ycol: '_year'}).copy()
        e['_year'] = pd.to_numeric(e['_year'], errors='coerce')
    e['_cty'] = e[ccol].astype(str).str.lower().str.strip()
    num = [c for c in e.columns if c not in (ccol, '_year', '_cty') and pd.api.types.is_numeric_dtype(e[c])]
    e = e[['_cty', '_year'] + num].dropna(subset=['_year']).drop_duplicates(['_cty', '_year'])
    e['_year'] = e['_year'].astype(int)

    def slug(s):
        s = str(s).lower()
        for ch in ' ,()%$/-':
            s = s.replace(ch, '_')
        return 'macro_' + '_'.join(p for p in s.split('_') if p)
    return e.rename(columns={c: slug(c) for c in num})


def load_econ(cfg: dict = CFG) -> pd.DataFrame | None:
    p = Path(cfg['DATA_DIR']) / cfg['ECON_FILE']
    if not p.exists():
        print(f'{p.name} not found: macro features skipped')
        return None
    e = prepare_econ(pd.read_csv(p))
    if e is None:
        print(f'{p.name} found but no country/year structure detected: macro features skipped')
    return e


def load_findex(cfg: dict = CFG) -> pd.DataFrame | None:
    """Global Findex Ghana 2021 microdata (optional). Returns None with a message when absent."""
    name = cfg.get('FINDEX_FILE')
    p = Path(cfg['DATA_DIR']) / name if name else None
    if p is None or not p.exists():
        print('Findex Ghana file not found (optional). Download it from the World Bank Microdata Library '
              '(catalog 4646) into data/ as micro_gha.csv or set CFG["FINDEX_FILE"].')
        return None
    return pd.read_stata(p) if p.suffix.lower() == '.dta' else pd.read_csv(p)


# ----------------------------------------------------------------------------- features + split
def make_features(df: pd.DataFrame, labelled: pd.DataFrame, econ, cfg: dict = CFG) -> pd.DataFrame:
    """One definition of the features for train, test and the scoring service (no train/serve skew)."""
    out = add_calendar(df)
    out = add_terms(out)
    out = add_history(out, labelled, cfg['GRACE_DAYS'])
    return add_macro(out, econ, lag_years=cfg['MACRO_LAG_YEARS'])


@dataclass
class Dataset:
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    lab: pd.DataFrame            # cleaned labelled Kenyan loans
    feat: pd.DataFrame           # lab + engineered features
    dev: pd.DataFrame            # older loans (model selection, calibration, threshold)
    test: pd.DataFrame           # most recent loans (one-shot evaluation)
    test_start: pd.Timestamp
    econ: pd.DataFrame | None
    groups: dict = field(default_factory=dict)   # feature groups
    all_clf: list = field(default_factory=list)  # Task 2 features
    shared: list = field(default_factory=list)   # Task 1-safe features (no loan terms)

    @property
    def y_dev(self):
        return self.dev['target']

    @property
    def g_dev(self):
        return self.dev['customer_id']


def temporal_split(feat: pd.DataFrame, test_frac: float):
    """Test = most recent ``test_frac`` of loans; dev loans whose due_date reaches into test are purged."""
    test_start = feat['disbursement_date'].quantile(1 - test_frac)
    is_test = feat['disbursement_date'] >= test_start
    is_dev = (feat['disbursement_date'] < test_start) & (feat['due_date'] < test_start)
    dev, test = feat[is_dev].reset_index(drop=True), feat[is_test].reset_index(drop=True)
    assert dev['due_date'].max() < test_start and dev['disbursement_date'].max() < test_start
    return dev, test, test_start, int((~is_dev & ~is_test).sum())


def prepare_dataset(cfg: dict = CFG, verbose: bool = True) -> Dataset:
    train_raw, test_raw = load_raw(cfg)
    lab = clean_labelled(train_raw)
    econ = load_econ(cfg)
    feat = make_features(lab, lab, econ, cfg)
    groups = feature_groups(feat)
    dev, test, test_start, n_purged = temporal_split(feat, cfg['TEST_FRAC'])
    if test['target'].sum() < 30:
        raise ValueError('Too few test defaults for stable estimates; increase data or TEST_FRAC')
    model_groups = ['base', 'calendar', 'terms', 'history'] + (['macro'] if cfg.get('USE_MACRO_FEATURES') else [])
    ds = Dataset(train_raw, test_raw, lab, feat, dev, test, test_start, econ, groups,
                 all_clf=[c for g in model_groups for c in groups[g]],
                 shared=[c for g in model_groups if g != 'terms' for c in groups[g]])
    if verbose:
        print(f'Train {train_raw.shape} -> cleaned {len(lab):,} loans (default rate {lab.target.mean():.3%})')
        print(f'Test starts {test_start.date()} | dev {len(dev):,} (defaults {dev.target.sum()}, {dev.target.mean():.2%}) '
              f'| test {len(test):,} (defaults {test.target.sum()}, {test.target.mean():.2%}) | purged {n_purged:,}')
        print('Feature groups:', {k: len(v) for k, v in groups.items()},
              f"| model features: Task 2 = {len(ds.all_clf)}, Task 1 = {len(ds.shared)} (macro as model input: {bool(cfg.get('USE_MACRO_FEATURES'))})")
    return ds


def make_folds(X, y, groups, seed: int, n: int):
    """StratifiedGroupKFold by customer; asserts no customer crosses train/validation."""
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    folds = list(sgkf.split(X, y, groups))
    for tr_i, va_i in folds:
        assert set(groups.iloc[tr_i]).isdisjoint(groups.iloc[va_i]), 'customer leakage across folds'
        assert y.iloc[va_i].sum() > 0, 'fold without positives'
    return folds


def temporal_folds(dev: pd.DataFrame, y: pd.Series, edges_q=(0.5, 0.625, 0.75, 0.875, 1.0)):
    """Forward-chaining folds with purge: training loans must be resolved before the validation block starts."""
    edges = dev['disbursement_date'].quantile(list(edges_q)).values
    folds = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        tri = np.where((dev.disbursement_date < lo) & (dev.due_date < lo))[0]
        last = i == len(edges) - 2
        vai = np.where((dev.disbursement_date >= lo) & ((dev.disbursement_date <= hi) if last else (dev.disbursement_date < hi)))[0]
        if y.iloc[vai].sum() > 0 and y.iloc[tri].sum() > 0:
            folds.append((tri, vai))
    return folds


def size_band(amounts, edges) -> pd.Series:
    return pd.Series(pd.cut(amounts, [-np.inf, *edges, np.inf],
                            labels=['Q1 smallest', 'Q2', 'Q3', 'Q4 largest']).astype(str), index=getattr(amounts, 'index', None))
