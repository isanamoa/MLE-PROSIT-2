"""Time-safe feature engineering, shared by training notebooks and the scoring service.

Every feature uses only information available at the decision moment (``disbursement_date``):
  * calendar  - from the disbursement date and the contractual due date
  * terms     - from the loan contract (amount, price, lender share)          [Task 2 only]
  * history   - backward ``merge_asof`` with ``allow_exact_matches=False``; past *outcomes*
                count only once the earlier loan's ``due_date`` (+ grace) has passed
  * macro     - country indicators joined on (country, year - lag)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAT_COLS = ['loan_type', 'lender_id', 'New_versus_Repeat']

# Model inputs for the history group. Counts, gaps and amounts are heavy-tailed (a few borrowers have hundreds of loans or
# very large past loans), so they enter models as log1p; the raw columns stay in the frame for checks and reporting.
HIST_COLS = ['log_n_prior_loans', 'no_prior_loans', 'log_days_since_last_loan', 'log_prior_mean_amount', 'log_last_amount',
             'log_n_resolved_prior', 'log_n_prior_defaults', 'prior_default_rate']
_LOG_HIST = ['n_prior_loans', 'days_since_last_loan', 'prior_mean_amount', 'last_amount', 'n_resolved_prior', 'n_prior_defaults']


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Pay-cycle and weekday effects (salary at month end, weekend due dates)."""
    out = df.copy()
    d = out['disbursement_date']
    out['disb_month'] = d.dt.month
    out['disb_quarter'] = d.dt.quarter
    out['disb_dow'] = d.dt.dayofweek
    out['disb_dom'] = d.dt.day
    out['disb_month_end'] = (d.dt.day >= 25).astype(int)
    due = out['due_date'] if 'due_date' in out else d + pd.to_timedelta(out['duration'], unit='D')
    out['due_dow'] = due.dt.dayofweek
    out['due_weekend'] = (due.dt.dayofweek >= 5).astype(int)
    return out


def add_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Loan-term features, all known at disbursement.

    fee_rate            = (Total_Amount_to_Repay - Total_Amount) / Total_Amount   (price of credit)
    daily_fee_rate      = fee_rate / duration                                      (price per day)
    lender_funded_share = Amount_Funded_By_Lender / Total_Amount
    lender_repay_share  = Lender_portion_to_be_repaid / Total_Amount_to_Repay
    """
    out = df.copy()
    amt = out['Total_Amount'].astype(float)
    rep = out['Total_Amount_to_Repay'].astype(float)
    out['log_amount'] = np.log1p(amt)
    out['fee_rate'] = (rep - amt) / amt.replace(0, np.nan)
    out['daily_fee_rate'] = out['fee_rate'] / out['duration'].clip(lower=1)
    out['lender_funded_share'] = out['Amount_Funded_By_Lender'] / amt.replace(0, np.nan)
    out['lender_repay_share'] = out['Lender_portion_to_be_repaid'] / rep.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def add_history(df: pd.DataFrame, labelled: pd.DataFrame, grace_days: int = 0) -> pd.DataFrame:
    """Borrower history that the lender would actually know at ``disbursement_date``.

    (a) activity: loans disbursed strictly BEFORE the current one (needs no labels)
    (b) outcomes: loans whose due_date (+grace) is strictly BEFORE the current disbursement.
        A prior loan that is still running is NOT counted -> no label leakage from the future.

    Results are written back **by position** (``.values``) so they always land on the right row,
    whatever the index of ``df``.
    """
    base = df[['customer_id', 'disbursement_date']].copy()
    base['_row'] = np.arange(len(base))
    base = base.sort_values('disbursement_date', kind='mergesort')

    a = labelled[['customer_id', 'disbursement_date', 'Total_Amount']].sort_values('disbursement_date', kind='mergesort').copy()
    a['n_prior_loans'] = a.groupby('customer_id').cumcount() + 1          # loans up to and including this one
    a['prior_mean_amount'] = a.groupby('customer_id')['Total_Amount'].cumsum() / a['n_prior_loans']
    a['last_loan_date'] = a['disbursement_date']
    a['last_amount'] = a['Total_Amount']
    a = a.rename(columns={'disbursement_date': '_a_date'})[
        ['customer_id', '_a_date', 'n_prior_loans', 'prior_mean_amount', 'last_loan_date', 'last_amount']]
    m = pd.merge_asof(base, a, left_on='disbursement_date', right_on='_a_date', by='customer_id',
                      direction='backward', allow_exact_matches=False)

    b = labelled[['customer_id', 'due_date', 'target']].dropna().copy()
    b['_b_date'] = b['due_date'] + pd.to_timedelta(grace_days, unit='D')
    b = b.sort_values('_b_date', kind='mergesort')
    b['n_resolved_prior'] = b.groupby('customer_id').cumcount() + 1
    b['n_prior_defaults'] = b.groupby('customer_id')['target'].cumsum()
    m = pd.merge_asof(m.sort_values('disbursement_date', kind='mergesort'),
                      b[['customer_id', '_b_date', 'n_resolved_prior', 'n_prior_defaults']],
                      left_on='disbursement_date', right_on='_b_date', by='customer_id',
                      direction='backward', allow_exact_matches=False)
    m = m.sort_values('_row')

    out = df.copy()
    for c in ['n_prior_loans', 'n_resolved_prior', 'n_prior_defaults']:
        out[c] = m[c].fillna(0).values
    out['no_prior_loans'] = (out['n_prior_loans'] == 0).astype(int)
    out['days_since_last_loan'] = (m['disbursement_date'] - m['last_loan_date']).dt.days.values
    out['prior_mean_amount'] = m['prior_mean_amount'].values
    out['last_amount'] = m['last_amount'].values
    out['prior_default_rate'] = np.where(out['n_resolved_prior'] > 0,
                                         out['n_prior_defaults'] / out['n_resolved_prior'].clip(lower=1), 0.0)
    for c in _LOG_HIST:
        out['log_' + c] = np.log1p(out[c].astype(float).clip(lower=0))
    if 'Total_Amount' in out:   # relative loan size vs the borrower's own past (terms feature, Task 2 only)
        out['amount_vs_prior_mean'] = (out['Total_Amount'] / out['prior_mean_amount']).replace([np.inf, -np.inf], np.nan)
    return out


def add_macro(df: pd.DataFrame, econ_prepared: pd.DataFrame | None, lag_years: int = 1) -> pd.DataFrame:
    """Join macro indicators by (country, year - lag). Annual figures for the current year are only
    published after the year ends, so the lag prevents look-ahead."""
    if econ_prepared is None:
        return df
    out = df.copy()
    idx = out.index
    out['_cty'] = out['country_id'].astype(str).str.lower().str.strip()
    out['_year'] = out['disbursement_date'].dt.year - lag_years
    out = out.merge(econ_prepared, on=['_cty', '_year'], how='left').drop(columns=['_cty', '_year'])
    out.index = idx
    return out


def feature_groups(feat: pd.DataFrame) -> dict:
    """Named feature groups (used for ablations and fairness exclusions). Macro columns are kept only if
    they are mostly non-missing and actually vary in the training data."""
    macro = [c for c in feat.columns if c.startswith('macro_')]
    macro = [c for c in macro if feat[c].notna().mean() > 0.5 and feat[c].nunique() > 1]
    groups = {
        'base': ['duration', 'loan_type', 'lender_id', 'New_versus_Repeat'],
        'calendar': ['disb_month', 'disb_quarter', 'disb_dow', 'disb_dom', 'disb_month_end', 'due_dow', 'due_weekend'],
        'terms': ['log_amount', 'fee_rate', 'daily_fee_rate', 'lender_funded_share', 'lender_repay_share',
                  'Lender_portion_Funded', 'amount_vs_prior_mean'],
        'history': list(HIST_COLS),
        'macro': macro,
    }
    return {k: [c for c in v if c in feat.columns] for k, v in groups.items()}


def brute_force_history_check(feat: pd.DataFrame, lab: pd.DataFrame, n: int = 300, seed: int = 42,
                              grace_days: int = 0) -> int:
    """Recompute history for a sample of loans by brute force; returns the number of mismatches (must be 0)."""
    chk = feat.sample(min(n, len(feat)), random_state=seed)
    bad = 0
    for _, r in chk.iterrows():
        prev = lab[(lab.customer_id == r.customer_id) & (lab.disbursement_date < r.disbursement_date)]
        res = prev[prev.due_date + pd.to_timedelta(grace_days, 'D') < r.disbursement_date]
        if (int(r.n_prior_loans) != len(prev) or int(r.n_resolved_prior) != len(res)
                or int(r.n_prior_defaults) != int(res.target.sum())):
            bad += 1
    return bad
