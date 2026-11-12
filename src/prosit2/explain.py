"""Per-applicant contributions and plain-language reason codes (shared by notebooks and the scoring service)."""
from __future__ import annotations

import numpy as np
import pandas as pd

REASONS = {
    'duration': 'The repayment period you requested is short or long relative to similar loans.',
    'loan_type': 'The type of loan you applied for has had higher repayment difficulty.',
    'lender_id': 'Loans from this lender have had higher repayment difficulty.',
    'New_versus_Repeat': 'You are a first-time borrower, so we have little repayment history to rely on.',
    'n_prior_loans': 'The number of previous loans on your record.',
    'no_prior_loans': 'We have no earlier loan record for you.',
    'days_since_last_loan': 'The time since your last loan.',
    'prior_mean_amount': 'The size of your earlier loans compared with this one.',
    'last_amount': 'The size of your most recent loan compared with this one.',
    'n_resolved_prior': 'The number of earlier loans that have already been repaid or closed.',
    'n_prior_defaults': 'A previous loan was not repaid on time.',
    'prior_default_rate': 'A previous loan was not repaid on time.',
    'log_amount': 'The amount requested is high for this type of loan.',
    'fee_rate': 'The total cost of this loan is high relative to the amount borrowed.',
    'daily_fee_rate': 'The daily cost of this loan is high because the term is short.',
    'amount_vs_prior_mean': 'This loan is much larger than your earlier loans.',
    'lender_funded_share': 'The share of the loan funded by the lender.',
    'lender_repay_share': 'The share of the repayment owed to the lender.',
    'Lender_portion_Funded': 'The share of the loan funded by the lender.',
}
TIMING_TEXT = 'The timing of the loan (day or month) is associated with higher repayment difficulty.'
MACRO_TEXT = 'Current economic conditions (such as inflation or interest rates) increase repayment risk.'


def reason_text(col: str, texts: dict = REASONS) -> str:
    if col in texts:
        return texts[col]
    if col.startswith('log_') and col[4:] in texts:
        return texts[col[4:]]
    if col.startswith('disb') or col.startswith('due'):
        return TIMING_TEXT
    if col.startswith('macro_'):
        return MACRO_TEXT
    return 'Your application pattern (' + col.replace('_', ' ') + ') is associated with higher repayment difficulty.'


class ReasonBuilder:
    """Picklable callable: maps risk-increasing contributions to the top-k plain-language reasons."""

    def __init__(self, texts: dict | None = None):
        self.texts = dict(texts or REASONS)

    def __call__(self, contrib: pd.Series, k: int = 3):
        pos = contrib[contrib > 0].sort_values(ascending=False).head(k)
        return [(c, float(v), reason_text(c, self.texts)) for c, v in pos.items()]


def source_col(feature_name: str, cols) -> str:
    """Map a transformed name (e.g. 'cat__lender_id_L12', 'num__duration') back to its raw column."""
    n = feature_name.split('__', 1)[-1]
    for c in sorted(cols, key=len, reverse=True):
        if n == c or n.startswith(c + '_'):
            return c
    return n


def contributions(pipe, Xdf: pd.DataFrame, feats, explainer=None) -> pd.DataFrame:
    """Per-applicant contributions on the log-odds scale, aggregated to raw columns.

    Linear models: coefficient x standardised value (relative to an average applicant).
    Tree models:   SHAP values (TreeExplainer).
    """
    prep, clf = pipe.named_steps['prep'], pipe.named_steps['clf']
    Xt = prep.transform(Xdf[feats])
    names = list(prep.get_feature_names_out())
    if hasattr(clf, 'coef_'):
        C = Xt * clf.coef_.ravel()
    else:
        if explainer is None:
            import shap
            explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
        C = np.asarray(sv)
    cdf = pd.DataFrame(C, columns=names).T.groupby(lambda n: source_col(n, feats)).sum().T
    cdf.index = Xdf.index
    return cdf
