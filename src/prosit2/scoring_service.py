"""Scoring service (ProSit Part E).

Usage::

    from prosit2.scoring_service import Scorer
    scorer = Scorer('outputs/checkpoints/best_models/final_bundle.joblib')
    scorer.score({... one application, see REQUIRED_FIELDS ...})

Returns the calibrated probability of default, a decision (approve / approve with reduced limit and
manual review / decline), an approved limit, the top-3 plain-language reasons and the latency.
The same feature code as in training is used, so there is no train/serve skew.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from . import features as fe
from .explain import contributions

REQUIRED_FIELDS = ['customer_id', 'country_id', 'loan_type', 'lender_id', 'duration', 'disbursement_date',
                   'Total_Amount', 'Total_Amount_to_Repay', 'Amount_Funded_By_Lender', 'Lender_portion_Funded',
                   'Lender_portion_to_be_repaid', 'New_versus_Repeat']


class Scorer:
    def __init__(self, bundle_path):
        self.b = joblib.load(bundle_path)
        self.feats = self.b['features']
        h = self.b['history']
        self.hist = {k: g for k, g in h.groupby('customer_id')}
        self.empty = h.iloc[0:0]
        self.pipe = self.b['pipeline']
        self.clf = self.pipe.named_steps['clf']
        self.explainer = None
        if not hasattr(self.clf, 'coef_'):
            import shap
            self.explainer = shap.TreeExplainer(self.clf)

    def _features(self, app: dict) -> pd.DataFrame:
        miss = [f for f in REQUIRED_FIELDS if f not in app or app[f] is None]
        if miss:
            raise ValueError(f'missing required fields: {miss}')
        df = pd.DataFrame([app])
        df['disbursement_date'] = pd.to_datetime(df['disbursement_date'])
        df['due_date'] = df['disbursement_date'] + pd.to_timedelta(df['duration'], unit='D')
        for c in fe.CAT_COLS:
            df[c] = df[c].astype(str)
        f = fe.add_calendar(df)
        f = fe.add_terms(f)
        f = fe.add_history(f, self.hist.get(app['customer_id'], self.empty), self.b['grace_days'])
        return fe.add_macro(f, self.b['econ'], self.b.get('macro_lag', 1))

    def score(self, app: dict) -> dict:
        t0 = time.perf_counter()
        f = self._features(app)
        X = f[self.feats]
        s = self.pipe.predict_proba(X)[:, 1] if hasattr(self.pipe, 'predict_proba') else expit(self.pipe.decision_function(X))
        pdv = float(self.b['calibrator'].predict(s)[0])
        thr = self.b['threshold']
        lim_log = float(self.b['limit_pipeline'].predict(f)[0])
        sd = self.b['limit_resid_sd']
        req = float(app['Total_Amount'])
        if pdv >= thr:
            decision, limit = 'decline', 0.0
        elif pdv >= self.b['review_fraction'] * thr:
            decision, limit = 'approve_reduced_limit_manual_review', min(req, float(np.expm1(lim_log)))
        else:
            decision, limit = 'approve', min(req, float(np.expm1(lim_log + sd)))
        contrib = contributions(self.pipe, f, self.feats, self.explainer).iloc[0]
        reasons = [{'feature': k, 'contribution': v, 'text': t} for k, v, t in self.b['reason_fn'](contrib)]
        return dict(probability_of_default=pdv, decision=decision, approved_limit=limit, threshold=thr,
                    top_reasons=reasons, model_version=self.b['version'],
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1))
