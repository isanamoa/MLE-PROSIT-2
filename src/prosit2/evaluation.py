"""Metrics, the cost matrix, threshold selection, calibration helpers, bootstrap and the
checkpointed experiment runner."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, fbeta_score, matthews_corrcoef, mean_absolute_error,
                             mean_squared_error, precision_recall_curve, precision_score, recall_score, roc_auc_score)

from .config import CFG
from .models import Calibrator, build_estimator, get_score


# ----------------------------------------------------------------------------- cost matrix
def cost_vectors(df: pd.DataFrame, recovery: float | None = None, coll: float | None = None, cfg: dict = CFG):
    """Per-loan money costs.

    FN_i (approve, borrower defaults) = principal_i * (1 - recovery) + principal_i * collection_cost_rate
    FP_i (decline a good borrower)    = fee_i = Total_Amount_to_Repay_i - Total_Amount_i   (forgone income)
    """
    recovery = cfg['RECOVERY_RATE'] if recovery is None else recovery
    coll = cfg['COLLECTION_COST_RATE'] if coll is None else coll
    P = df['Total_Amount'].values.astype(float)
    fee = (df['Total_Amount_to_Repay'] - df['Total_Amount']).clip(lower=0).values.astype(float)
    return P * (1 - recovery) + P * coll, fee


def expected_cost(y, decline, fn_c, fp_c) -> float:
    y = np.asarray(y)
    decline = np.asarray(decline).astype(bool)
    return float(fn_c[(y == 1) & ~decline].sum() + fp_c[(y == 0) & decline].sum())


def analytic_threshold(fn_c, fp_c) -> float:
    """Decline when p*FN > (1-p)*FP  <=>  p > FP / (FP + FN) (mean-cost approximation)."""
    return float(np.mean(fp_c) / (np.mean(fp_c) + np.mean(fn_c)))


def sweep(y, p, fn_c, fp_c, extra=()):
    y = np.asarray(y)
    grid = np.unique(np.concatenate([np.quantile(p, np.linspace(0.40, 0.9995, 400)), np.array(extra, float), [1.01]]))
    rows = []
    for t in grid:
        d = p >= t
        tp, fp = ((y == 1) & d).sum(), ((y == 0) & d).sum()
        rows.append(dict(thr=t, decline_rate=d.mean(), approval=1 - d.mean(), cost=expected_cost(y, d, fn_c, fp_c),
                         precision=tp / max(d.sum(), 1), recall=tp / max((y == 1).sum(), 1), fpr=fp / max((y == 0).sum(), 1)))
    s = pd.DataFrame(rows)
    s['f1'] = 2 * s.precision * s.recall / (s.precision + s.recall).replace(0, np.nan)
    return s


def choose_threshold(sw: pd.DataFrame, min_approval: float) -> float:
    ok = sw[sw.approval >= min_approval]
    return float(ok.loc[ok.cost.idxmin(), 'thr'])


# ----------------------------------------------------------------------------- classification metrics
def clf_metrics(y, s, thr, fn_c=None, fp_c=None) -> dict:
    y, s = np.asarray(y), np.asarray(s)
    yh = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    out = dict(thr=float(thr), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
               accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
               recall=recall_score(y, yh, zero_division=0), specificity=tn / max(tn + fp, 1),
               f1=f1_score(y, yh, zero_division=0), f2=fbeta_score(y, yh, beta=2, zero_division=0),
               balanced_accuracy=balanced_accuracy_score(y, yh),
               mcc=matthews_corrcoef(y, yh) if len(set(yh)) > 1 else 0.0,
               fpr=fp / max(fp + tn, 1), fnr=fn / max(fn + tp, 1), flag_rate=yh.mean())
    if fn_c is not None:
        out['expected_cost'] = expected_cost(y, yh, fn_c, fp_c)
    return out


def f1_opt_threshold(y, s) -> float:
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[np.nanargmax(f)]) if len(t) else 0.5


def topq_threshold(s, q) -> float:
    return float(np.quantile(s, 1 - q))


def threshold_free(y, s) -> dict:
    return dict(pr_auc=average_precision_score(y, s), roc_auc=roc_auc_score(y, s), brier=brier_score_loss(y, np.clip(s, 0, 1)))


def ece(y, p, bins=10) -> float:
    """Expected calibration error with quantile bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def cv_calibrate(oof, y, method, folds):
    """Calibrated OOF probabilities: calibrator fitted on other folds' OOF scores, applied to the held-out fold."""
    out = np.full(len(oof), np.nan)
    for tr_i, va_i in folds:
        out[va_i] = Calibrator(method).fit(oof[tr_i], np.asarray(y)[tr_i]).predict(oof[va_i])
    return out


def choose_calibrator(cal_tbl: pd.DataFrame, rel_margin: float = 0.02) -> str:
    """Prefer Platt scaling (monotone: preserves the ranking, so PR-AUC is unchanged). Choose isotonic only if its
    cross-validated Brier score is at least ``rel_margin`` (relative) lower: isotonic maps scores to a step function,
    which creates ties and can lower PR-AUC (visible in the 'pr_auc' column of the dev-OOF calibration table)."""
    b_p, b_i = cal_tbl.loc['platt', 'brier'], cal_tbl.loc['isotonic', 'brier']
    return 'isotonic' if b_i < (1 - rel_margin) * b_p else 'platt'


def calibration_report(oof, y, folds):
    cands = {'raw': oof, 'platt': cv_calibrate(oof, y, 'platt', folds), 'isotonic': cv_calibrate(oof, y, 'isotonic', folds)}
    rows = {k: dict(brier=brier_score_loss(y, np.clip(p, 0, 1)), ece=ece(y, p), mean_pred=p.mean(), observed=np.mean(y),
                    pr_auc=average_precision_score(y, p)) for k, p in cands.items()}
    return pd.DataFrame(rows).T, cands


# ----------------------------------------------------------------------------- regression metrics
def reg_metrics(yl, pl) -> dict:
    """RMSE / MAE on the log scale and on the original currency scale, plus R^2 on log scale."""
    yo, po = np.expm1(yl), np.clip(np.expm1(pl), 0, None)
    return dict(rmse_log=float(np.sqrt(mean_squared_error(yl, pl))), mae_log=float(mean_absolute_error(yl, pl)),
                rmse_orig=float(np.sqrt(mean_squared_error(yo, po))), mae_orig=float(mean_absolute_error(yo, po)),
                r2_log=float(1 - np.sum((yl - pl) ** 2) / np.sum((yl - np.mean(yl)) ** 2)))


# ----------------------------------------------------------------------------- bootstrap
def cluster_index(groups):
    """Row indices per cluster (customer) for a cluster bootstrap."""
    uc, inv = np.unique(np.asarray(groups), return_inverse=True)
    return [np.where(inv == i)[0] for i in range(len(uc))]


def ci95(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def psi(a: pd.Series, b: pd.Series, bins=10, cat=False) -> float:
    """Population stability index of b relative to a (0.1 = moderate, 0.25 = major shift)."""
    if cat:
        pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
        k = pa.index.union(pb.index)
        pa, pb = pa.reindex(k).fillna(0) + 1e-4, pb.reindex(k).fillna(0) + 1e-4
    else:
        e = np.unique(np.nanquantile(a.dropna(), np.linspace(0, 1, bins + 1)))
        e[0], e[-1] = -np.inf, np.inf
        pa = np.histogram(a.dropna(), e)[0] / max(a.notna().sum(), 1) + 1e-4
        pb = np.histogram(b.dropna(), e)[0] / max(b.notna().sum(), 1) + 1e-4
    return float(np.sum((np.asarray(pb) - np.asarray(pa)) * np.log(np.asarray(pb) / np.asarray(pa))))


# ----------------------------------------------------------------------------- experiment runner
class ExperimentRunner:
    """Grouped-CV experiments with checkpoint/resume.

    For every experiment: OOF scores on ``folds`` -> metrics -> refit on all dev -> save model + results.
    Re-running skips experiments whose files already exist (delete them, or pass ``force=True``, to retrain).
    """

    def __init__(self, X, y, folds, fn_c, fp_c, dirs: dict, pos_weight: float, cfg: dict = CFG):
        self.X, self.y, self.folds = X, y, folds
        self.fn_c, self.fp_c = fn_c, fp_c
        self.dirs, self.pos_weight, self.cfg = dirs, pos_weight, cfg
        self.ckpt = {'base': dirs['ckpt_base'], 'imb': dirs['ckpt_imb'], 'opt': dirs['ckpt_opt'], 'abl': dirs['ckpt_abl']}
        self.registry_csv = Path(dirs['results']) / 'experiment_metrics.csv'
        self.res: dict = {}
        self.failed: list = []

    # registry ------------------------------------------------------------
    def load_registry(self) -> pd.DataFrame:
        return pd.read_csv(self.registry_csv) if self.registry_csv.exists() else pd.DataFrame()

    def _save_row(self, row):
        reg = self.load_registry()
        if len(reg):
            reg = reg[reg.exp_id != row['exp_id']]
        pd.concat([reg, pd.DataFrame([row])], ignore_index=True).to_csv(self.registry_csv, index=False)

    def table(self, stage=None) -> pd.DataFrame:
        r = self.load_registry()
        if stage and len(r):
            r = r[r.stage == stage]
        return r.sort_values('pr_auc', ascending=False).reset_index(drop=True) if len(r) else r

    def model_path(self, exp_id, ckpt='opt'):
        return self.ckpt[ckpt] / f'{exp_id}.model.joblib'

    # run -----------------------------------------------------------------
    def run(self, exp_id, stage, model, strategy, feats, params=None, ckpt='base', folds=None, force=False, verbose=True):
        folds = folds or self.folds
        rpath, mpath = self.ckpt[ckpt] / f'{exp_id}.res.joblib', self.ckpt[ckpt] / f'{exp_id}.model.joblib'
        if rpath.exists() and mpath.exists() and not force:
            self.res[exp_id] = joblib.load(rpath)
            if verbose:
                print(f'  resumed {exp_id}')
            return self.res[exp_id]
        X, y, cfg = self.X, self.y, self.cfg
        try:
            t0 = time.time()
            oof = np.full(len(y), np.nan)
            fold_ap = []
            for tr_i, va_i in folds:
                est = build_estimator(model, strategy, feats, params, self.pos_weight, cfg)
                est.fit(X.iloc[tr_i], y.iloc[tr_i])
                s = get_score(est, X.iloc[va_i])
                oof[va_i] = s
                fold_ap.append(average_precision_score(y.iloc[va_i], s))
            assert not np.isnan(oof).any()
            tf = threshold_free(y, oof)
            m_f1 = clf_metrics(y, oof, f1_opt_threshold(y, oof), self.fn_c, self.fp_c)
            m_tq = clf_metrics(y, oof, topq_threshold(oof, cfg['TOP_Q']), self.fn_c, self.fp_c)
            full = build_estimator(model, strategy, feats, params, self.pos_weight, cfg).fit(X, y)
            row = dict(exp_id=exp_id, stage=stage, model=model, strategy=strategy, n_features=len(feats),
                       params=json.dumps(params or {}, default=str), **tf,
                       ap_fold_mean=np.mean(fold_ap), ap_fold_sd=np.std(fold_ap, ddof=1),
                       **{f'f1opt_{k}': m_f1[k] for k in ['thr', 'precision', 'recall', 'f1', 'f2', 'mcc',
                                                          'balanced_accuracy', 'fpr', 'fnr']},
                       **{f'top_{k}': m_tq[k] for k in ['precision', 'recall', 'f1', 'fpr']},
                       apparent_pr_auc=average_precision_score(y, get_score(full, X)),
                       fit_seconds=round(time.time() - t0, 1))
            res = dict(row=row, oof=oof, fold_ap=fold_ap, feats=list(feats))
            joblib.dump(full, mpath)
            joblib.dump(res, rpath)
            self._save_row(row)
            self.res[exp_id] = res
            if verbose:
                print(f"  done {exp_id:<34} PR-AUC={row['pr_auc']:.4f} (fold {row['ap_fold_mean']:.4f}+-{row['ap_fold_sd']:.4f}) "
                      f"ROC={row['roc_auc']:.3f} [{row['fit_seconds']}s]")
            return res
        except Exception as e:  # keep the sweep going; failures are reported at the end
            self.failed.append(dict(exp_id=exp_id, error=repr(e)))
            print(f'  FAILED {exp_id}: {e!r}')
            (Path(self.dirs['reports']) / 'failed_experiments.json').write_text(json.dumps(self.failed, indent=2))
            return None
