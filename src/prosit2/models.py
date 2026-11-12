"""Preprocessing pipelines, estimator factory, calibration.

All learned transforms (imputation, scaling, encoding, resampling) live *inside* pipelines, so they
are fitted on training folds only.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CFG, F
from .features import CAT_COLS


def make_preprocessor(cols, min_cat_freq: int | None = None):
    """Numeric: median impute + standardise. Categorical: mode impute + one-hot with rare levels pooled;
    unseen levels at scoring time go to the infrequent bucket."""
    cat = [c for c in cols if c in CAT_COLS]
    num = [c for c in cols if c not in CAT_COLS]
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                         ('oh', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                              min_frequency=min_cat_freq or CFG['MIN_CAT_FREQ'], sparse_output=False))])
    return ColumnTransformer([('num', num_pipe, num), ('cat', cat_pipe, cat)], remainder='drop',
                             verbose_feature_names_out=True)


class SubsampledClassifier(BaseEstimator, ClassifierMixin):
    """Kernel SVMs scale ~O(n^2): fit on all positives + a random subset of negatives."""

    def __init__(self, estimator=None, max_n=15000, seed=42):
        self.estimator = estimator
        self.max_n = max_n
        self.seed = seed

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.RandomState(self.seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        k = max(self.max_n - len(pos), 1)
        if len(neg) > k:
            neg = rng.choice(neg, k, replace=False)
        idx = np.concatenate([pos, neg])
        self.est_ = clone(self.estimator).fit(X[idx], y[idx])
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return self.est_.decision_function(np.asarray(X))

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


class SubsampledRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator=None, max_n=15000, seed=42):
        self.estimator = estimator
        self.max_n = max_n
        self.seed = seed

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(len(y), min(self.max_n, len(y)), replace=False)
        self.est_ = clone(self.estimator).fit(X[idx], y[idx])
        return self

    def predict(self, X):
        return self.est_.predict(np.asarray(X))


class Calibrator:
    """Post-hoc calibration. Platt = logistic regression on the logit of the score; isotonic = monotone step map."""

    def __init__(self, method='platt'):
        self.method = method

    @staticmethod
    def _z(s):
        return logit(np.clip(np.asarray(s, float), 1e-6, 1 - 1e-6)).reshape(-1, 1)

    def fit(self, s, y):
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        if self.method == 'platt':
            self.m_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._z(s), y)
        elif self.method == 'isotonic':
            self.m_ = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(np.asarray(s, float), y)
        return self

    def predict(self, s):
        if self.method == 'platt':
            return self.m_.predict_proba(self._z(s))[:, 1]
        if self.method == 'isotonic':
            return self.m_.predict(np.asarray(s, float))
        return np.asarray(s, float)


def get_sampler(name: str, cfg: dict = CFG):
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    r, s = cfg['SAMPLING_RATIO'], cfg['SEED']
    return {'none': None, 'cw': None, 'builtin': None,
            'ros': RandomOverSampler(sampling_strategy=r, random_state=s),
            'rus': RandomUnderSampler(sampling_strategy=r, random_state=s),
            'smote': SMOTE(sampling_strategy=r, random_state=s),
            'bsmote': BorderlineSMOTE(sampling_strategy=r, random_state=s),
            'adasyn': ADASYN(sampling_strategy=r, random_state=s),
            'smote_tomek': SMOTETomek(sampling_strategy=r, random_state=s)}[name]


def make_clf(model: str, strategy: str, params: dict | None = None, pos_weight: float = 1.0, cfg: dict = CFG):
    import lightgbm as lgb
    import xgboost as xgb
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    seed = cfg['SEED']
    cw = 'balanced' if strategy == 'cw' else None
    spw = pos_weight if strategy == 'cw' else 1.0
    builders = {
        'majority': lambda: DummyClassifier(strategy='prior'),
        'logreg': lambda: LogisticRegression(C=1.0, max_iter=3000, class_weight=cw, random_state=seed),
        'knn': lambda: KNeighborsClassifier(n_neighbors=50, weights='distance', n_jobs=-1),
        'dtree': lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight=cw, random_state=seed),
        'rf': lambda: RandomForestClassifier(n_estimators=F(300, 60), min_samples_leaf=20, max_features='sqrt',
                                             class_weight=('balanced_subsample' if cw else None), n_jobs=-1, random_state=seed),
        'et': lambda: ExtraTreesClassifier(n_estimators=F(300, 60), min_samples_leaf=20,
                                           class_weight=('balanced_subsample' if cw else None), n_jobs=-1, random_state=seed),
        'lgbm': lambda: lgb.LGBMClassifier(n_estimators=F(400, 80), learning_rate=0.03, num_leaves=15, min_child_samples=50,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                           class_weight=cw, random_state=seed, n_jobs=-1, verbose=-1),
        'xgb': lambda: xgb.XGBClassifier(n_estimators=F(400, 80), learning_rate=0.03, max_depth=4, subsample=0.8,
                                         colsample_bytree=0.8, scale_pos_weight=spw, tree_method='hist',
                                         eval_metric='aucpr', random_state=seed, n_jobs=-1, verbosity=0),
        'brf': lambda: BalancedRandomForestClassifier(n_estimators=F(300, 60), min_samples_leaf=5, n_jobs=-1, random_state=seed,
                                                      sampling_strategy='all', replacement=True, bootstrap=True),
        'eec': lambda: EasyEnsembleClassifier(n_estimators=F(10, 4), random_state=seed, n_jobs=1),
        'svc': lambda: SubsampledClassifier(SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced'),
                                            max_n=cfg['MAX_SVC_ROWS'], seed=seed),
    }
    clf = builders[model]()
    if params:
        clf.set_params(**{k.replace('clf__', ''): v for k, v in params.items()})
    return clf


def build_estimator(model: str, strategy: str, feats, params: dict | None = None, pos_weight: float = 1.0, cfg: dict = CFG):
    from imblearn.pipeline import Pipeline as ImbPipeline
    steps = [('prep', make_preprocessor(feats))]
    s = get_sampler(strategy, cfg)
    if s is not None:
        steps.append(('sampler', s))
    steps.append(('clf', make_clf(model, strategy, params, pos_weight, cfg)))
    return ImbPipeline(steps)


def get_score(est, X):
    return est.predict_proba(X)[:, 1] if hasattr(est, 'predict_proba') else expit(est.decision_function(X))


def reg_est(name: str, feats, alpha: float | None = None, cfg: dict = CFG):
    """Task 1 regressors on log1p(Total_Amount)."""
    import lightgbm as lgb
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.svm import SVR
    seed = cfg['SEED']
    builders = {
        'mean': lambda: DummyRegressor(strategy='mean'),
        'ols': lambda: LinearRegression(),
        'ridge': lambda: Ridge(alpha=alpha or 10.0),
        'lasso': lambda: Lasso(alpha=alpha or 1e-3, max_iter=10000),
        'rf': lambda: RandomForestRegressor(n_estimators=F(200, 40), min_samples_leaf=10, n_jobs=-1, random_state=seed),
        'lgbm': lambda: lgb.LGBMRegressor(n_estimators=F(400, 80), learning_rate=0.05, num_leaves=31, min_child_samples=30,
                                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                                          n_jobs=-1, verbose=-1),
        'svr': lambda: SubsampledRegressor(SVR(C=1.0, epsilon=0.1), max_n=min(cfg['MAX_SVC_ROWS'], 8000), seed=seed),
    }
    return Pipeline([('prep', make_preprocessor(feats)), ('reg', builders[name]())])
