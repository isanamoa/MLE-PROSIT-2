"""ProSit 2 (ICS553): responsible credit scoring for NkwaPay, Kenya -> Ghana.

Modules
-------
config          paths, configuration, output folders, environment record
data            loading, cleaning, macro + Findex loaders, temporal split, grouped folds
features        time-safe feature engineering (shared with the scoring service)
models          preprocessing pipelines, estimator factory, calibration
evaluation      cost matrix, metrics, threshold selection, bootstrap, experiment runner
explain         contributions and plain-language reason codes
scoring_service the deployable Scorer
"""
__all__ = ['config', 'data', 'features', 'models', 'evaluation', 'explain', 'scoring_service']
