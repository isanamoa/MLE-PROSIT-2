# MLE-PROSIT-2: Responsible credit scoring for digital micro-lending

ICS553 Machine Learning Essentials, Prosit 2 (Group 4). NkwaPay wants to lend in Ghana using a model trained on labelled
Kenyan loans. This repository builds:

* **Task 2 (classification):** a calibrated probability of default with a cost-based approve/decline threshold;
* **Task 1 (regression):** a loan-amount (limit) model;
* a **Kenya → Ghana transfer study**, a **fairness and regulatory audit**, and a **scoring service** that returns a
  decision, a limit and the top-3 plain-language reasons.

## Repository layout

```
├── README.md                 ← you are here (how to reproduce end-to-end)
├── requirements.txt          ← pinned library versions
├── data/                     ← NOT in git: put Train.csv, Test.csv, economic_indicators.csv, micro_gha.csv here
├── src/prosit2/              ← data-loading and modelling code shared by all notebooks and the scoring service
│   ├── config.py             ← the ONLY place to change settings (paths, seeds, cost assumptions, FAST mode)
│   ├── data.py               ← loading, cleaning, macro + Findex loaders, temporal split, grouped folds
│   ├── features.py           ← time-safe feature engineering (calendar, terms, history, macro)
│   ├── models.py             ← preprocessing pipelines, estimator factory, calibration
│   ├── evaluation.py         ← cost matrix, metrics, threshold selection, bootstrap, experiment runner
│   ├── explain.py            ← per-applicant contributions and reason codes
│   └── scoring_service.py    ← Scorer.score(application) -> PD, decision, limit, reasons, latency
├── notebooks/
│   ├── 01_eda.ipynb                    ← Part A framing, feature audit, cost matrix; EDA; feature checks
│   ├── 02_classification.ipynb         ← Task 2: validation integrity, 11 models, imbalance, tuning, ablation,
│   │                                      derivations, calibration, threshold, test + bootstrap CIs, reason codes
│   ├── 03_regression.ipynb             ← Task 1: OLS/Ridge/Lasso derivation, 7 models, residual diagnostics
│   └── 04_transfer_and_fairness.ipynb  ← Part C transfer, Part D fairness/proxy/mitigation/model card/regulation,
│                                          Findex, borrower explainer, Part E scoring service (parity + latency)
├── docs/                     ← EDA report (PROSIT2_EDA_Report.pdf)
├── eda/                      ← stand-alone EDA script and figures behind docs/PROSIT2_EDA_Report.pdf
└── outputs/                  ← created by the notebooks (git-ignored): checkpoints, results, plots, reports
```

`ProSit2_Credit_Scoring.ipynb` (the original single Colab notebook) is kept for reference; the four notebooks above
replace it. (Earlier `Improved_Prosit2_Credit_Scoring*` drafts are not included: their borrower-history code misaligned rows.)

## Reproduce end-to-end

### 1. Environment
```bash
git clone <this repo> && cd MLE-PROSIT-2
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data
Download from the [African Credit Scoring Challenge (Zindi)](https://zindi.africa/competitions/african-credit-scoring-challenge)
and place in `data/`:

| File | Required | Notes |
|---|---|---|
| `Train.csv`, `Test.csv` | yes | 68,654 labelled Kenyan loans; 18,594 unlabelled Kenya + Ghana applications |
| `economic_indicators.csv` | recommended | wide format (`Country, Indicator, YR2001…YR2023`); reshaped automatically |
| `micro_gha.csv` | recommended | [Global Findex 2021 Ghana microdata](https://microdata.worldbank.org/index.php/catalog/4646) (1,000 adults, survey-weighted; free registration). Used in 04 §3.3 (Part D.4) |

### 3. Run the notebooks in order
Interactively: `jupyter lab`, open `notebooks/`, and *Run All* in 01 → 02 → 03 → 04.

Headless (same result):
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_classification.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_regression.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_transfer_and_fairness.ipynb
```
02 and 03 save hand-off files to `outputs/artifacts/`, which 04 reads, so the order matters.

Measured on an 8-core Windows laptop (2026-11-12); a free Colab CPU is roughly 2–3× slower:

| Notebook | Full run | FAST mode |
|---|---|---|
| 01_eda | 0.1 min | 0.3 min |
| 02_classification | 19 min (50 CV experiments, tuning, ablations) | 3 min |
| 03_regression | 14 min (RF and SVR regressors dominate) | 1 min |
| 04_transfer_and_fairness | 0.6 min | 0.5 min |
| **Total** | **≈34 min** | **≈5 min** |

**Smoke test first.** Set the environment variable `PROSIT2_FAST=1` (or uncomment `CFG.update(FAST=True)` in each
notebook's setup cell). FAST mode uses fewer trees, folds and bootstrap draws and writes to `outputs_fast/`, so it can never
overwrite or be resumed into the full run.

**Resume.** 02 checkpoints every experiment in `outputs/checkpoints/`; re-running skips finished experiments. Delete a
`*.res.joblib` file (or the whole `outputs/` folder) to retrain.

### 4. Google Colab
Every notebook starts with a **Colab setup cell** (it does nothing locally). One-time preparation:

1. In Google Drive create the folder `MyDrive/MLE Prosit 2/data/` and upload `Train.csv`, `Test.csv`,
   `economic_indicators.csv` and `micro_gha.csv`. (Different folder? Edit `DRIVE_DIR` in the setup cell.)
2. Open a notebook in Colab: *File → Open notebook → GitHub*, paste `isanamoa/MLE-PROSIT-2`, pick e.g.
   `notebooks/01_eda.ipynb`.
3. Run the setup cell first, then *Runtime → Run all*. It mounts Drive, clones the repo, installs the libraries Colab lacks
   (imbalanced-learn, lightgbm, xgboost, shap, fairlearn) without touching Colab's numpy/pandas/scikit-learn (so no restart),
   and points outputs at `MyDrive/MLE Prosit 2/outputs/`.
4. Run 01 → 02 → 03 → 04. **Each Colab notebook runs on its own VM**, so the hand-off files (`outputs/artifacts/`) and the
   checkpoints must be on Drive, which the setup cell arranges. If Colab disconnects during 02, just re-run it: finished
   experiments are loaded from the Drive checkpoints.

A free Colab CPU has 2 cores, so expect 02 to take roughly 40–60 minutes (FAST mode: under 10). The code was also tested
against an older library stack (pandas 2.2, numpy 2.1, scikit-learn 1.6) to cover Colab's preinstalled versions.

### 5. Use the scoring service
```python
import sys; sys.path.insert(0, 'src')
from prosit2.scoring_service import Scorer, REQUIRED_FIELDS
scorer = Scorer('outputs/checkpoints/best_models/final_bundle.joblib')
scorer.score({'customer_id': 256582, 'country_id': 'Kenya', 'loan_type': 'Type_5', 'lender_id': '251804', 'duration': 14,
              'disbursement_date': '2022-12-17', 'Total_Amount': 24200.0, 'Total_Amount_to_Repay': 45494.0,
              'Amount_Funded_By_Lender': 3226.0, 'Lender_portion_Funded': 0.1333, 'Lender_portion_to_be_repaid': 6065.0,
              'New_versus_Repeat': 'Repeat Loan'})
# -> {'probability_of_default': ..., 'decision': 'approve' | 'approve_reduced_limit_manual_review' | 'decline',
#     'approved_limit': ..., 'top_reasons': [...], 'latency_ms': ...}
```

## Key outputs (after a full run)
| Path | Content |
|---|---|
| `outputs/results/experiment_metrics.csv` | every experiment, OOF metrics |
| `outputs/results/validation_scheme_comparison.csv` | stratified vs grouped vs temporal CV |
| `outputs/results/final_test_metrics.csv`, `test_summary_ci.csv` | one-shot test results + bootstrap CIs |
| `outputs/results/fairness_*.csv`, `proxy_analysis.csv`, `transfer_*.csv` | Part C/D tables |
| `outputs/reports/model_card.md`, `borrower_explainer.md`, `final_model_report.txt` | Part D/E documents |
| `outputs/reports/environment_0*.json` | library versions + config used for each notebook |
| `outputs/checkpoints/best_models/final_bundle.joblib` | deployable model bundle |

## Configuration and assumptions to justify in the report
All in `src/prosit2/config.py`:
* `RECOVERY_RATE = 0.20`, `COLLECTION_COST_RATE = 0.08`: **placeholders** for the cost matrix; justify for Ghana.
* `CURRENCY = 'KES'`, `FX_TO_GHS = None`: amounts in the data are Kenyan shillings. Set `FX_TO_GHS` (GHS per KES, with
  source and date) to report costs in cedis.
* `MIN_APPROVAL_RATE = 0.80`: business constraint on the operating threshold.
* `TEST_FRAC = 0.20`, `GRACE_DAYS = 0`, `MACRO_LAG_YEARS = 1`, `SEED = 42`.

## Reproducibility statement
* One configuration file; all paths relative to the repo root (overridable by `PROSIT2_DATA_DIR` / `PROSIT2_OUT_DIR`).
* Fixed seeds (`SEED = 42` for reported folds and models, `SEED + 1` for tuning folds); pinned library versions; each notebook
  writes the versions and config it ran with to `outputs/reports/environment_0*.json`.
* No learned statistic touches the test set: preprocessing, resampling, tuning, calibration and threshold selection use dev
  data only; the temporal test set is scored once. The brute-force history check in 01 must report 0 mismatches.
* Multi-threaded LightGBM/XGBoost can produce tiny run-to-run differences in the last decimal places.

## AI-use declaration
Parts of this code were refactored with AI assistance (Claude). The team must state in the report which parts, what was
checked, changed or rejected, and be able to explain and reproduce every result without AI, as required by the syllabus.
