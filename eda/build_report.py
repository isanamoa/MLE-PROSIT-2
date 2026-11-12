"""Builds the EDA PDF report from eda/eda_stats.json and eda/figures/*.png.
Run from repo root after eda_analysis.py:  python eda/build_report.py
"""
import json, os
import matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                PageBreak, KeepTogether, ListFlowable, ListItem)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "eda", "figures")
S = json.load(open(os.path.join(ROOT, "eda", "eda_stats.json")))
OUT = os.path.join(ROOT, "PROSIT2_EDA_Report.pdf")

fd = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(fd, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", os.path.join(fd, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", os.path.join(fd, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

INK = colors.HexColor("#0b0b0b"); INK2 = colors.HexColor("#52514e")
BLUE = colors.HexColor("#2a78d6"); ORANGE = colors.HexColor("#eb6834")
PALE = colors.HexColor("#f3f2ee"); RULE = colors.HexColor("#d6d4cd")
PALEBLUE = colors.HexColor("#eaf2fb"); PALEOR = colors.HexColor("#fdf0ea")

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["Normal"], fontName="DV", fontSize=9.2, leading=13.2, textColor=INK, spaceAfter=5)
SM = ParagraphStyle("SM", parent=B, fontSize=7.8, leading=10, spaceAfter=0)
SMB = ParagraphStyle("SMB", parent=SM, fontName="DVB")
CAP = ParagraphStyle("CAP", parent=B, fontSize=7.8, leading=10, textColor=INK2, alignment=TA_LEFT, spaceAfter=10)
H1 = ParagraphStyle("H1", parent=B, fontName="DVB", fontSize=14, leading=18, spaceBefore=10, spaceAfter=6, textColor=INK, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=B, fontName="DVB", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=4, textColor=BLUE, keepWithNext=1)
TITLE = ParagraphStyle("T", parent=B, fontName="DVB", fontSize=22, leading=27, spaceAfter=6)
SUB = ParagraphStyle("SUB", parent=B, fontSize=11, leading=15, textColor=INK2)
CALL = ParagraphStyle("CALL", parent=B, fontSize=9, leading=12.6, spaceAfter=0)

story = []
P = lambda t, st=B: story.append(Paragraph(t, st))


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def fig(name, w=17, cap=None):
    from reportlab.lib.utils import ImageReader
    path = os.path.join(FIG, name)
    iw, ih = ImageReader(path).getSize()
    h = w * ih / iw
    items = [Image(path, width=w * cm, height=h * cm)]
    if cap:
        items.append(Paragraph(cap, CAP))
    story.append(KeepTogether(items))


def table(rows, widths, header=True, zebra=True, fs=7.8, keep=None):
    st = ParagraphStyle("t", parent=SM, fontSize=fs, leading=fs + 2.2)
    sth = ParagraphStyle("th", parent=st, fontName="DVB", textColor=colors.white)
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(str(c), sth if (header and i == 0) else st) for c in r])
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    sty = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        sty.append(("BACKGROUND", (0, 0), (-1, 0), BLUE))
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                sty.append(("BACKGROUND", (0, i), (-1, i), PALE))
    t.setStyle(TableStyle(sty))
    if keep is None:
        keep = len(rows) <= 12
    story.append(KeepTogether([t, Spacer(1, 8)]) if keep else t)
    if not keep:
        story.append(Spacer(1, 8))


def callout(title, text, bg=PALEBLUE, bar=BLUE):
    t = Table([[Paragraph(f"<b>{title}</b><br/>{text}", CALL)]], colWidths=[17 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(t); story.append(Spacer(1, 8))


def bullets(items, st=B):
    story.append(ListFlowable([ListItem(Paragraph(i, st), leftIndent=12, value="•") for i in items],
                              bulletType="bullet", start="•", leftIndent=12, bulletFontName="DV", bulletFontSize=8))
    story.append(Spacer(1, 4))


# ---------------- numbers ----------------
n, d = S["train_shape"][0], S["n_default"]
cv = {(r["features"], r["model"]): r for r in S["cv_demo"]}
core_h = cv[("Core application features", "HistGB (balanced)")]
core_l = cv[("Core application features", "LogReg (balanced)")]
safe_h = cv[("+ safe history (merge_asof)", "HistGB (balanced)")]
safe_l = cv[("+ safe history (merge_asof)", "LogReg (balanced)")]
fee_h = cv[("+ repay-derived (suspect)", "HistGB (balanced)")]
leak_h = cv[("+ leaky customer target-mean", "HistGB (balanced)")]
gh = S["ghana"]; C = S["cost"]
tenure = {r["New_versus_Repeat"]: r for r in S["rate_tenure"]}
fee_bins = S["fee_rate_bins"]
fb = {r["band"]: r for r in fee_bins}
dq = S["dq"]

# ---------------- cover ----------------
story.append(Spacer(1, 2.2 * cm))
P("ICS553 Machine Learning Essentials · Prosit 2", SUB)
P("Responsible Credit Scoring for Digital Micro-lending", TITLE)
P("Exploratory Data Analysis and Solution Guide", ParagraphStyle("s2", parent=SUB, fontSize=14, leading=18, textColor=BLUE))
story.append(Spacer(1, 0.6 * cm))
P("Prepared from the perspective of a Machine Learning Expert &amp; Trainer for the NkwaPay credit-scoring team. "
  "Every number in this report was computed from the files in this repository "
  "(<i>Train.csv, Test.csv, economic_indicators.csv, VariableDefinitions.txt</i>) by "
  "<i>eda/eda_analysis.py</i>; the report itself is produced by <i>eda/build_report.py</i>, so it can be regenerated end-to-end.", B)
story.append(Spacer(1, 0.5 * cm))
kp = [
    ["Train rows", f"{n:,}", "Defaults", f"{d:,} ({pct(S['default_rate'], 2)})"],
    ["Test rows", f"{S['test_shape'][0]:,}", "Test split", f"Kenya {S['test_by_country']['Kenya']:,} · Ghana {S['test_by_country']['Ghana']:,}"],
    ["Customers (train)", f"{S['n_customers_train']:,}", "Train period", f"{S['train_dates'][0]} → {S['train_dates'][1]}"],
    ["Missing values", f"{S['missing_train']} / {S['missing_test']}", "Customer overlap train∩test", f"{S['overlap_customers']:,} customers · {S['overlap_loans']} loan ids"],
]
table(kp, [3.3, 3.2, 4.4, 6.1], header=False, zebra=True, fs=8.5)
story.append(Spacer(1, 0.4 * cm))
P("<b>Contents</b>", H2)
for i, s in enumerate(["Executive summary: the ten things you must know", "The problem as an ML task", "Data inventory and quality",
                       "Target and class imbalance", "Distributions and skew", "Categorical fields and segment risk",
                       "Time: drift and regime change", "Customers, repeat borrowing and train/test overlap",
                       "Leakage audit (feature audit table)", "Validation design: what the numbers really are",
                       "Kenya → Ghana: covariate and concept shift", "Macro-economic indicators",
                       "A data-driven cost matrix and threshold", "Fairness pre-audit", "Recommended solution roadmap",
                       "Limitations and reproducibility"], 1):
    P(f"{i}. {s}", ParagraphStyle("toc", parent=B, spaceAfter=1))
story.append(PageBreak())

# ---------------- 1 exec summary ----------------
P("1. Executive summary: the ten things you must know", H1)
P("The brief warns that success depends on <i>leakage-free validation and time-safe features</i> more than on clever models. "
  "The data strongly confirm this. These are the findings that should shape every design decision:")
bullets([
    f"<b>Severe imbalance.</b> {d:,} defaults in {n:,} loans ({pct(S['default_rate'], 2)}). Accuracy is useless (a model that approves "
    f"everyone scores {pct(1 - S['default_rate'], 1)}). Use PR-AUC (no-skill = {S['baseline_prauc']:.3f}), F1 at a cost-chosen threshold, and calibration.",
    f"<b>A hidden leak in <i>Total_Amount_to_Repay</i>.</b> The implied fee rate (repay ÷ amount − 1) has a univariate ROC-AUC of "
    f"{S['univariate_auc']['fee_rate']:.3f}. Repaid loans have a median 'fee' of {pct(dq['fee_rate_mean_repaid'], 2)}; defaulted loans {pct(dq['fee_rate_mean_default'], 1)}. "
    f"Loans whose 'fee' exceeds 20% default {pct((fb['20-50%']['default'] + fb['>50%']['default']) / (fb['20-50%']['default'] + fb['>50%']['default'] + fb['20-50%']['repaid'] + fb['>50%']['repaid']), 0)} of the time. "
    "This pattern is what you would see if penalties/late fees were added after the due date. Treat this column (and <i>Lender_portion_to_be_repaid</i>) as post-outcome unless the lender confirms it is the contracted amount.",
    f"<b>Honest performance is modest.</b> With only decision-time features, gradient boosting reaches PR-AUC ≈ {core_h['group_prauc']:.2f} (GroupKFold) and "
    f"{core_h['temporal_prauc']:.2f} (temporal), against {fee_h['stratified_prauc']:.2f} if the suspect repay columns are used, and {leak_h['stratified_prauc']:.2f} with a naive customer target-mean. "
    "If your notebook shows PR-AUC above 0.6, suspect leakage first.",
    f"<b>Strong regime change in time.</b> {pct(S['rate_by_year'][1]['count'] / n, 0)} of training loans were disbursed in 2022, almost all between July and November, when the default rate was ~1%. "
    f"Before and after that window default rates run at 10–20% (2023: {pct(S['rate_by_year'][2]['mean'])}, 2024: {pct(S['rate_by_year'][3]['mean'])}). The temporal split therefore tests on a riskier world.",
    f"<b>Product and lender are the dominant risk drivers.</b> Type_1 loans (90% of volume) default {pct(S['rate_loan_type'][0]['mean'], 2)}; Type_14 and Type_2 default above 60%. "
    "Lender 267278 (94% of volume) defaults 1.2%; the three smaller lenders 9–41%. Duration is monotonic: 7-day loans 1.2%, &gt;90-day loans 45%.",
    f"<b>Ghana is almost entirely out-of-distribution.</b> {pct(S['ghana_rows_unseen_type_or_lender'], 0)} of Ghana test rows come from a loan type or lender that never appears in Kenyan training "
    f"(Type_3 = {pct(gh['loan_type']['Type_3'], 0)} of Ghana; four new lenders). A domain classifier separates Kenya from Ghana with AUC {S['domain_auc']:.3f}, "
    f"and still {S['domain_auc_minimal']:.3f} using only duration, tenure, lender share and calendar fields. Importance weighting is not viable (effective sample size ≈ 0).",
    f"<b>Customers repeat heavily.</b> {S['n_customers_train']:,} customers hold {n:,} loans (median {S['loans_per_customer']['median']:.0f}, max {S['loans_per_customer']['max']}). "
    f"Prior default is the single strongest legitimate signal: {pct(S['safe_hist']['rate_if_prior_default'], 0)} default rate after a prior default vs {pct(S['safe_hist']['rate_if_no_prior_default'], 1)} otherwise. "
    "History features must be built with a backward <i>merge_asof</i> (strictly earlier dates) and validated with GroupKFold.",
    f"<b>Thin-file borrowers are much riskier.</b> New loans default {pct(tenure['New Loan']['mean'], 1)} vs {pct(tenure['Repeat Loan']['mean'], 1)} for repeat loans, "
    f"but there are only {tenure['New Loan']['count']} of them. Any model will tend to penalise them; this is the central fairness tension.",
    f"<b>Overlap with Test.</b> {S['overlap_customers']:,} customers and {S['overlap_loans']} loan ids appear in both files; all of these are Kenyan. "
    "Historical outcomes of these customers can legitimately feed history features for Kenyan test rows, but never for Ghana (no Ghana customer is known).",
    f"<b>Margins are thin, so costs are lopsided.</b> The median 7-day Kenyan loan is KES {C['median_amount_7d']:,.0f} with a median fee of {pct(C['median_fee_rate_7d'], 2)}. "
    "Losing the principal costs ~100× the fee forgone, so a cost-optimal threshold sits near 1%, which would decline many borrowers. Your cost matrix and an approval-rate floor must be argued explicitly.",
])
story.append(PageBreak())

# ---------------- 2 problem framing ----------------
P("2. The problem as an ML task", H1)
P("NkwaPay must decide at <b>application time</b> whether to approve, how much to lend and at what price. "
  "The closest column to that moment is <b>disbursement_date</b> (no application timestamp exists). Anything that is only known after disbursement "
  "(repayment behaviour, penalties, the outcome itself) must be excluded.")
table([
    ["", "Task 1 – Limit setting (regression)", "Task 2 – Default risk (classification)"],
    ["Target", "log1p(Total_Amount): the amount to approve", "target (1 = defaulted)"],
    ["Unit of prediction", "One loan application", "One loan application"],
    ["Must NOT use as input", "Amount_Funded_By_Lender, Lender_portion_to_be_repaid, Total_Amount_to_Repay (all are arithmetic functions of Total_Amount)", "target, Total_Amount_to_Repay, Lender_portion_to_be_repaid (post-outcome suspect), any same-day/future history"],
    ["Baseline", "Mean (or median) of log amount by product", "Majority class (approve all); PR-AUC = prevalence"],
    ["Primary metrics", "RMSE / MAE on log and KES scales; residual-vs-fitted", "PR-AUC, ROC-AUC, F1/precision/recall at a cost-chosen threshold, calibration"],
    ["Validation", "GroupKFold(customer) + temporal hold-out", "GroupKFold(customer) + temporal hold-out; stratified only as the 'naive' comparison"],
], [3.2, 6.9, 6.9])
P("<b>Label definition to state in Part A.</b> A loan is a default (target = 1) if it was not repaid in full according to the lender's rule by the end of the observation window following "
  "<i>due_date</i>. The data dictionary does not state the grace period; you should say so and treat it as a documented assumption (e.g. 'unpaid 30 days after due date').")

# ---------------- 3 data inventory ----------------
P("3. Data inventory and quality", H1)
table([
    ["File", "Rows × cols", "Content", "Notes"],
    ["Train.csv", f"{S['train_shape'][0]:,} × {S['train_shape'][1]}", "Kenyan loans with default label", f"{S['train_dates'][0]} → {S['train_dates'][1]}; 0 missing values"],
    ["Test.csv", f"{S['test_shape'][0]:,} × {S['test_shape'][1]}", "Applications to score (no label)", f"Kenya {S['test_dates']['Kenya'][0]}→{S['test_dates']['Kenya'][1]}; Ghana {S['test_dates']['Ghana'][0]}→{S['test_dates']['Ghana'][1]}"],
    ["economic_indicators.csv", "27 × 25", "9 World Bank/FRED indicators × Ghana, Kenya, Côte d'Ivoire, 2001–2023", f"{S['macro_missing_cells']} missing cells; nothing after 2023"],
    ["Global Findex Ghana 2021", "not in repo", "Financial-inclusion microdata (n=1,000)", "Must be downloaded separately for Part D.4"],
], [3.5, 2.4, 5.2, 5.9])
P("<b>Data-quality checks.</b> The tables are clean in the narrow sense (no nulls, no duplicate IDs, and <i>due_date − disbursement_date</i> equals <i>duration</i> on every row), "
  "but several anomalies need a documented decision:")
bullets([
    f"{dq['repay_lt_amount']} loans have Total_Amount_to_Repay &lt; Total_Amount (negative 'fee'), and {dq['repay_zero']} have a repay amount of zero.",
    f"{dq['lender_funded_zero']:,} loans ({pct(dq['lender_funded_zero'] / n, 1)}) have Amount_Funded_By_Lender = 0 – the lender has no exposure on them. "
    f"{dq['lender_portion_gt1']} loan has Lender_portion_Funded &gt; 1.",
    f"{dq['amount_lt_100']} loans are below 100 KES (minimum 2). Check whether these are test or reversal records.",
    f"{S['loan_ids_multi_rows']:,} tbl_loan_id values appear on more than one row ({S['n_loanid_train']:,} unique loan ids for {n:,} rows). "
    "The ID column is customer+loan+lender, so one loan can be co-funded by several lenders. Decide whether the modelling unit is the row or the loan, and never split one loan across folds.",
    "Amounts are in local currency (KES in Kenya, GHS in Ghana; 1 GHS ≈ 13–19 KES over 2021–23). Raw amounts cannot be compared across countries without conversion (e.g. to USD with the exchange-rate indicator) or within-country standardisation.",
])

# ---------------- 4 imbalance ----------------
P("4. Target and class imbalance", H1)
fig("01_class_imbalance.png", 12, f"Figure 1. {d:,} defaults out of {n:,} loans ({pct(S['default_rate'], 2)}), roughly 1 in 55.")
P("Implications for modelling:")
bullets([
    "Report PR-AUC first. Its no-skill value equals the prevalence (0.018 here; 0.039 in the temporal hold-out), so always print it next to the score.",
    "Prefer <b>class weights</b> (class_weight='balanced' or scale_pos_weight ≈ 53) over SMOTE. Oversampling inside CV is easy to get wrong and distorts calibration.",
    "Class weighting shifts predicted probabilities upward, so <b>recalibrate</b> (Platt or isotonic on an out-of-fold set) before applying the cost threshold "
    "p* = C<sub>FP</sub>/(C<sub>FP</sub>+C<sub>FN</sub>).",
    "With ~250 defaults per fold, F1 is noisy: bootstrap 95% CIs are mandatory and the ±0.026 leaderboard noise floor is realistic.",
])

# ---------------- 5 skew ----------------
P("5. Distributions and skew", H1)
rows = [["Column", "Median", "Mean", "Max", "Skew (raw)", "Skew (log1p)"]]
for r in S["numeric_summary"]:
    rows.append([r["col"], f"{r['50%']:,.2f}", f"{r['mean']:,.2f}", f"{r['max']:,.2f}", f"{r['skew']:.1f}", f"{r['skew_log1p']:.2f}"])
table(rows, [5.2, 2.2, 2.4, 2.8, 2.1, 2.3])
fig("02_skew_log.png", 17, "Figure 2. Top: raw distributions (log-scaled counts) are extremely right-skewed. Bottom: after log1p the amounts are near-symmetric; "
    "defaulters (orange) sit at higher amounts and much longer durations.")
P("All monetary columns have skew above 50 (Total_Amount max = 23 million KES versus a median of 5,249). "
  "<b>Use log1p for every amount</b>, both as model input (essential for logistic regression, kNN and SVM) and as the Task 1 regression target. "
  "Duration is effectively categorical (7 days = 95% of loans; 64 distinct values), so treat it as a banded categorical or use log(duration).")

# ---------------- 6 categorical ----------------
P("6. Categorical fields and segment risk", H1)
rows = [["Field", "Unique (train)", "Unique (test)", "Unseen in test", "Top level share"]]
for r in S["cardinality"]:
    rows.append([r["col"], f"{r['train_unique']:,}", f"{r['test_unique']:,}", f"{r['unseen_in_test']:,}", pct(r["top_share"], 1)])
table(rows, [4.2, 3, 3, 3, 3.8])
fig("03_default_by_segment.png", 17, "Figure 3. Default rate by product, lender, duration band and amount decile. Dashed line = overall rate.")
P(f"<b>Reading the segments.</b> Risk is concentrated, not spread out. Type_1 / lender 267278 / 7-day loans make up the low-risk core. "
  f"Long-duration products and small lenders are the high-risk tail: lender 245684 defaults {pct(S['rate_lender'][3]['mean'], 0)}, lender 267277 {pct(S['rate_lender'][2]['mean'], 0)}. "
  "Amount has a weak, non-monotonic relationship with default except in the top decile (4.6%). "
  "Several loan types have fewer than 30 loans, so encode loan_type with a rare-level bucket ('Other') or with regularised target encoding fitted inside each fold. "
  "Note that loan_type, lender_id and duration are strongly confounded (a product is a lender × tenor combination), which matters for SHAP reason codes.")
P("country_id is constant (Kenya) in training, so the model cannot learn a country effect. That is the root of the transfer problem in Section 11.")

# ---------------- 7 time ----------------
P("7. Time: drift and regime change", H1)
fig("04_temporal.png", 17, "Figure 4. Volume (top) and monthly default rate (bottom). Almost all volume falls in Jul–Nov 2022, the only period with a ~1% default rate.")
rows = [["Year", "Loans", "Default rate"]] + [[r["year"], f"{r['count']:,}", pct(r["mean"], 2)] for r in S["rate_by_year"]]
table(rows, [3, 3, 3])
P(f"The portfolio has two regimes: a large, low-risk burst (Jul–Nov 2022, almost all Type_1 loans from lender 267278) and a thin, much riskier tail of other products. "
  f"An 80/20 time split cuts on {S['temporal_cut_date']}, so the hold-out default rate is {pct(S['baseline_prauc_temporal'], 1)}, twice the training rate. "
  "Consequences: (i) a temporal hold-out is the most honest proxy for deployment; (ii) calendar features (month, day-of-month) will partly learn the regime, so check that they do not dominate; "
  "(iii) calibration fitted on 2022 will under-predict risk later. Recalibrate on the most recent period and monitor drift.")

# ---------------- 8 customers ----------------
P("8. Customers, repeat borrowing and train/test overlap", H1)
fig("05_customer_history.png", 17, "Figure 5. Left: loans per customer. Right: default rate falls steadily with the number of prior loans.")
lp = S["loans_per_customer"]
P(f"Customers borrow repeatedly: median {lp['median']:.0f} loans, 90th percentile {lp['p90']:.0f}, maximum {lp['max']}; only {pct(lp['single_loan_customers_pct'], 0)} have a single loan. "
  f"{S['customers_ever_default']:,} customers ever default, and {pct(S['defaults_from_repeat_defaulters_pct'], 0)} of all defaults come from customers with more than one default. "
  "Default behaviour is therefore <b>clustered by customer</b>. A random split puts the same customer in train and validation and lets the model memorise them.")
P("<b>Time-safe history features</b> (Part B.2). Sort by disbursement_date and use <i>pd.merge_asof(..., by='customer_id', direction='backward', allow_exact_matches=False)</i> "
  "so that only loans disbursed <i>strictly before</i> the current date are visible. Candidate features: number of prior loans, prior defaults, prior default rate, days since last loan, "
  "prior mean amount, amount relative to the customer's previous maximum, and number of distinct prior lenders. "
  f"Same-day loans are common, which is why exact matches must be excluded. With a plain cumulative sum, 'had prior default' gives a 61% default rate, but the strictly-earlier version gives "
  f"{pct(S['safe_hist']['rate_if_prior_default'], 0)}. The difference is same-day leakage.")
P(f"<b>Overlap with Test.</b> {S['overlap_customers']:,} customers ({S['test_rows_with_known_customer']:,} Kenyan test rows) and {S['overlap_loans']} loan ids appear in both files. "
  "Recommended policy: (1) for history features of test rows, use the labelled train loans that are strictly earlier; this mirrors production, where the lender knows a customer's past. "
  "(2) Never put train rows of the same loan id into a validation fold of another. (3) Report test-time performance separately for 'known' vs 'new-to-model' customers. "
  "All Ghana applicants are new to the model.")

# ---------------- 9 leakage audit ----------------
P("9. Leakage audit (feature audit table, Part A.1c)", H1)
fig_rows = [["Column", "Known at decision?", "Role", "Reason"],
            ["ID", "Yes", "Drop", "Row identifier (customer+loan+lender)."],
            ["customer_id", "Yes", "Group key / history key", "Used for GroupKFold and merge_asof; not a raw feature (identity, memorisation)."],
            ["country_id", "Yes", "Fairness attribute, macro join", "Constant in train, so it cannot be learned. Used to join macro data and slice metrics."],
            ["tbl_loan_id", "Yes", "Drop", "Sequential identifier; a proxy for time (AUC 0.58). Use the date instead."],
            ["lender_id", "Yes", "Feature (categorical)", "Strong risk driver. Four Ghana lenders are unseen, so a fallback is needed."],
            ["loan_type", "Yes", "Feature (categorical)", "Strongest product signal. Type_3/Type_8 unseen; bucket rare levels."],
            ["Total_Amount", "Yes (requested/offered)", "Feature (T2); TARGET (T1)", "log1p. Must not be a feature in Task 1."],
            ["Total_Amount_to_Repay", "Contract: yes · as recorded: likely NO", "Drop (or replace with contracted price)", f"Fee-rate AUC {S['univariate_auc']['fee_rate']:.2f}; defaulters' 'fee' ≈ 22% vs 0.8%. Looks like post-due penalties."],
            ["disbursement_date", "Yes", "Calendar features, time split, as-of key", "Closest column to the decision moment."],
            ["due_date", "Yes (contractual)", "Drop", "Redundant: exactly disbursement_date + duration."],
            ["duration", "Yes", "Feature", "Monotonic risk (1% → 45%). Band or log-transform."],
            ["New_versus_Repeat", "Yes", "Feature + fairness attribute", "New loans 21% default vs 1.7% repeat."],
            ["Amount_Funded_By_Lender", "Yes", "Feature (T2 only)", "= Lender_portion_Funded × Total_Amount, so it leaks into Task 1."],
            ["Lender_portion_Funded", "Yes", "Feature", "Lender's risk share (structure of the deal)."],
            ["Lender_portion_to_be_repaid", "Likely NO", "Drop", "Same post-outcome pattern as Total_Amount_to_Repay."],
            ["target", "No (outcome)", "TARGET (T2)", "1 = defaulted."]]
table(fig_rows, [3.9, 3.1, 3.6, 6.4], fs=7.4)
rows = [["Implied fee band", "Repaid", "Defaulted", "Default rate"]] + [[r["band"], f"{r['repaid']:,}", f"{r['default']:,}", pct(r["rate"], 1)] for r in fee_bins]
P("<b>Evidence for the repay-amount leak.</b> If the fee were a pre-agreed price, defaulters would not pay systematically higher 'fees'. Instead the default rate climbs with the fee:")
table(rows, [4, 3, 3, 3])
callout("Decision required from the team",
        "Ask the lender (or state as an assumption) whether Total_Amount_to_Repay is the contracted amount at disbursement or the amount owed at close. "
        "Unless it is confirmed as contracted, exclude it and its derivatives. If you keep it, report results <b>with and without</b> it. "
        "A model built on it will collapse in production, where the penalty part does not yet exist.", PALEOR, ORANGE)

# ---------------- 10 validation ----------------
P("10. Validation design: what the numbers really are", H1)
P("To show how much each choice changes the answer, the same two models (class-weighted logistic regression and histogram gradient boosting, no tuning) "
  "were scored under three validation schemes and four feature sets:")
fig("08_cv_demo.png", 17, "Figure 6. PR-AUC (out-of-fold) by model, feature set and validation scheme. Dashed line = no-skill PR-AUC (0.018).")
rows = [["Feature set", "Model", "Stratified PR-AUC", "GroupKFold PR-AUC", "Temporal PR-AUC", "Temporal ROC-AUC"]]
for r in S["cv_demo"]:
    rows.append([r["features"], r["model"].split(" ")[0], f"{r['stratified_prauc']:.3f}", f"{r['group_prauc']:.3f}", f"{r['temporal_prauc']:.3f}", f"{r['temporal_auc']:.3f}"])
table(rows, [4.8, 1.9, 2.6, 2.6, 2.5, 2.6])
infl = (core_h['stratified_prauc'] - core_h['group_prauc']) / core_h['group_prauc']
infl_t = (core_h['stratified_prauc'] - core_h['temporal_prauc']) / core_h['temporal_prauc']
bullets([
    f"<b>Grouping inflation.</b> For gradient boosting, stratified K-fold over-states PR-AUC by {pct(infl, 0)} relative to GroupKFold, and by {pct(infl_t, 0)} relative to the temporal split. "
    "Flexible models memorise customers; logistic regression barely changes. This is exactly the comparison Part B.3 asks you to report.",
    f"<b>Leaky features.</b> Adding repay-derived columns lifts PR-AUC from ~{core_h['group_prauc']:.2f} to ~{fee_h['group_prauc']:.2f}, and a customer target-mean that includes the current loan lifts it to ~{leak_h['group_prauc']:.2f}. "
    "Both gains are illusory.",
    f"<b>Safe history helps modestly.</b> merge_asof features raise ROC-AUC (HistGB GroupKFold {core_h['group_auc']:.3f} → {safe_h['group_auc']:.3f}) with a small PR-AUC gain. "
    "Expect tuned models to improve on this, but the realistic ceiling is far below the leaky numbers.",
    "<b>Which to trust for deployment:</b> the temporal split (train on older, test on newer, customers grouped), because deployment means scoring future loans from partly new customers. "
    "Use GroupKFold for model selection inside the training window.",
])

# ---------------- 11 shift ----------------
P("11. Kenya → Ghana: covariate and concept shift (Part C)", H1)
fig("06_kenya_ghana_shift.png", 17, "Figure 7. Kenya test looks like Kenya train. Ghana uses different products (Type_3, Type_2), different lenders, longer tenors (8% at 180 days) and a different currency.")
table([
    ["Characteristic", "Kenya (train)", "Ghana (test)"],
    ["Main loan types", "Type_1 90%, Type_7 4%", f"Type_3 {pct(gh['loan_type']['Type_3'], 0)} (unseen), Type_2 {pct(gh['loan_type']['Type_2'], 0)}, Type_8 1% (unseen)"],
    ["Lenders", "267278 94%, 251804 5%", "296542 51%, 297183 36% (all four main lenders unseen); 245684 3% (seen; 41% default in Kenya)"],
    ["Duration", "7 days 95%", "7 days 86%; 180–365 days ~13%"],
    ["Median amount", f"KES {S['kenya_train']['amount_median']:,.0f}", f"GHS {gh['amount_median']:,.0f} (≈ KES 40,000 at the 2022 rate, about 8× larger)"],
    ["Median implied fee", pct(S['kenya_train']['fee_rate_median'], 2), pct(gh['fee_rate_median'], 1)],
    ["New-loan share", pct(S['kenya_train']['new_share'], 1), pct(gh['new_share'], 1)],
    ["Customers with history in train", "n/a", f"{gh['known_customers']} (none)"],
], [3.6, 5, 8.4])
P(f"<b>Domain classifier.</b> A gradient-boosting classifier that predicts 'Ghana vs Kenya' from the loan features reaches ROC-AUC {S['domain_auc']:.4f}, and {S['domain_auc_minimal']:.3f} even with only "
  "duration, tenure, lender share and calendar fields. The two populations barely overlap. Density-ratio importance weights collapse: effective sample size is "
  f"{S['iw_ess_pct'] * 100:.4f}% of the Kenyan rows. <b>Importance weighting cannot rescue the transfer</b>, because there is no support overlap to reweight.")
P("<b>What this means for the Ghana model.</b> Prediction for Ghana is extrapolation, not interpolation. Recommended stance:")
bullets([
    "Build a model whose inputs generalise across countries: currency-free features (amount relative to the country median or in USD, log duration, lender-share, tenure, history), "
    "with product and lender mapped to <i>attributes</i> (tenor, price, lender share) rather than raw IDs that do not exist in Ghana.",
    "Report Ghana predictions with wide uncertainty: score distribution, share of applicants flagged, and comparison with Kenya at matched attributes. Do not report a Ghana accuracy, because none can be measured.",
    "Propose a <b>controlled pilot</b>: approve a random or low-limit 'exploration' sample in Ghana, collect 3–6 months of labels, then recalibrate or retrain. "
    "Until then, use conservative limits rather than a hard decline rule (see the Bank of Ghana Directive on responsible lending).",
    "Evidence needed to trust the model in Ghana: labelled Ghana outcomes; stable calibration on them; a domain classifier AUC well below 1 after feature redesign; and lender confirmation that product terms are comparable.",
])

# ---------------- 12 macro ----------------
P("12. Macro-economic indicators", H1)
fig("07_macro.png", 17, "Figure 8. Selected indicators, 2010–2023. Ghana's 2022–23 inflation (31%, 38%) and depreciation have no Kenyan analogue.")
P(f"The file has 9 indicators for Ghana, Kenya and Côte d'Ivoire up to {S['macro_last_year']} ({S['macro_missing_cells']} missing cells; precipitation ends in 2021; Ghana has no lending-rate or real-interest series). "
  "Points to note when joining by country-year:")
bullets([
    "Within Kenyan training data there are only 4 distinct years, and one (2022) holds 94% of loans. A macro feature is therefore almost a <b>year dummy</b>: it will learn the regime, not a macro effect. "
    "Expect little genuine signal, and check that it does not dominate importance.",
    "Ghana's values (inflation 31–38%) lie far outside the Kenyan training range (6–8%), so tree models will clamp and linear models will extrapolate. Either is risky; standardise within country or leave macro out of the Ghana score.",
    "2024 loans have no 2024 indicators. Use the previous year's values (lag 1). This is also the only leakage-safe choice, since annual figures are published after the year ends.",
    "Macro data is most useful for the <b>narrative</b> (Ghana's cedi depreciation raises borrower stress) and for stress-testing thresholds, not as a strong predictor.",
])

# ---------------- 13 cost matrix ----------------
P("13. A data-driven cost matrix and threshold (Part A.3)", H1)
P("Appendix A of the brief tells you to derive your own figures. The data give these anchors (medians over repaid loans, so the suspect penalty part is mostly excluded):")
ke_amt = C["median_amount_7d"]; ke_fee = ke_amt * C["median_fee_rate_7d"]
gh_amt = C["ghana_median_amount_ghs"]; gh_fee = gh_amt * C["ghana_median_fee_rate"]
def thr(fee, amt, rec, coll):
    fn = amt * (1 - rec) + coll
    return fn, fee / (fee + fn)
rows = [["Scenario", "Principal", "Fee if repaid (C_FP)", "Recovery", "Collection cost", "C_FN", "Decline if p >"]]
for lab, cur, amt, fee, coll in [("Kenya 7-day (median)", "KES", ke_amt, ke_fee, 200), ("Ghana (median, all tenors)", "GHS", gh_amt, gh_fee, 25)]:
    for rec in [0.0, 0.2, 0.4]:
        fn, t = thr(fee, amt, rec, coll)
        rows.append([lab if rec == 0.0 else "", f"{cur} {amt:,.0f}", f"{cur} {fee:,.0f}", pct(rec, 0), f"{cur} {coll}", f"{cur} {fn:,.0f}", pct(t, 2)])
table(rows, [3.6, 2.2, 2.5, 1.9, 2.4, 2.2, 2.2])
P(f"The Kenyan median 7-day fee is only {pct(C['median_fee_rate_7d'], 2)} of principal (about KES {ke_fee:,.0f}), so a single default wipes out the profit on roughly "
  f"{ke_amt * 0.8 / ke_fee:,.0f} good loans. The cost-optimal threshold of ~0.9% is <b>below the base rate</b>: a pure profit rule would decline most applicants. "
  "This is the explicit tension in the driving question. Ways to resolve it, all of which you should discuss:")
bullets([
    "Add an <b>approval-rate floor</b> (e.g. approve ≥ 90% of applicants) and choose the threshold that minimises expected loss subject to that floor.",
    "Use the probability for <b>limits and pricing</b>, not only accept/reject: lower the limit for medium-risk applicants rather than declining (Task 1 + Task 2 combined).",
    "Include the lender's true revenue (interest + platform fees, repeat-customer lifetime value) in C<sub>FP</sub>. The observed fee may understate it.",
    "Show the operating point under at least two alternative recovery and cost assumptions, as the table does, and report how approval rate and expected loss change.",
])
P("<i>Collection-cost figures (KES 200, GHS 25) are illustrative assumptions for agent follow-up, SMS and airtime. Replace them with your own sourced estimates.</i>", CAP)

# ---------------- 14 fairness ----------------
P("14. Fairness pre-audit (Part D)", H1)
P("Before any model, the raw default rates show where a risk model will concentrate declines. These are the gaps your disaggregated metrics must examine:")
table([
    ["Fairness point", "What the data show", "Risk / what to check"],
    ["Borrower tenure", f"New {pct(tenure['New Loan']['mean'], 1)} vs Repeat {pct(tenure['Repeat Loan']['mean'], 1)} default; only {tenure['New Loan']['count']} new loans", "Thin-file applicants will be declined disproportionately. Report selection rate and recall by tenure; consider a separate starter limit instead of a decline."],
    ["Country", "No Ghana labels; Ghana products and lenders unseen", "Score distribution for Ghana vs Kenya at matched attributes; selection-rate ratio; uncertainty statement."],
    ["loan_type / lender_id", "Small lenders and long-tenor products carry most defaults (9–72%)", "Is the model really scoring the lender's product design rather than the borrower? Check per-lender PR-AUC and calibration."],
    ["Loan-size band", "Default rate roughly flat except the top decile (4.6%)", "Check whether small-loan borrowers (likely poorer) face higher decline rates at the same true risk (regressivity)."],
    ["Proxies", "lender_id ↔ country; duration ↔ product; calendar ↔ regime", "Proxy analysis: how well can protected-style attributes (country, tenure) be predicted from the features?"],
], [3.2, 6.2, 7.6])
P("Gender and region are not in the competition data. Use Global Findex Ghana 2021 (download required) to describe who lacks access to credit "
  "(e.g. account ownership and borrowing by gender, income quintile, rural/urban), and argue whether a model that declines thin-file applicants widens that gap.")

# ---------------- 15 roadmap ----------------
P("15. Recommended solution roadmap", H1)
table([
    ["Step", "What to do", "Why (from this EDA)"],
    ["1. Freeze the feature set", "Use the audit in §9. Drop ID, tbl_loan_id, due_date, Total_Amount_to_Repay, Lender_portion_to_be_repaid. In Task 1 also drop Amount_Funded_By_Lender.", "Leakage dominates every other effect."],
    ["2. Preprocess", "log1p amounts; band duration; rare-level bucket for loan_type/lender; one-hot for linear models, native categorical for boosting; scaler for LR/kNN/SVM; fit everything inside a sklearn Pipeline.", "Skew &gt; 50; 22 loan types with long tail; unseen levels in test."],
    ["3. Engineer features", "merge_asof history (prior loans, prior defaults, rate, recency, amount vs past max); calendar (month, day-of-month, weekday); lag-1 macro by country-year; currency-free amount (USD or within-country z-score).", "History is the strongest legitimate signal; currencies differ."],
    ["4. Validate", "Report stratified vs GroupKFold(customer_id) vs temporal (train ≤ 2022-11, test after) side by side; pick models with GroupKFold, report final numbers on temporal.", "Stratified inflates by ~14–40%; strong regime shift."],
    ["5. Model ladder", "Majority/mean baselines → logistic regression (L2, class-weighted) → pruned decision tree → kNN → random forest → gradient boosting (LightGBM/HistGB, class-weighted) → optional RBF-SVC/SVR. Ridge/Lasso/OLS and boosting for Task 1.", "Brief requires ≥5 models; committee prefers a simple model."],
    ["6. Evaluate", "PR-AUC, ROC-AUC, threshold sweep, confusion matrix at cost threshold, reliability diagram + isotonic/Platt; bootstrap 95% CI for F1 and PR-AUC; compare against the ±0.026 noise floor.", "Few positives, so metrics are noisy."],
    ["7. Choose the model", "If boosting beats logistic regression by less than the CI width, ship logistic regression (with monotone, explainable coefficients) as the regulated scorer.", "Expected outcome stated in the brief; explainability requirement."],
    ["8. Transfer and fairness", "Domain classifier and score distributions for Ghana; disaggregated metrics and selection rates; proxy analysis; one mitigation (e.g. tenure-specific thresholds or a starter-limit policy).", "§11 and §14."],
    ["9. Ship", "joblib pipeline + score(application) → {p_default, decision, limit, top-3 reason codes}; document required inputs and measure latency.", "Part E."],
], [3.3, 8.2, 5.5], fs=7.4, keep=False)
P("<b>Suggested repository layout</b>: <i>01_eda.ipynb</i> (this analysis), <i>02_classification.ipynb</i>, <i>03_regression.ipynb</i>, <i>04_transfer_and_fairness.ipynb</i>, "
  "<i>src/features.py</i> (merge_asof history, shared by training and the scorer, so features are identical at train and serve time), <i>src/score.py</i>, <i>models/</i>, README with exact run order.")

# ---------------- 16 limitations ----------------
P("16. Limitations and reproducibility", H1)
bullets([
    "The validation-scheme experiment uses untuned models and a simple feature set. It shows the <i>direction and size</i> of the biases, not the best achievable score.",
    "The leakage diagnosis of Total_Amount_to_Repay is an inference from the data pattern. It should be confirmed with the data provider.",
    "Global Findex Ghana and the Bank of Ghana Directive are not in the repository, so Part D.3–D.4 still need to be done.",
    "Reproduce with: <i>pip install pandas numpy matplotlib scikit-learn reportlab</i>, then <i>python eda/eda_analysis.py</i> followed by <i>python eda/build_report.py</i> (random_state = 0 throughout).",
    "AI-use declaration: this EDA was prepared with the assistance of an AI coding assistant (Claude). Per the course policy, team members must check, understand and be able to defend every figure and claim.",
])


def on_page(c, doc):
    c.saveState()
    c.setFont("DV", 7.5); c.setFillColor(INK2)
    if doc.page > 1:
        c.drawString(2 * cm, 1.2 * cm, "PROSIT 2 · NkwaPay credit scoring · Exploratory Data Analysis")
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        c.setStrokeColor(RULE); c.setLineWidth(0.4); c.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
    c.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                        title="PROSIT 2 – EDA and Solution Guide", author="ICS553 MLE")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
