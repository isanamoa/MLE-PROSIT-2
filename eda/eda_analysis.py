"""
PROSIT 2 - Exploratory Data Analysis (African Credit Scoring / NkwaPay)
Produces figures in eda/figures and a stats dictionary in eda/eda_stats.json.
Run from repo root:  python eda/eda_analysis.py
"""
import json, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "eda", "figures")
os.makedirs(FIG, exist_ok=True)

# ---------- style ----------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=170)
    plt.close(fig)


S = {}  # stats collected for the report

# ---------- load ----------
tr = pd.read_csv(os.path.join(ROOT, "Train.csv"), parse_dates=["disbursement_date", "due_date"])
te = pd.read_csv(os.path.join(ROOT, "Test.csv"), parse_dates=["disbursement_date", "due_date"])
ec = pd.read_csv(os.path.join(ROOT, "economic_indicators.csv"))

S["train_shape"] = list(tr.shape)
S["test_shape"] = list(te.shape)
S["n_default"] = int(tr.target.sum())
S["default_rate"] = float(tr.target.mean())
S["missing_train"] = int(tr.isna().sum().sum())
S["missing_test"] = int(te.isna().sum().sum())
S["dup_rows_train"] = int(tr.drop(columns="ID").duplicated().sum())
S["dup_ids_train"] = int(tr.ID.duplicated().sum())
S["test_by_country"] = te.country_id.value_counts().to_dict()
S["train_dates"] = [str(tr.disbursement_date.min().date()), str(tr.disbursement_date.max().date())]
S["test_dates"] = {c: [str(g.disbursement_date.min().date()), str(g.disbursement_date.max().date())]
                   for c, g in te.groupby("country_id")}

# ---------- derived columns ----------
for d in (tr, te):
    d["fee"] = d.Total_Amount_to_Repay - d.Total_Amount
    d["fee_rate"] = np.where(d.Total_Amount > 0, d.Total_Amount_to_Repay / d.Total_Amount - 1, np.nan)
    d["days_to_due"] = (d.due_date - d.disbursement_date).dt.days
    d["log_amount"] = np.log1p(d.Total_Amount)
    d["month"] = d.disbursement_date.dt.to_period("M")
    d["year"] = d.disbursement_date.dt.year
    d["dow"] = d.disbursement_date.dt.dayofweek
    d["dom"] = d.disbursement_date.dt.day
    d["is_new"] = (d.New_versus_Repeat == "New Loan").astype(int)

# ---------- data quality ----------
S["dq"] = {
    "repay_lt_amount": int((tr.Total_Amount_to_Repay < tr.Total_Amount).sum()),
    "repay_zero": int((tr.Total_Amount_to_Repay == 0).sum()),
    "lender_portion_gt1": int((tr.Lender_portion_Funded > 1).sum()),
    "lender_funded_zero": int((tr.Amount_Funded_By_Lender == 0).sum()),
    "duration_mismatch": int((tr.days_to_due != tr.duration).sum()),
    "duration_mismatch_pct": float((tr.days_to_due != tr.duration).mean()),
    "amount_lt_100": int((tr.Total_Amount < 100).sum()),
    "fee_rate_median": float(tr.fee_rate.median()),
    "fee_rate_p90": float(tr.fee_rate.quantile(.9)),
    "fee_rate_mean_default": float(tr.loc[tr.target == 1, "fee_rate"].median()),
    "fee_rate_mean_repaid": float(tr.loc[tr.target == 0, "fee_rate"].median()),
    "repay_zero_default_rate": float(tr.loc[tr.Total_Amount_to_Repay == 0, "target"].mean())
    if (tr.Total_Amount_to_Repay == 0).any() else None,
}
# numeric summary + skew
num_cols = ["Total_Amount", "Total_Amount_to_Repay", "duration", "Amount_Funded_By_Lender",
            "Lender_portion_Funded", "Lender_portion_to_be_repaid"]
desc = tr[num_cols].describe(percentiles=[.5, .99]).T
desc["skew"] = tr[num_cols].skew()
desc["skew_log1p"] = np.log1p(tr[num_cols]).skew()
S["numeric_summary"] = desc.round(3).reset_index().rename(columns={"index": "col"}).to_dict("records")

# ---------- cardinality ----------
card = []
for c in ["customer_id", "tbl_loan_id", "lender_id", "loan_type", "country_id", "New_versus_Repeat", "duration"]:
    card.append({"col": c, "train_unique": int(tr[c].nunique()), "test_unique": int(te[c].nunique()),
                 "unseen_in_test": int(len(set(te[c]) - set(tr[c]))),
                 "top_share": float(tr[c].value_counts(normalize=True).iloc[0])})
S["cardinality"] = card
S["unseen_loan_types"] = sorted(set(te.loan_type) - set(tr.loan_type))
S["unseen_lenders"] = sorted(map(int, set(te.lender_id) - set(tr.lender_id)))

# ---------- overlap & grouping ----------
cust_tr, cust_te = set(tr.customer_id), set(te.customer_id)
S["overlap_customers"] = len(cust_tr & cust_te)
S["overlap_loans"] = len(set(tr.tbl_loan_id) & set(te.tbl_loan_id))
S["n_customers_train"] = int(tr.customer_id.nunique())
S["n_loanid_train"] = int(tr.tbl_loan_id.nunique())
lpc = tr.groupby("customer_id").size()
S["loans_per_customer"] = {"median": float(lpc.median()), "mean": float(lpc.mean()),
                           "p90": float(lpc.quantile(.9)), "max": int(lpc.max()),
                           "single_loan_customers_pct": float((lpc == 1).mean())}
rows_per_loan = tr.groupby("tbl_loan_id").size()
S["loan_ids_multi_rows"] = int((rows_per_loan > 1).sum())
cust_def = tr.groupby("customer_id").target.agg(["sum", "count"])
S["customers_ever_default"] = int((cust_def["sum"] > 0).sum())
S["defaults_from_repeat_defaulters_pct"] = float(cust_def.loc[cust_def["sum"] > 1, "sum"].sum() / tr.target.sum())
# share of defaulter loans whose customer has another default
S["mean_default_rate_within_defaulters"] = float((cust_def["sum"] / cust_def["count"])[cust_def["sum"] > 0].mean())
ov = te[te.customer_id.isin(cust_tr)]
S["test_rows_with_known_customer"] = int(len(ov))
S["test_rows_with_known_customer_by_country"] = ov.country_id.value_counts().to_dict()

# ---------- FIG 1: class imbalance ----------
fig, ax = plt.subplots(figsize=(5.2, 2.2))
vc = tr.target.value_counts().sort_index()
ax.barh(["Repaid (0)", "Defaulted (1)"], vc.values, color=[BLUE, ORANGE], height=0.55)
for i, v in enumerate(vc.values):
    ax.text(v + 600, i, f"{v:,}  ({v / len(tr):.2%})", va="center", color=INK, fontsize=9)
ax.set_xlim(0, vc.max() * 1.3); ax.set_xlabel("Number of loans"); ax.grid(axis="y", visible=False)
ax.set_title("Target distribution in Train.csv (Kenya)")
save(fig, "01_class_imbalance.png")

# ---------- FIG 2: skew raw vs log ----------
fig, axes = plt.subplots(2, 3, figsize=(9, 4.6))
for j, c in enumerate(["Total_Amount", "Amount_Funded_By_Lender", "duration"]):
    axes[0, j].hist(tr[c], bins=60, color=BLUE); axes[0, j].set_title(f"{c} (raw)", fontsize=8.5)
    axes[0, j].set_yscale("log")
    axes[1, j].hist(np.log1p(tr.loc[tr.target == 0, c]), bins=60, color=BLUE, alpha=.65, density=True, label="Repaid")
    axes[1, j].hist(np.log1p(tr.loc[tr.target == 1, c]), bins=60, color=ORANGE, alpha=.65, density=True, label="Defaulted")
    axes[1, j].set_title(f"log1p({c.replace('Amount_Funded_By_Lender', 'Lender_amt')})", fontsize=8.5)
axes[1, 0].legend()
fig.tight_layout()
save(fig, "02_skew_log.png")


# ---------- FIG 3: default rate by categorical ----------
def rate_table(df, col, min_n=1):
    g = df.groupby(col).target.agg(["mean", "count", "sum"]).sort_values("count", ascending=False)
    return g[g["count"] >= min_n]


lt = rate_table(tr, "loan_type")
ld = rate_table(tr, "lender_id")
nr = rate_table(tr, "New_versus_Repeat")
tr["dur_band"] = pd.cut(tr.duration, [0, 7, 14, 30, 60, 90, 2000], labels=["<=7", "8-14", "15-30", "31-60", "61-90", ">90"])
dbd = tr.groupby("dur_band", observed=True).target.agg(["mean", "count", "sum"])
tr["amt_band"] = pd.qcut(tr.Total_Amount, 10, duplicates="drop")
abd = tr.groupby("amt_band", observed=True).target.agg(["mean", "count", "sum"])
S["rate_loan_type"] = lt.reset_index().round(4).to_dict("records")
S["rate_lender"] = ld.reset_index().round(4).to_dict("records")
S["rate_tenure"] = nr.reset_index().round(4).to_dict("records")
S["rate_duration_band"] = dbd.reset_index().astype({"dur_band": str}).round(4).to_dict("records")
S["rate_amount_decile"] = abd.reset_index().astype({"amt_band": str}).round(4).to_dict("records")

fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))
base = tr.target.mean()
t = lt[lt["count"] >= 30].sort_values("mean")
ax = axes[0, 0]; ax.barh(t.index, t["mean"], color=BLUE, height=.6)
for i, (m, n) in enumerate(zip(t["mean"], t["count"])):
    ax.text(m + .002, i, f"n={n:,}", va="center", fontsize=7, color=INK2)
ax.axvline(base, color=INK2, ls="--", lw=1); ax.set_title("Default rate by loan_type (n>=30)")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0)); ax.grid(axis="y", visible=False)
ax = axes[0, 1]; t = ld.sort_values("mean")
ax.barh(t.index.astype(str), t["mean"], color=AQUA, height=.5)
for i, (m, n) in enumerate(zip(t["mean"], t["count"])):
    ax.text(m + .002, i, f"n={n:,}", va="center", fontsize=7, color=INK2)
ax.axvline(base, color=INK2, ls="--", lw=1); ax.set_title("Default rate by lender_id")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0)); ax.grid(axis="y", visible=False)
ax = axes[1, 0]
ax.bar(dbd.index.astype(str), dbd["mean"], color=VIOLET, width=.6)
for i, (m, n) in enumerate(zip(dbd["mean"], dbd["count"])):
    ax.text(i, m + .003, f"n={n:,}", ha="center", fontsize=7, color=INK2)
ax.axhline(base, color=INK2, ls="--", lw=1); ax.set_title("Default rate by loan duration (days)")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0)); ax.grid(axis="x", visible=False)
ax = axes[1, 1]
ax.bar(range(len(abd)), abd["mean"], color=ORANGE, width=.6)
ax.set_xticks(range(len(abd))); ax.set_xticklabels([f"D{i + 1}" for i in range(len(abd))])
ax.axhline(base, color=INK2, ls="--", lw=1); ax.set_title("Default rate by Total_Amount decile (D1=smallest)")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0)); ax.grid(axis="x", visible=False)
fig.text(0.5, -0.01, "Dashed line = overall default rate (1.83%)", ha="center", color=INK2, fontsize=8)
fig.tight_layout()
save(fig, "03_default_by_segment.png")

# ---------- FIG 4: temporal ----------
m = tr.groupby("month").target.agg(["mean", "count"])
S["monthly"] = {"min_rate": float(m["mean"].min()), "max_rate": float(m["mean"].max()),
                "peak_month": str(m["count"].idxmax()), "n_months": int(len(m))}
yr = tr.groupby("year").target.agg(["mean", "count"])
S["rate_by_year"] = yr.reset_index().round(4).to_dict("records")
fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
x = m.index.to_timestamp()
axes[0].bar(x, m["count"], width=20, color=BLUE, label="Train (Kenya)")
tm = te.groupby(["month", "country_id"]).size().unstack(fill_value=0)
axes[0].plot(tm.index.to_timestamp(), tm.get("Kenya", 0), color=AQUA, lw=2, label="Test Kenya")
axes[0].plot(tm.index.to_timestamp(), tm.get("Ghana", 0), color=ORANGE, lw=2, label="Test Ghana")
axes[0].set_title("Monthly loan volume"); axes[0].legend(ncol=3, loc="upper left")
axes[1].plot(x, m["mean"], color=ORANGE, lw=2, marker="o", ms=3)
axes[1].axhline(base, color=INK2, ls="--", lw=1)
axes[1].set_title("Monthly default rate (Train)")
axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
fig.tight_layout()
save(fig, "04_temporal.png")

# calendar
dw = tr.groupby("dow").target.mean()
S["dow_rate_range"] = [float(dw.min()), float(dw.max())]

# ---------- FIG 5: customer history ----------
trs = tr.sort_values(["customer_id", "disbursement_date", "tbl_loan_id"]).copy()
trs["prior_loans"] = trs.groupby("customer_id").cumcount()
trs["prior_defaults"] = trs.groupby("customer_id").target.cumsum() - trs.target
hb = trs.assign(pl_band=pd.cut(trs.prior_loans, [-1, 0, 2, 5, 10, 25, 10000],
                               labels=["0", "1-2", "3-5", "6-10", "11-25", ">25"]))
hbt = hb.groupby("pl_band", observed=True).target.agg(["mean", "count"])
pdft = trs.assign(had=np.where(trs.prior_defaults > 0, "Had prior default", "No prior default")) \
    .groupby("had").target.agg(["mean", "count"])
S["rate_prior_loans"] = hbt.reset_index().astype({"pl_band": str}).round(4).to_dict("records")
S["rate_prior_default"] = pdft.reset_index().round(4).to_dict("records")
fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
axes[0].hist(lpc.clip(upper=60), bins=60, color=BLUE); axes[0].set_yscale("log")
axes[0].set_title("Loans per customer in Train (clipped at 60)"); axes[0].set_xlabel("loans")
axes[1].bar(hbt.index.astype(str), hbt["mean"], color=VIOLET, width=.6)
for i, (mm, n) in enumerate(zip(hbt["mean"], hbt["count"])):
    axes[1].text(i, mm + .002, f"n={n:,}", ha="center", fontsize=7, color=INK2)
axes[1].set_title("Default rate by # prior loans (time-ordered)")
axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0)); axes[1].grid(axis="x", visible=False)
fig.tight_layout()
save(fig, "05_customer_history.png")

# ---------- FIG 6: Kenya vs Ghana shift ----------
ke = pd.concat([tr.assign(src="Kenya train"), te[te.country_id == "Kenya"].assign(src="Kenya test")])
gh = te[te.country_id == "Ghana"].assign(src="Ghana test")
allx = pd.concat([ke, gh])
S["ghana"] = {
    "loan_type": gh.loan_type.value_counts(normalize=True).round(4).head(8).to_dict(),
    "lender": {str(k): v for k, v in gh.lender_id.value_counts(normalize=True).round(4).to_dict().items()},
    "duration": {str(k): v for k, v in gh.duration.value_counts(normalize=True).round(4).head(8).to_dict().items()},
    "amount_median": float(gh.Total_Amount.median()), "amount_mean": float(gh.Total_Amount.mean()),
    "fee_rate_median": float(gh.fee_rate.median()), "new_share": float(gh.is_new.mean()),
    "known_customers": int(gh.customer_id.isin(cust_tr).sum()),
}
S["kenya_test"] = {"amount_median": float(te[te.country_id == "Kenya"].Total_Amount.median()),
                   "new_share": float(te[te.country_id == "Kenya"].is_new.mean())}
S["kenya_train"] = {"amount_median": float(tr.Total_Amount.median()), "new_share": float(tr.is_new.mean()),
                    "fee_rate_median": float(tr.fee_rate.median())}
fig, axes = plt.subplots(2, 2, figsize=(9.2, 6))
srcs = [("Kenya train", BLUE), ("Kenya test", AQUA), ("Ghana test", ORANGE)]
ax = axes[0, 0]
cats = allx.loan_type.value_counts().head(8).index
w = .27
for k, (s, col) in enumerate(srcs):
    v = allx[allx.src == s].loan_type.value_counts(normalize=True).reindex(cats).fillna(0)
    ax.bar(np.arange(len(cats)) + (k - 1) * w, v.values, width=w, color=col, label=s)
ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=35, fontsize=7)
ax.set_title("loan_type share"); ax.legend(fontsize=7); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
ax = axes[0, 1]
cats = allx.lender_id.value_counts().index
for k, (s, col) in enumerate(srcs):
    v = allx[allx.src == s].lender_id.value_counts(normalize=True).reindex(cats).fillna(0)
    ax.bar(np.arange(len(cats)) + (k - 1) * w, v.values, width=w, color=col, label=s)
ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats.astype(str), rotation=35, fontsize=7)
ax.set_title("lender_id share (4 Ghana lenders unseen in Train)"); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
ax = axes[1, 0]
for s, col in srcs:
    ax.hist(np.log10(allx.loc[allx.src == s, "Total_Amount"].clip(lower=1)), bins=50, density=True,
            histtype="step", lw=2, color=col, label=s)
ax.set_title("log10(Total_Amount), local currency"); ax.set_xlabel("log10 amount (KES vs GHS - not comparable)")
ax = axes[1, 1]
db = ["<=7", "8-14", "15-30", "31-90", ">90"]
for k, (s, col) in enumerate(srcs):
    v = pd.cut(allx.loc[allx.src == s, "duration"], [0, 7, 14, 30, 90, 5000], labels=db).value_counts(normalize=True).reindex(db)
    ax.bar(np.arange(len(db)) + (k - 1) * w, v.values, width=w, color=col)
ax.set_xticks(range(len(db))); ax.set_xticklabels(db); ax.set_title("duration band share (days)")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
fig.tight_layout()
save(fig, "06_kenya_ghana_shift.png")

# ---------- domain classifier ----------
dfe = pd.concat([tr.assign(dom_y=0), gh.assign(dom_y=1)])
feat_dom = pd.DataFrame({
    "log_amount": dfe.log_amount, "duration": dfe.duration, "fee_rate": dfe.fee_rate.fillna(0),
    "lender_portion": dfe.Lender_portion_Funded, "is_new": dfe.is_new,
    "loan_type": dfe.loan_type.astype("category").cat.codes, "dow": dfe.dow, "dom": dfe.dom,
})
skf = StratifiedKFold(5, shuffle=True, random_state=0)
aucs = []
oof_dom = np.zeros(len(dfe))
for a, b in skf.split(feat_dom, dfe.dom_y):
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(feat_dom.iloc[a], dfe.dom_y.iloc[a])
    oof_dom[b] = clf.predict_proba(feat_dom.iloc[b])[:, 1]
    aucs.append(roc_auc_score(dfe.dom_y.iloc[b], oof_dom[b]))
S["domain_auc"] = float(np.mean(aucs))
# without amount (currency effect)
aucs2 = []
fd2 = feat_dom.drop(columns=["log_amount"])
for a, b in skf.split(fd2, dfe.dom_y):
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(fd2.iloc[a], dfe.dom_y.iloc[a])
    aucs2.append(roc_auc_score(dfe.dom_y.iloc[b], clf.predict_proba(fd2.iloc[b])[:, 1]))
S["domain_auc_no_amount"] = float(np.mean(aucs2))
aucs3 = []
fd3 = feat_dom[["duration", "is_new", "dow", "dom", "lender_portion"]]
for a, b in skf.split(fd3, dfe.dom_y):
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(fd3.iloc[a], dfe.dom_y.iloc[a])
    aucs3.append(roc_auc_score(dfe.dom_y.iloc[b], clf.predict_proba(fd3.iloc[b])[:, 1]))
S["domain_auc_minimal"] = float(np.mean(aucs3))
S["ghana_rows_unseen_type"] = float((~gh.loan_type.isin(set(tr.loan_type))).mean())
S["ghana_rows_unseen_lender"] = float((~gh.lender_id.isin(set(tr.lender_id))).mean())
S["ghana_rows_unseen_type_or_lender"] = float(((~gh.loan_type.isin(set(tr.loan_type))) | (~gh.lender_id.isin(set(tr.lender_id)))).mean())
fr_bins = pd.crosstab(pd.cut(tr.fee_rate, [-1.01, 0, 0.01, 0.05, 0.1, 0.2, 0.5, 100],
                             labels=["<=0", "0-1%", "1-5%", "5-10%", "10-20%", "20-50%", ">50%"]), tr.target)
S["fee_rate_bins"] = [{"band": str(i), "repaid": int(r[0]), "default": int(r[1]), "rate": float(r[1] / (r[0] + r[1]))} for i, r in fr_bins.iterrows()]
# importance weights for Kenya rows
p = np.clip(oof_dom[:len(tr)], 1e-4, 1 - 1e-4)
wts = p / (1 - p) * (len(tr) / len(gh))
S["iw_ess_pct"] = float((wts.sum() ** 2 / (wts ** 2).sum()) / len(tr))
S["iw_share_weight_top1pct"] = float(np.sort(wts)[::-1][:len(wts) // 100].sum() / wts.sum())

# ---------- macro ----------
long = ec.melt(id_vars=["Country", "Indicator"], var_name="yr", value_name="v")
long["yr"] = long.yr.str[2:].astype(int)
S["macro_indicators"] = list(ec.Indicator.unique())
S["macro_missing_cells"] = int(ec.iloc[:, 2:].isna().sum().sum())
mac = long[long.Country.isin(["Kenya", "Ghana"]) & (long.yr >= 2019)].pivot_table(index=["Indicator", "Country"], columns="yr", values="v")
S["macro_recent"] = {f"{i} | {c}": {str(k): (None if pd.isna(v) else round(float(v), 2)) for k, v in r.items()}
                     for (i, c), r in mac.iterrows()}
S["macro_last_year"] = int(long.dropna().yr.max())
key = ["Inflation, consumer prices (annual %)", "Official exchange rate (LCU per US$, period average)",
       "Lending interest rate (%)", "Unemployment rate"]
fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))
for ax, k in zip(axes, key):
    for c, col in [("Kenya", BLUE), ("Ghana", ORANGE)]:
        s = long[(long.Indicator == k) & (long.Country == c) & (long.yr >= 2010)]
        ax.plot(s.yr, s.v, color=col, lw=2, label=c)
    ax.axvspan(2021.75, 2023.99, color=GRID, alpha=.6)
    ax.set_title(k.split("(")[0].strip()[:28], fontsize=8.5)
axes[0].legend(fontsize=7)
fig.text(0.5, -0.04, "Shaded = loan period covered by the data (2021-2024). Indicators end in 2023.", ha="center", fontsize=8, color=INK2)
fig.tight_layout()
save(fig, "07_macro.png")

# ---------- leakage probe: univariate AUC ----------
uni = {}
for c in ["Total_Amount", "Total_Amount_to_Repay", "duration", "Amount_Funded_By_Lender", "Lender_portion_Funded",
          "Lender_portion_to_be_repaid", "fee_rate", "is_new", "dow", "dom", "lender_id", "customer_id", "tbl_loan_id"]:
    v = tr[c].fillna(0)
    a = roc_auc_score(tr.target, v)
    uni[c] = round(float(max(a, 1 - a)), 4)
S["univariate_auc"] = uni
S["corr_amount_repay"] = float(np.corrcoef(tr.log_amount, np.log1p(tr.Total_Amount_to_Repay))[0, 1])
S["corr_funded_tobe"] = float(np.corrcoef(np.log1p(tr.Amount_Funded_By_Lender), np.log1p(tr.Lender_portion_to_be_repaid))[0, 1])

# ---------- validation-scheme demo ----------
X_fee = pd.DataFrame({
    "log_repay": np.log1p(tr.Total_Amount_to_Repay), "fee_rate": tr.fee_rate.fillna(0),
    "log_lender_repay": np.log1p(tr.Lender_portion_to_be_repaid)})
X = pd.DataFrame({
    "log_amount": tr.log_amount, "duration": tr.duration, "lender_portion": tr.Lender_portion_Funded,
    "log_lender_amt": np.log1p(tr.Amount_Funded_By_Lender), "is_new": tr.is_new,
    "loan_type": tr.loan_type.astype("category").cat.codes, "lender": tr.lender_id.astype("category").cat.codes,
    "dow": tr.dow, "dom": tr.dom, "month": tr.disbursement_date.dt.month,
})
# naive (leaky) customer target-encoding: mean target of all the customer's loans INCLUDING the current one
X_leaky = X.assign(cust_def_rate_leaky=tr.groupby("customer_id").target.transform("mean"),
                   cust_n_loans=tr.groupby("customer_id").target.transform("count"))
# safe history via backward merge_asof (strictly earlier loans)
trs2 = tr[["ID", "customer_id", "disbursement_date", "target"]].sort_values("disbursement_date").copy()
trs2["one"] = 1
hist = trs2.sort_values(["customer_id", "disbursement_date"]).copy()
hist["cum_loans"] = hist.groupby("customer_id").one.cumsum()
hist["cum_def"] = hist.groupby("customer_id").target.cumsum()
hist["last_date"] = hist.disbursement_date
# collapse to one row per customer-date (end of day state) so same-day loans are not seen
hist_d = hist.groupby(["customer_id", "disbursement_date"]).agg(cum_loans=("cum_loans", "max"), cum_def=("cum_def", "max"),
                                                               last_date=("last_date", "max")).reset_index()
hist_d = hist_d.sort_values("disbursement_date")
merged = pd.merge_asof(trs2, hist_d, on="disbursement_date", by="customer_id", direction="backward", allow_exact_matches=False)
merged["cum_loans"] = merged.cum_loans.fillna(0)
merged["cum_def"] = merged.cum_def.fillna(0)
merged["past_def_rate"] = np.where(merged.cum_loans > 0, merged.cum_def / merged.cum_loans.replace(0, 1), -1)
merged["days_since_last"] = (merged.disbursement_date - merged.last_date).dt.days.fillna(-1)
merged = merged.set_index("ID").loc[tr.ID]
X_safe = X.assign(prior_loans=merged.cum_loans.values, prior_def_rate=merged.past_def_rate.values,
                  prior_defaults=merged.cum_def.values, days_since_last=merged.days_since_last.values)
S["safe_hist"] = {"rows_with_prior": float((merged.cum_loans > 0).mean()),
                  "rate_if_prior_default": float(tr.target[(merged.cum_def > 0).values].mean()),
                  "rate_if_no_prior_default": float(tr.target[(merged.cum_def == 0).values].mean())}

y = tr.target.values
groups = tr.customer_id.values
order = np.argsort(tr.disbursement_date.values, kind="stable")
cut = int(len(tr) * 0.8)
S["temporal_cut_date"] = str(tr.disbursement_date.values[order][cut])[:10]


def hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=31,
                                          class_weight="balanced", random_state=0)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def cv_scores(Xm, make, scheme):
    if scheme == "temporal":
        a, b = order[:cut], order[cut:]
        m_ = make().fit(Xm.iloc[a], y[a]); pp = m_.predict_proba(Xm.iloc[b])[:, 1]
        return average_precision_score(y[b], pp), roc_auc_score(y[b], pp)
    split = (StratifiedKFold(5, shuffle=True, random_state=0).split(Xm, y) if scheme == "stratified"
             else GroupKFold(5).split(Xm, y, groups))
    oof = np.zeros(len(y))
    for a, b in split:
        oof[b] = make().fit(Xm.iloc[a], y[a]).predict_proba(Xm.iloc[b])[:, 1]
    return average_precision_score(y, oof), roc_auc_score(y, oof)


res = []
for fname, Xm in [("Core application features", X), ("+ repay-derived (suspect)", pd.concat([X, X_fee], axis=1)),
                  ("+ leaky customer target-mean", X_leaky), ("+ safe history (merge_asof)", X_safe)]:
    for mname, mk in [("LogReg (balanced)", lr), ("HistGB (balanced)", hgb)]:
        row = {"features": fname, "model": mname}
        for sc in ["stratified", "group", "temporal"]:
            ap, auc = cv_scores(Xm, mk, sc)
            row[f"{sc}_prauc"] = round(float(ap), 4); row[f"{sc}_auc"] = round(float(auc), 4)
        res.append(row)
        print(row)
S["cv_demo"] = res
S["baseline_prauc"] = float(y.mean())
S["baseline_prauc_temporal"] = float(y[order[cut:]].mean())

fig, ax = plt.subplots(figsize=(9.5, 3.4))
short = {"Core application features": "Core", "+ repay-derived (suspect)": "+Repay-derived",
         "+ leaky customer target-mean": "+Leaky cust. mean", "+ safe history (merge_asof)": "+Safe history"}
labels = [f"{r['model'].split(' ')[0]}\n{short[r['features']]}" for r in res]
xx = np.arange(len(res)); w = .26
for k, (sc, col, nm) in enumerate([("stratified", BLUE, "Stratified K-fold"), ("group", AQUA, "GroupKFold (customer)"),
                                    ("temporal", ORANGE, "Temporal 80/20")]):
    vals = [r[f"{sc}_prauc"] for r in res]
    ax.bar(xx + (k - 1) * w, vals, width=w, color=col, label=nm)
ax.axhline(y.mean(), color=INK2, ls="--", lw=1, label="No-skill PR-AUC")
ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=6.8); ax.set_ylabel("PR-AUC"); ax.legend(ncol=4, fontsize=7.5, loc="upper left")
ax.set_title("Validation scheme matters: PR-AUC by feature set x model x CV scheme"); ax.grid(axis="x", visible=False)
ax.set_ylim(0, 1.0)
save(fig, "08_cv_demo.png")

# ---------- cost matrix from data ----------
med_amt = float(tr.Total_Amount.median()); med_fee = float(tr.fee_rate.median())
mean_fee_amt = float((tr.Total_Amount_to_Repay - tr.Total_Amount).clip(lower=0).median())
S["cost"] = {"median_amount_kes": med_amt, "median_fee_rate": med_fee, "median_fee_kes": mean_fee_amt,
             "median_amount_7d": float(tr.loc[tr.duration == 7, "Total_Amount"].median()),
             "median_fee_rate_7d": float(tr.loc[tr.duration == 7, "fee_rate"].median()),
             "median_fee_rate_30d": float(tr.loc[tr.duration == 30, "fee_rate"].median()),
             "default_amount_median": float(tr.loc[tr.target == 1, "Total_Amount"].median()),
             "ghana_median_amount_ghs": float(gh.Total_Amount.median()),
             "ghana_median_fee_rate": float(gh.fee_rate.median())}

json.dump(S, open(os.path.join(ROOT, "eda", "eda_stats.json"), "w"), indent=1, default=str)
print(json.dumps({k: S[k] for k in ["domain_auc_minimal", "ghana_rows_unseen_type_or_lender", "fee_rate_bins", "rate_tenure", "rate_prior_default", "loans_per_customer", "loan_ids_multi_rows", "test_rows_with_known_customer_by_country", "rate_by_year", "temporal_cut_date", "monthly", "unseen_loan_types", "unseen_lenders", "domain_auc", "domain_auc_no_amount", "iw_ess_pct", "cost", "safe_hist", "univariate_auc"]}, indent=1, default=str))
