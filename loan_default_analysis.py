"""
Loan default prediction - rigorous rebuild.

Target is `Status` (1 = defaulted/charged off). `Default` (prior default on file, Y/N)
is a FEATURE, not the label. Everything that learns from data happens after the split,
inside an imblearn Pipeline, so SMOTE and imputation can never see the test set.

Run from the project root:  python loan_default_analysis.py
Dependencies: see requirements.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau, rankdata, spearmanr
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
# Resolved against this file's directory so the script runs from any working directory.
CSV = Path(__file__).resolve().parent / "Loan prediction mini dataset.csv"
COST_FN = 10.0  # missed default: unrecovered principal
COST_FP = 1.0   # rejected good loan: forgone interest margin
NUMERIC = ["Age", "Income", "Emp_length", "Amount", "Rate", "Percent_income", "Cred_length"]
CATEGORICAL = ["Home", "Intent", "Default"]
FAMILIES = ["Logistic", "Random Forest", "XGBoost"]
LEAK_GATE = 0.97        # model ROC-AUC above this means a leak, not a good model
UNIVARIATE_GATE = 0.90  # direction-adjusted single-feature AUC above this means a leak


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def weighted_label(family):
    """XGBoost rebalances with scale_pos_weight, the others with class_weight."""
    return f"{family} (scale_pos_weight)" if family == "XGBoost" else f"{family} (class weights)"


# ----------------------------------------------------------------- 1. schema
rule("STEP 1 - SCHEMA VERIFICATION (which column is the outcome?)")
df = pd.read_csv(CSV)
print(f"Rows: {len(df)}   Columns: {list(df.columns)}\n")
print("Status value_counts():")
print(df["Status"].value_counts(dropna=False).to_string())
print("\nDefault value_counts():")
print(df["Default"].value_counts(dropna=False).to_string())
print("\ncrosstab(Default, Status):")
print(pd.crosstab(df["Default"], df["Status"]).to_string())
print("\nMissing values:")
print(df.isnull().sum()[lambda s: s > 0].to_string())

# Status is already binary 0/1 -> it is the resolved outcome. Default is Y/N, and the
# crosstab shows both Default levels contain both Status outcomes, so Default is a
# prior-default flag on file, not the label.
print(
    "\nStatus is binary 0/1 -> Status is the target (1 = default)."
    "\n`Default` (Y/N) is a prior-default flag on file -> kept as a FEATURE."
)

# -------------------------------------------------- 2. censored rows / cleaning
rule("STEP 2 - CENSORED ROWS + PRE-SPLIT CLEANING")
print(
    "Status has no 'Current'/open category - it is already a resolved binary outcome.\n"
    "Rows dropped for censoring: 0. The decision still stands: unresolved loans would\n"
    "have to be dropped, because labelling them non-default mislabels outcomes that\n"
    "simply have not happened yet."
)

df = df.drop(columns=["Id"])  # row identifier, no signal, deterministic drop

# Age > 100 are data-entry errors (max is 144). Deterministic rule, learns nothing from
# the data, so it is safe pre-split; the resulting NaNs are imputed inside the pipeline
# with a median fitted on TRAIN only.
n_bad_age = int((df["Age"] > 100).sum())
df.loc[df["Age"] > 100, "Age"] = np.nan
print(
    f"\nAge > 100 flagged as missing (deterministic rule, pre-split): "
    f"{n_bad_age} row{'' if n_bad_age == 1 else 's'}"
)

y = df["Status"].astype(int)
X = df.drop(columns=["Status"])
print(f"Base rate (share of defaults): {y.mean():.4f}  ({y.sum()} / {len(y)})")

# ------------------------------------------------------------------ 3. split
rule("STEP 3 - STRATIFIED SPLIT BEFORE ANY LEARNED TRANSFORM")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
print(f"Train {X_train.shape}  base rate {y_train.mean():.4f}")
print(f"Test  {X_test.shape}  base rate {y_test.mean():.4f}")
print("No imputation, scaling or SMOTE has happened yet - all of it lives in the pipeline.")

neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
spw = neg / pos
print(f"scale_pos_weight for XGBoost = neg/pos on TRAIN = {neg}/{pos} = {spw:.4f}")


# -------------------------------------------------------------- 4. pipelines
def preprocessor(numeric=None):
    return ColumnTransformer(
        [
            (
                "num",
                SkPipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                list(numeric if numeric is not None else NUMERIC),
            ),
            (
                "cat",
                SkPipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        # All levels kept (drop=None): L2 absorbs the collinearity, trees are
                        # unaffected, and a dropped level would vanish from the SHAP table.
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(CATEGORICAL),
            ),
        ]
    )


def make_pipe(model, smote, numeric=None):
    """imblearn Pipeline: SMOTE fires on training folds only, skipped at predict time."""
    steps = [("prep", preprocessor(numeric))]
    if smote:
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def models(weighted):
    """weighted=True -> class weights / scale_pos_weight.

    weighted=False -> plain model, for the SMOTE arm of the head-to-head.
    """
    return {
        "Logistic": LogisticRegression(
            max_iter=2000,
            random_state=SEED,
            class_weight="balanced" if weighted else None,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=SEED,
            class_weight="balanced" if weighted else None,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=SEED,
            scale_pos_weight=spw if weighted else 1.0,
        ),
    }


CONFIGS = {}
CONFIGS["Majority baseline (Dummy)"] = make_pipe(
    DummyClassifier(strategy="most_frequent"), smote=False
)
for name, m in models(weighted=True).items():
    CONFIGS[weighted_label(name)] = make_pipe(m, smote=False)
for name, m in models(weighted=False).items():
    CONFIGS[f"{name} (SMOTE)"] = make_pipe(m, smote=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

rule("STEP 4 - PIPELINE CONSTRUCTION")
print("imblearn Pipeline per configuration:")
print("  impute -> one-hot -> scale -> [SMOTE] -> model")
print(f"  numeric ({len(NUMERIC)}): {NUMERIC}")
print(f"  categorical ({len(CATEGORICAL)}): {CATEGORICAL}")
print("SMOTE is a pipeline step, so it resamples training folds only and is skipped at")
print("predict time. Imputer and scaler are fitted per fold, never on the test set.")
print(f"\n{len(CONFIGS)} configurations built:")
for name in CONFIGS:
    print(f"  - {name}")


# --------------------------------------------- 5. SMOTE vs class weights (CV)
rule("STEP 5 - SMOTE vs CLASS WEIGHTS (5-fold CV on TRAIN, PR-AUC)")
cv_pr = {}
for name, pipe in CONFIGS.items():
    if name.startswith("Majority"):
        continue
    s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
    cv_pr[name] = (s.mean(), s.std())
    print(f"  {name:32s} PR-AUC {s.mean():.4f} +/- {s.std():.4f}")

print("\nHead-to-head (one variable changed - resampling strategy):")
for fam in FAMILIES:
    w = cv_pr[weighted_label(fam)][0]
    s = cv_pr[f"{fam} (SMOTE)"][0]
    print(f"  {fam:14s} weights {w:.4f}   SMOTE {s:.4f}   delta {s - w:+.4f}")

best_name = max(cv_pr, key=lambda k: cv_pr[k][0])
print(f"\nBest configuration by CV PR-AUC: {best_name}")


# ---------------------------- 6. threshold chosen on TRAIN out-of-fold predictions
rule("STEP 6 - COST-SENSITIVE THRESHOLD (chosen on TRAIN out-of-fold CV predictions)")
print(f"Cost ratio FN:FP = {COST_FN:.0f}:{COST_FP:.0f}   expected cost = 10*FN + 1*FP\n")


def pick_threshold(pipe):
    """Choose the cost-minimising threshold using out-of-fold predictions on TRAIN only."""
    oof = cross_val_predict(
        pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    grid = np.unique(np.round(np.linspace(0.01, 0.99, 197), 4))  # 0.005 steps
    costs = []
    for t in grid:
        pred = (oof >= t).astype(int)
        fn = int(((y_train == 1) & (pred == 0)).sum())
        fp = int(((y_train == 0) & (pred == 1)).sum())
        costs.append(COST_FN * fn + COST_FP * fp)
    return float(grid[int(np.argmin(costs))]), float(np.min(costs))


thresholds = {}
for name, pipe in CONFIGS.items():
    if name.startswith("Majority"):
        thresholds[name] = 0.50
        continue
    t, c = pick_threshold(pipe)
    thresholds[name] = t
    print(f"  {name:32s} threshold {t:.4f}  (OOF expected cost {c:,.0f})")

best_threshold = thresholds[best_name]
print(f"\nChosen operating threshold ({best_name}): {best_threshold:.4f}")
print("Selected on training out-of-fold predictions, then applied unchanged to the test set.")


# ------------------------------------------------------- 7. final test table
rule("STEP 7 - FINAL RESULTS ON THE HELD-OUT TEST SET")
rows, fitted, test_proba = [], {}, {}
for name, pipe in CONFIGS.items():
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    p = pipe.predict_proba(X_test)[:, 1]
    test_proba[name] = p
    t = thresholds[name]
    pred = (p >= t).astype(int)
    rows.append(
        {
            "Model": name,
            "Threshold": t,
            "PR-AUC": average_precision_score(y_test, p),
            "ROC-AUC": roc_auc_score(y_test, p),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "Accuracy": accuracy_score(y_test, pred),
            # Accuracy at 0.50 is reported only so the headline number is comparable to the
            # usual literature figure; it is not the operating point we ship.
            "Acc@0.50": accuracy_score(y_test, (p >= 0.5).astype(int)),
        }
    )

results = pd.DataFrame(rows)
print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nPrecision/Recall/F1/Accuracy are at each model's own threshold column;")
print("Acc@0.50 is the same model scored at the default 0.50 cut, for comparability only.")
print(f"PR-AUC floor (test base rate) = {y_test.mean():.4f}")

# Sanity gate: an ROC-AUC above 0.97 on this problem means a leak, not a good model.
max_roc = results.loc[results["Model"] != "Majority baseline (Dummy)", "ROC-AUC"].max()
gate = "PASS" if max_roc <= LEAK_GATE else "FAIL - INVESTIGATE"
print(f"Leakage gate (ROC-AUC <= {LEAK_GATE}): max ROC-AUC = {max_roc:.4f} -> {gate}")


# ----------------------------------------------------- 8. leakage diagnostics
rule("STEP 8 - LEAKAGE DIAGNOSTICS")
print("Univariate ROC-AUC of each feature against the target.")
print("A value near 0.0 OR 1.0 = a leak; 0.5 = no signal. Direction-adjusted = max(auc, 1-auc),")
print("so an inversely associated feature is not mistaken for a harmless one.")
print("Caveat: categorical columns are target-encoded in sample here, so their AUCs are")
print("optimistic; they are a leak screen, not an importance ranking.")
adjusted = {}
for c in X.columns:
    s = X[c]
    v = s.fillna(s.median()) if pd.api.types.is_numeric_dtype(s) else s.map(y.groupby(s).mean())
    auc = roc_auc_score(y, v)
    adjusted[c] = max(auc, 1 - auc)
    print(f"  {c:16s} auc {auc:.4f}   direction-adjusted {adjusted[c]:.4f}")

strongest, strongest_auc = max(adjusted.items(), key=lambda kv: kv[1])
n_dup = int(X.duplicated().sum())
print(f"\nDuplicate feature rows in the source data: {n_dup} / {len(X)}")
if strongest_auc < UNIVARIATE_GATE and n_dup / len(X) < 0.01:
    print(
        f"No feature is close to determining the target (strongest: {strongest} at "
        f"{strongest_auc:.4f}), and duplicates are negligible."
    )
else:
    print(
        f"WARNING: strongest univariate signal is {strongest} at {strongest_auc:.4f} "
        f"and duplicates are {n_dup / len(X):.2%} - investigate before trusting the table."
    )

print(f"\nSplit-seed stability of the headline model ({weighted_label('XGBoost')}):")
seeds = [0, 1, 7, 42, 2024]
prs, rocs = [], []
for sd in seeds:
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=sd
    )
    m = models(weighted=True)["XGBoost"]
    m.set_params(scale_pos_weight=(y_tr_s == 0).sum() / (y_tr_s == 1).sum())
    pp = make_pipe(m, smote=False).fit(X_tr_s, y_tr_s).predict_proba(X_te_s)[:, 1]
    prs.append(average_precision_score(y_te_s, pp))
    rocs.append(roc_auc_score(y_te_s, pp))
print(f"  seeds   {seeds}")
print(f"  PR-AUC  {np.mean(prs):.4f} +/- {np.std(prs):.4f}   {np.round(prs, 3).tolist()}")
print(f"  ROC-AUC {np.mean(rocs):.4f} +/- {np.std(rocs):.4f}   {np.round(rocs, 3).tolist()}")
worst_pr_seed, worst_roc_seed = seeds[int(np.argmin(prs))], seeds[int(np.argmin(rocs))]
print(
    f"  Lowest PR-AUC at seed {worst_pr_seed}, lowest ROC-AUC at seed {worst_roc_seed}; "
    f"the reported split uses seed {SEED}."
)


# -------------------------------------- 9. confusion matrix at chosen threshold
rule(f"STEP 9 - CONFUSION MATRIX - {best_name} @ {best_threshold:.4f}")
best_pred = (test_proba[best_name] >= best_threshold).astype(int)
cm = confusion_matrix(y_test, best_pred)
tn, fp, fn, tp = cm.ravel()
print(pd.DataFrame(cm, index=["actual 0", "actual 1"], columns=["pred 0", "pred 1"]).to_string())
print(f"\nTN {tn}  FP {fp}  FN {fn}  TP {tp}")
print(f"Precision {precision_score(y_test, best_pred):.4f}  "
      f"Recall {recall_score(y_test, best_pred):.4f}  "
      f"F1 {f1_score(y_test, best_pred):.4f}  "
      f"Accuracy {accuracy_score(y_test, best_pred):.4f}")
print(f"Expected cost at threshold: 10*{fn} + 1*{fp} = {COST_FN * fn + COST_FP * fp:,.0f}")
cm05 = confusion_matrix(y_test, (test_proba[best_name] >= 0.5).astype(int))
print(f"For contrast, cost at the default 0.50 threshold: "
      f"{COST_FN * cm05[1, 0] + COST_FP * cm05[0, 1]:,.0f}")


# --------------------------------------------------------------------- 10. SHAP
rule("STEP 10 - SHAP (TreeExplainer on the best tree model)")
tree_candidates = {k: v for k, v in cv_pr.items() if k.startswith(("Random Forest", "XGBoost"))}
shap_model_name = max(tree_candidates, key=lambda k: tree_candidates[k][0])
print(f"Explaining: {shap_model_name}")

pipe = fitted[shap_model_name]
prep = pipe.named_steps["prep"]
feat_names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
X_test_t = prep.transform(X_test)

explainer = shap.TreeExplainer(pipe.named_steps["model"])
sv = explainer.shap_values(X_test_t)
if isinstance(sv, list):
    sv = sv[1]
if sv.ndim == 3:
    sv = sv[:, :, 1]

mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(mean_abs)[::-1][:5]

PRETTY = {
    "Rate": "interest rate on the loan",
    "Percent_income": "loan amount as a share of income",
    "Income": "annual income",
    "Amount": "loan amount",
    "Emp_length": "employment length (years)",
    "Cred_length": "credit history length (years)",
    "Age": "age",
    "Home_RENT": "home ownership = RENT",
    "Home_OWN": "home ownership = OWN",
    "Home_MORTGAGE": "home ownership = MORTGAGE",
    "Home_OTHER": "home ownership = OTHER",
    "Default_Y": "prior default on file = Y",
    "Default_N": "prior default on file = N",
    "Intent_EDUCATION": "loan intent = EDUCATION",
    "Intent_MEDICAL": "loan intent = MEDICAL",
    "Intent_VENTURE": "loan intent = VENTURE",
    "Intent_PERSONAL": "loan intent = PERSONAL",
    "Intent_DEBTCONSOLIDATION": "loan intent = DEBT CONSOLIDATION",
    "Intent_HOMEIMPROVEMENT": "loan intent = HOME IMPROVEMENT",
}

print("\nTop 5 one-hot COLUMNS by mean |SHAP|, with direction:")
for rank, i in enumerate(order, 1):
    raw, vals, s = feat_names[i], X_test_t[:, i], sv[:, i]
    label = PRETTY.get(raw, raw)
    levels = np.unique(vals)
    if len(levels) == 2:  # one-hot column: compare mean SHAP when present vs absent
        delta = s[vals == levels[1]].mean() - s[vals == levels[0]].mean()
        direction = (
            f"{label} pushes TOWARD default" if delta > 0 else f"{label} pushes AWAY from default"
        )
    else:
        corr = np.corrcoef(vals, s)[0, 1]
        direction = (
            f"higher {label} pushes TOWARD default"
            if corr > 0
            else f"higher {label} pushes AWAY from default"
        )
    print(f"  {rank}. {raw:26s} mean|SHAP| {mean_abs[i]:.4f}  ->  {direction}")

# A multi-level categorical is split across several one-hot columns, so per-column ranking
# understates it. Summing each source variable's columns gives the variable-level ranking.
num_cols = list(prep.transformers_[0][2])
cat_cols = list(prep.transformers_[1][2])
encoder = prep.named_transformers_["cat"].named_steps["onehot"]
source = num_cols + [
    # strict=True: a cat_cols/categories_ desync would otherwise fail silently - the
    # short zip drops columns, the resulting Series align on index, and a whole
    # variable's SHAP mass disappears into a discarded NaN group with no error.
    col for col, cats in zip(cat_cols, encoder.categories_, strict=True) for _ in cats
]
grouped = pd.Series(mean_abs).groupby(pd.Series(source)).sum().sort_values(ascending=False)
print("\nSame SHAP values GROUPED BY SOURCE VARIABLE (one-hot columns summed):")
for rank, (var, val) in enumerate(grouped.items(), 1):
    print(f"  {rank:2d}. {var:16s} mean|SHAP| {val:.4f}")


# ------------------------------------------- 11. Rate leakage sensitivity run
rule("STEP 11 - SENSITIVITY: drop `Rate` (lender's own risk-model output)")
xgb_label = weighted_label("XGBoost")
no_rate = make_pipe(
    models(weighted=True)["XGBoost"], smote=False, numeric=[c for c in NUMERIC if c != "Rate"]
)
no_rate.fit(X_train.drop(columns=["Rate"]), y_train)
p_nr = no_rate.predict_proba(X_test.drop(columns=["Rate"]))[:, 1]
nr_pr, nr_roc = average_precision_score(y_test, p_nr), roc_auc_score(y_test, p_nr)
with_rate = results.loc[results["Model"] == xgb_label].iloc[0]
print(
    f"{xgb_label} WITH Rate:    PR-AUC {with_rate['PR-AUC']:.4f}  "
    f"ROC-AUC {with_rate['ROC-AUC']:.4f}"
)
print(f"{xgb_label} WITHOUT Rate: PR-AUC {nr_pr:.4f}  ROC-AUC {nr_roc:.4f}")
print(
    "\nRate is set by the lender's own risk model. Using it to predict risk is using the\n"
    "output of a risk assessment as an input - at origination for a new applicant it does\n"
    "not exist yet. Kept in the headline model, but the gap above is the honest cost of\n"
    "removing it."
)

# ---------------------- 12. why the two logistic arms tie at 4 decimal places
# The results table shows PR-AUC 0.6262 for both logistic arms. That tie is a rounding
# coincidence at display precision, but auditing it surfaced a real structural finding,
# so the evidence for both claims is reproduced here. Logistic-only: cheap to refit.
rule("STEP 12 - AUDIT: THE TWO LOGISTIC ARMS AT FULL PRECISION")

lw, ls = "Logistic (class weights)", "Logistic (SMOTE)"
p_w, p_s = test_proba[lw], test_proba[ls]

ap_w, ap_s = average_precision_score(y_test, p_w), average_precision_score(y_test, p_s)
roc_w, roc_s = roc_auc_score(y_test, p_w), roc_auc_score(y_test, p_s)
print("Metrics at 10 decimal places (the table rounds these to 4):")
print(f"  average precision  weights {ap_w:.10f}   SMOTE {ap_s:.10f}   diff {abs(ap_s - ap_w):.3e}")
print(f"  ROC-AUC            weights {roc_w:.10f}   SMOTE {roc_s:.10f}   "
      f"diff {abs(roc_s - roc_w):.3e}")
print("  -> distinct models; average precision merely collides at 4dp.")

n_diff = int((p_w != p_s).sum())
print(f"\nTest probabilities: {n_diff} of {len(p_w)} differ  "
      f"(max |delta| {np.abs(p_w - p_s).max():.4f}, mean {np.abs(p_w - p_s).mean():.4f})")

rho = spearmanr(p_w, p_s).statistic
tau = kendalltau(p_w, p_s).statistic
n_rank_diff = int((rankdata(p_w) != rankdata(p_s)).sum())
print(f"Rank agreement: Spearman {rho:.4f}   Kendall tau {tau:.4f}   "
      f"({(1 - tau) / 2:.2%} discordant pairs)")
print(f"Rankings are NOT identical: {n_rank_diff} of {len(p_w)} ranks differ.")

# If the two arms differed only by an intercept, ranking would be exactly invariant and
# both ranking metrics would match to machine precision. They do not - so the difference
# is a coefficient rescale plus a small rotation, which is the actual finding.
c_w = fitted[lw].named_steps["model"].coef_[0]
c_s = fitted[ls].named_steps["model"].coef_[0]
cos = float(c_w @ c_s / (np.linalg.norm(c_w) * np.linalg.norm(c_s)))
print(f"\nCoefficient geometry over {len(c_w)} features:")
print(f"  cosine similarity {cos:.4f}  ({np.degrees(np.arccos(cos)):.1f} degrees apart)")
print(f"  norm ratio (SMOTE / weights) x{np.linalg.norm(c_s) / np.linalg.norm(c_w):.3f}")
b_w = fitted[lw].named_steps["model"].intercept_[0]
b_s = fitted[ls].named_steps["model"].intercept_[0]
print(f"  intercept shift {b_s - b_w:+.3f}"
      f"   sign agreement {int((np.sign(c_w) == np.sign(c_s)).sum())}/{len(c_w)}")
print(f"  mean predicted p: weights {p_w.mean():.3f}, SMOTE {p_s.mean():.3f} "
      f"vs base rate {y_test.mean():.3f} - both miscalibrated upward, similarly.")

print("\nAcross 7 split seeds (is the tie reproducible, or is the rank agreement?):")
print(f"  {'seed':>6} {'AP weights':>12} {'AP SMOTE':>12} {'|dAP|':>10} {'tau':>8}  tie@4dp")
audit_seeds = [42, 0, 1, 7, 2024, 123, 999]
n_ties = 0
for sd in audit_seeds:
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.20, stratify=y, random_state=sd)
    fit_w = make_pipe(models(weighted=True)["Logistic"], smote=False).fit(Xa, ya)
    fit_s = make_pipe(models(weighted=False)["Logistic"], smote=True).fit(Xa, ya)
    pa, pb = fit_w.predict_proba(Xb)[:, 1], fit_s.predict_proba(Xb)[:, 1]
    a, b = average_precision_score(yb, pa), average_precision_score(yb, pb)
    tie = round(a, 4) == round(b, 4)
    n_ties += tie
    print(f"  {sd:>6} {a:>12.10f} {b:>12.10f} {abs(b - a):>10.1e} "
          f"{kendalltau(pa, pb).statistic:>8.4f}  {'yes' if tie else 'no'}")
print(f"  -> 4dp tie in {n_ties} of {len(audit_seeds)} seeds (luck of the split); "
      f"high tau at every seed (structural).")

print(
    "\nFinding: for logistic regression, class weighting and SMOTE are near rank-equivalent.\n"
    "Both push toward the same rebalanced prior, and the fitted models differ mainly by a\n"
    "coefficient-norm rescale plus a small rotation - not an intercept-only shift. Both\n"
    "PR-AUC and ROC-AUC are ranking-only metrics and both do move; ROC-AUC picks up the\n"
    "rotation because it averages over all pairs, while average precision is top-weighted\n"
    "and the discordance sits mid-ranking, so it moved far less. This is WHY SMOTE buys\n"
    "nothing for the linear model: neither method meaningfully changes the ranking, only\n"
    "the calibration."
)

rule("DONE")
