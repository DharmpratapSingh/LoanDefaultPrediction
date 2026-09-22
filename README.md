# Loan Default Prediction

Predicts whether a loan will default, from 8,145 originated loans with borrower and loan
attributes (age, income, home ownership, employment length, loan intent, amount, rate, loan-to-income
ratio, prior default on file, credit history length).

Run from the project root:

```
pip install -r requirements.txt
python loan_default_analysis.py
```

Dependencies (verified working set, Python 3.11): numpy 2.3.5, pandas 3.0.2, scikit-learn 1.8.0,
imbalanced-learn 0.14.1, xgboost 3.2.0, shap 0.51.0 — pinned in
[`requirements.txt`](requirements.txt).

Everything is in one script: `loan_default_analysis.py`. Full numbers in
[`results.md`](results.md).

## What it does

- **Target is `Status`** (1 = default, base rate 21.8%). `Default` (Y/N) is a *feature* — it is a
  prior-default flag on the applicant's file, not the outcome being predicted.
- **The split comes first.** Stratified 80/20 at `random_state=42`. Nothing that learns a
  statistic from the data — imputation, scaling, SMOTE — happens before that line.
- **One `imblearn.pipeline.Pipeline`** per model: impute → one-hot → scale → (SMOTE) → model, with
  a `ColumnTransformer` separating numeric from categorical. Using imblearn's Pipeline rather than
  sklearn's is the whole point: SMOTE fires on training folds only and is skipped automatically at
  predict time.
- **Baselines are rows in the table**, not an afterthought: a majority-class `DummyClassifier` and
  a balanced logistic regression.
- **Three model families, not six**: logistic regression, random forest, and XGBoost — each run in
  two configurations (class weights / `scale_pos_weight` vs SMOTE), which is why the table below
  has seven rows for three families plus the baseline.

## Results

`Precision`, `Recall`, `F1` and `Accuracy` are at each model's own operating threshold (the
`Threshold` column); `Acc@0.50` is the same model scored at the default 0.50 cut, shown only for
comparability.

| Model | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 | Accuracy | Acc@0.50 |
|---|---|---|---|---|---|---|---|---|
| Majority baseline (Dummy) | 0.500 | 0.2179 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.7821 | 0.7821 |
| Logistic (class weights) | 0.235 | 0.6262 | 0.8207 | 0.3238 | 0.8901 | 0.4748 | 0.5709 | 0.7551 |
| Random Forest (class weights) | 0.135 | 0.8032 | 0.8750 | 0.3796 | 0.8704 | 0.5287 | 0.6618 | 0.9042 |
| **XGBoost (scale_pos_weight)** | **0.195** | **0.8270** | **0.8987** | **0.4213** | **0.8901** | **0.5719** | **0.7096** | **0.8883** |
| Logistic (SMOTE) | 0.225 | 0.6262 | 0.8221 | 0.3305 | 0.8901 | 0.4821 | 0.5832 | 0.7594 |
| Random Forest (SMOTE) | 0.190 | 0.8015 | 0.8777 | 0.4228 | 0.8479 | 0.5642 | 0.7145 | 0.9012 |
| XGBoost (SMOTE) | 0.140 | 0.8216 | 0.8952 | 0.4331 | 0.8761 | 0.5797 | 0.7231 | 0.9024 |

PR-AUC floor is the base rate, 0.2179. The majority baseline gets **78% accuracy while catching
zero defaults** — which is exactly why accuracy is reported last and never led with.

**SMOTE did not beat class weights.** Head-to-head on 5-fold CV PR-AUC over the training set,
with only the resampling strategy changed: Logistic +0.0004, Random Forest −0.0065, XGBoost
−0.0091. Every delta is inside its own CV standard deviation. Class weights win on cost and
determinism.

### Why both logistic arms show PR-AUC 0.6262

The identical 4dp value in the two logistic rows prompted an audit (STEP 12 of the script). It is
a rounding coincidence — the arms are independently fitted, all 1,629 test probabilities differ,
and the metrics separate once unrounded: average precision 0.6261804336 vs 0.6261860054 (5.6e−06
apart), ROC-AUC 0.8207 vs 0.8221 (1.4e−03 apart).

The audit surfaced something more useful, though. **For logistic regression, class weighting and
SMOTE are near rank-equivalent.** Both push toward the same rebalanced prior, and the fitted
models differ mainly by a coefficient-norm rescale (×1.139) plus a small 8.3-degree rotation
(cosine similarity 0.9896) — *not* an intercept-only shift; the intercept moves just −0.107. That
distinction is the point: a pure intercept shift would leave ranking exactly invariant and both
metrics would match to machine precision. Instead ranking does move slightly (Spearman 0.9972,
Kendall tau 0.9602, 1.99% discordant pairs, 1,588 of 1,629 ranks differing).

Both PR-AUC and ROC-AUC are ranking-only metrics and both do move. ROC-AUC catches the rotation
because it averages over all pairs; average precision is top-weighted and the discordance sits
mid-ranking, so it moved ~250× less. Across 7 split seeds the tau stays at 0.96–0.98 (structural)
while the 4dp tie shows up in only 2 of 7 (luck of the split). **This is why SMOTE buys nothing
for the linear model: neither method meaningfully changes the ranking, only the calibration** —
mean predicted probability 0.390 vs 0.379, both well above the 0.218 base rate. Full table in
[`results.md`](results.md).

## The threshold decision

**Cost ratio FN:FP = 10:1.** A missed default costs unrecovered principal; a false positive costs
only the interest margin on a good loan you declined. Expected cost = 10·FN + 1·FP.

The threshold was swept on **training out-of-fold CV predictions**, not on test, then applied to
the held-out test set unchanged. Chosen: **0.195**.

| | pred 0 | pred 1 |
|---|---|---|
| **actual 0** | TN 840 | FP 434 |
| **actual 1** | FN 39 | TP 316 |

Precision 0.421 · Recall 0.890 · Expected cost **824** at 0.195, versus **1,100** at the default
0.50 — a 25% reduction. Accuracy falls from 0.8883 (the `Acc@0.50` cell above) to 0.7096 in the
process. That is the trade working as designed, not a regression: at 10:1, buying 316 caught
defaults for 434 declined good loans is correct.

## `Rate` is target leakage in an underwriting context

`Rate` is the single strongest feature (top of the SHAP ranking, mean|SHAP| 0.89). It is also
**set by the lender's own risk model**. Predicting default from it means feeding the output of a
risk assessment back in as an input — and at origination, for a genuinely new applicant, the rate
does not exist yet.

It is kept in the headline model, with the caveat stated. The sensitivity run without it:

| | PR-AUC | ROC-AUC |
|---|---|---|
| With `Rate` | 0.8270 | 0.8987 |
| **Without `Rate`** | **0.7446** | **0.8521** |

Quote the no-Rate row for a pre-origination underwriting model. The with-Rate row is valid only
for loans that have already been priced — portfolio monitoring, secondary-market pricing.

## Top SHAP findings

Ranked by one-hot column, with direction:

1. **`Rate`** (0.890) — higher interest rate pushes toward default
2. **`Income`** (0.760) — higher income pushes away from default
3. **`Percent_income`** (0.713) — larger loan as a share of income pushes toward default
4. **`Home_RENT`** (0.325) — renting pushes toward default
5. **`Home_OWN`** (0.297) — outright ownership pushes away from default

All five carry the economically expected sign. Per-column ranking is not the variable-level
ranking, though: a multi-level categorical is spread across several one-hot columns, so `Home`
takes two of the five slots while `Intent` is buried despite being the model's second-strongest
variable once its columns are summed. Grouped by source variable the order is **Rate (0.890),
Intent (0.803), Income (0.760), Percent_income (0.713), Home (0.683)**. Either way the conclusion
about `Default` holds — the prior-default flag the original version used as its *target* scores
0.168, eighth of ten.

## What the original version got wrong

The original (`project_code.py`, `LoanDefaultPrediction.ipynb`) reported ROC-AUC up to 0.92 and
accuracy up to 0.86 across six models, and essentially none of it was measuring what it claimed.
**It predicted the wrong column**: it took `Default` — a Y/N flag recording whether the applicant
had a *prior* default on file — as the target, converted it to 1/0, and then left `Status`, the
actual loan outcome, sitting in the feature matrix (`X = dataset_one_hot.drop(['Id','Default'], axis=1)`).
So the real outcome was a predictor of a proxy. On top of that, the entire preprocessing chain ran
**before** `train_test_split`: `Emp_length`/`Rate` were imputed on the full dataset, `MinMaxScaler`
was fit on the full dataset, SMOTE was applied to the *whole* dataset, and PCA was fit on the
SMOTE-resampled whole dataset — only then was the data split. That ordering leaks in two
compounding ways: synthetic minority rows interpolated from training neighbours end up in the test
set, and the test set was balanced 50/50, so every reported accuracy/precision/recall/F1 is against
a fabricated class distribution rather than the real 22% base rate. Evaluation was 0.5-threshold
accuracy and ROC-AUC only, with no PR-AUC, no baseline to beat, and no cost reasoning. The
neural-net cells passed the test set in as `validation_data`. The model-comparison bar charts were
built from hard-coded literal metric lists, the second of them (In[31]) still carrying the comment
`# Update these based on your results` on every metric. And SHAP was run on `X_test_pca` — the
explanations were of four anonymous principal components, so they could not name a feature at all.
Six models (logistic, lasso, random forest, XGBoost, SVM, neural net, plus a stacking ensemble)
were run on this foundation; breadth substituted for rigour.

**What changed:** correct target (`Status`) with `Default` demoted to a feature; a single
stratified split before anything is learned from the data; every transform moved inside an
imblearn pipeline so SMOTE touches training folds only; PCA dropped so SHAP can name real
features; three model families instead of six models, each with an explicit imbalance strategy;
PR-AUC as the headline metric against a stated floor, with a majority-class baseline in the table;
a threshold chosen against a stated 10:1 cost ratio on out-of-fold training predictions; and every
number in this README printed by a script that runs end to end. The result is a lower-looking
ROC-AUC (0.899 vs the original's claimed 0.92) that actually means something.

## Sanity checks

Base rate, PR-AUC floor, SMOTE-vs-weights gap, and the ROC-AUC > 0.97 leakage tripwire all pass
(max ROC-AUC 0.8987, comfortably clear). Two expectation bands were missed by small margins —
PR-AUC 0.827 against 0.72–0.82, ROC-AUC 0.899 against 0.92–0.95 — and were investigated rather
than accepted. No feature exceeds a direction-adjusted univariate ROC-AUC of 0.72 against the
target (adjusted as `max(auc, 1 - auc)`, so `Income`'s raw 0.3198 is correctly read as a
0.6802-strength inverse signal rather than as weakness); only 8 of 8,145 rows are duplicates; and
across 5 split seeds the model gives PR-AUC 0.857 ± 0.018 and ROC-AUC 0.920 ± 0.013, with seed 42
producing the lowest of both. The two misses point in *opposite* directions, which is the opposite
of a leakage signature. Details in [`results.md`](results.md).
