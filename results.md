# Results

All numbers below are printed by `loan_default_analysis.py`. The split, the models, and the
threshold search all use `random_state=42`; the only place other seeds appear is the split-seed
stability check at the end, which deliberately varies the split (0, 1, 7, 42, 2024). Test set is
1,629 rows held out before any imputation, scaling, or resampling.

## Setup

| | |
|---|---|
| Target | `Status` (1 = default) |
| `Default` (Y/N) | **feature** — prior default on file, not the label |
| Rows | 8,145 (0 dropped for censoring — `Status` is already resolved binary, no `Current` category) |
| Base rate | 0.2182 (1,777 / 8,145) |
| Split | stratified 80/20, `random_state=42` → train 6,516 / test 1,629 |
| `Age > 100` | 1 row, set to NaN pre-split by deterministic rule, median-imputed inside the pipeline |
| Missing | `Rate` 762, `Emp_length` 236 — imputed inside the pipeline, train-fold statistics only |
| `scale_pos_weight` | 5094 / 1422 = 3.5823 (computed on train) |

## Test-set results

Each model uses its own cost-minimising threshold, selected on **training out-of-fold CV
predictions** and then applied unchanged to test. The baseline is at 0.50.
`Precision`, `Recall`, `F1` and `Accuracy` are all at that model's own threshold; `Acc@0.50` is
the same model scored at the default 0.50 cut, added only for comparability with the usual
literature figure — it is not the operating point.

| Model | Threshold | PR-AUC | ROC-AUC | Precision | Recall | F1 | Accuracy | Acc@0.50 |
|---|---|---|---|---|---|---|---|---|
| Majority baseline (Dummy) | 0.500 | 0.2179 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.7821 | 0.7821 |
| Logistic (class weights) | 0.235 | 0.6262 | 0.8207 | 0.3238 | 0.8901 | 0.4748 | 0.5709 | 0.7551 |
| Random Forest (class weights) | 0.135 | 0.8032 | 0.8750 | 0.3796 | 0.8704 | 0.5287 | 0.6618 | 0.9042 |
| **XGBoost (scale_pos_weight)** | **0.195** | **0.8270** | **0.8987** | **0.4213** | **0.8901** | **0.5719** | **0.7096** | **0.8883** |
| Logistic (SMOTE) | 0.225 | 0.6262 | 0.8221 | 0.3305 | 0.8901 | 0.4821 | 0.5832 | 0.7594 |
| Random Forest (SMOTE) | 0.190 | 0.8015 | 0.8777 | 0.4228 | 0.8479 | 0.5642 | 0.7145 | 0.9012 |
| XGBoost (SMOTE) | 0.140 | 0.8216 | 0.8952 | 0.4331 | 0.8761 | 0.5797 | 0.7231 | 0.9024 |

PR-AUC floor (test base rate) = **0.2179**. Every model clears it; the majority baseline sits
exactly on it and has zero recall, which is the point of including it — 78% accuracy while
catching no defaults at all.

## SMOTE vs class weights

5-fold stratified CV on the training set, scored by PR-AUC (average precision). One variable
changed: SMOTE step with an unweighted model, vs no SMOTE with class weights /
`scale_pos_weight`.

| Model family | Class weights / `scale_pos_weight` | SMOTE | Delta (SMOTE − weights) |
|---|---|---|---|
| Logistic | 0.6617 ± 0.0134 | 0.6620 ± 0.0125 | +0.0004 |
| Random Forest | 0.8443 ± 0.0173 | 0.8378 ± 0.0199 | −0.0065 |
| XGBoost | **0.8660 ± 0.0168** | 0.8569 ± 0.0198 | −0.0091 |

All three deltas are inside ±0.02 and every one is smaller than its own CV standard deviation.
SMOTE buys nothing here. Class weights are cheaper, deterministic, and leave the probabilities
on a scale you can threshold, so the headline model uses them.

Best configuration by CV PR-AUC: **XGBoost (scale_pos_weight)**.

### Why both logistic arms show PR-AUC 0.6262

The two logistic rows in the test table carry the same PR-AUC to four decimals. That tie
prompted an audit (reproduced by STEP 12 of the script), and it turned out to be two separate
things: a display coincidence, and a real finding underneath it.

**The tie is a rounding coincidence.** The arms are independently fitted pipelines. All 1,629
test probabilities differ (max |Δ| 0.1457, mean 0.0171), and the metrics differ once you stop
rounding:

| | Class weights | SMOTE | Difference |
|---|---|---|---|
| Average precision | 0.6261804336 | 0.6261860054 | 5.6e−06 |
| ROC-AUC | 0.8207022354 | 0.8221084750 | 1.4e−03 |

**The real finding: for logistic regression, class weighting and SMOTE are near
rank-equivalent.** Both push the fit toward the same rebalanced prior. The two fitted models
differ mainly by a coefficient-norm rescale (×1.139) plus a small rotation — cosine similarity
0.9896, i.e. 8.3 degrees apart, with 17 of 19 coefficient signs agreeing.

Crucially this is **not** an intercept-only shift: the intercept moves just −0.107. That
distinction matters, because a pure intercept shift would leave the ranking exactly invariant and
both ranking metrics would match to machine precision. Instead the ranking does move — Spearman
0.9972, Kendall tau 0.9602 (1.99% discordant pairs), and 1,588 of 1,629 ranks differ.

**Both PR-AUC and ROC-AUC are ranking-only metrics, and both do move.** ROC-AUC picks up the
rotation (1.4e−03) because it averages over all pairs; average precision is top-weighted and the
~2% discordance sits mid-ranking, so it moved roughly 250× less here. Neither metric sees the
calibration difference at all, though there is one: mean predicted probability 0.390 (weights) vs
0.379 (SMOTE), both badly above the 0.218 base rate.

Across 7 split seeds, the rank agreement is structural and the tie is not:

| Seed | AP (weights) | AP (SMOTE) | \|ΔAP\| | Kendall tau | Tie at 4dp |
|---|---|---|---|---|---|
| 42 | 0.6261804336 | 0.6261860054 | 5.6e−06 | 0.9602 | yes |
| 0 | 0.6501781695 | 0.6475494129 | 2.6e−03 | 0.9691 | no |
| 1 | 0.6515959701 | 0.6524799596 | 8.8e−04 | 0.9622 | no |
| 7 | 0.6838910003 | 0.6836006849 | 2.9e−04 | 0.9708 | no |
| 2024 | 0.6247108839 | 0.6247369447 | 2.6e−05 | 0.9770 | yes |
| 123 | 0.6370980163 | 0.6357089527 | 1.4e−03 | 0.9676 | no |
| 999 | 0.6518017141 | 0.6510025182 | 8.0e−04 | 0.9685 | no |

Tau stays in 0.96–0.98 at every seed; the 4dp tie appears in only 2 of 7. **This is why SMOTE
buys nothing for the linear model — neither method meaningfully changes the ranking, and only
the calibration differs.**

## Cost-sensitive threshold

**Stated cost ratio: FN:FP = 10:1.** A missed default costs unrecovered principal; a false
positive costs the interest margin on a good loan that was declined. Expected cost = 10·FN + 1·FP.

The threshold was chosen on **training out-of-fold CV predictions**, never on test, then applied
to the held-out test set unchanged.

| | |
|---|---|
| Chosen threshold | **0.1950** |
| Selected on | 5-fold out-of-fold predictions on train (OOF expected cost 2,838) |
| Test expected cost at 0.195 | 10·39 + 1·434 = **824** |
| Test expected cost at 0.50 | **1,100** |

Moving off the default 0.50 threshold cuts expected cost by 25%.

Confusion matrix, XGBoost (scale_pos_weight) @ 0.1950, test set:

| | pred 0 | pred 1 |
|---|---|---|
| **actual 0** | TN 840 | FP 434 |
| **actual 1** | FN 39 | TP 316 |

Precision 0.4213 · Recall 0.8901 · F1 0.5719 · Accuracy 0.7096

At 10:1 this is the correct trade: catch 316 of 355 defaults at the price of declining 434 good
loans. Accuracy drops from 0.8883 (`Acc@0.50` in the table above) to 0.7096 and that is the
intended outcome, not a regression — accuracy is the wrong objective when one error costs ten
times the other.

## SHAP findings

`TreeExplainer` on XGBoost (scale_pos_weight), computed on the transformed test set, with one-hot
column names mapped back to readable features.

| # | Column | mean\|SHAP\| | Direction |
|---|---|---|---|
| 1 | `Rate` | 0.8902 | Higher interest rate on the loan pushes **toward** default |
| 2 | `Income` | 0.7604 | Higher annual income pushes **away from** default |
| 3 | `Percent_income` | 0.7130 | A larger loan as a share of income pushes **toward** default |
| 4 | `Home_RENT` | 0.3248 | Renting pushes **toward** default |
| 5 | `Home_OWN` | 0.2968 | Outright ownership pushes **away from** default |

**Per-column ranking is not the variable-level ranking.** A multi-level categorical is split
across several one-hot columns, so ranking columns gives `Home` two of the five slots and buries
`Intent`, which is actually the model's second-strongest variable once its six columns are summed.
Summing each source variable's columns gives the true order — Rate, Intent, Income,
Percent_income, Home:

| # | Source variable | mean\|SHAP\| (summed) |
|---|---|---|
| 1 | `Rate` | 0.8902 |
| 2 | `Intent` | 0.8027 |
| 3 | `Income` | 0.7604 |
| 4 | `Percent_income` | 0.7130 |
| 5 | `Home` | 0.6827 |
| 6 | `Amount` | 0.1960 |
| 7 | `Emp_length` | 0.1689 |
| 8 | `Default` | 0.1681 |
| 9 | `Age` | 0.1226 |
| 10 | `Cred_length` | 0.0684 |

All five directions are the economically expected sign, which is a weak but real check that the
model has not latched onto an artifact. The conclusion about `Default` is unchanged either way
(0.168 on both rankings, 8th of 10): the prior-default flag the original project used as its
*target* carries far less signal than debt-service burden, the priced rate, or loan purpose.

## Rate as target leakage

`Rate` is the strongest feature in the model, and it is **set by the lender's own risk model**.
Using it to predict default is using the output of a risk assessment as an input to a risk
assessment. At origination, for a genuinely new applicant, the rate does not exist yet.

Sensitivity run, XGBoost (scale_pos_weight), `Rate` dropped entirely:

| | PR-AUC | ROC-AUC |
|---|---|---|
| With `Rate` | 0.8270 | 0.8987 |
| Without `Rate` | 0.7446 | 0.8521 |

Dropping it costs 0.082 PR-AUC. The no-Rate figure is the one to quote for a pre-origination
underwriting model; the with-Rate figure is valid only for scoring loans that have already been
priced (portfolio monitoring, secondary-market pricing).

## Sanity checks

| Check | Expected | Observed | Verdict |
|---|---|---|---|
| Base rate | ~0.20 | 0.2182 | pass |
| PR-AUC floor = base rate | ~0.20 | 0.2179, dummy sits on it | pass |
| Gradient boosting PR-AUC | 0.72–0.82 | 0.8270 | +0.007 over band — see below |
| Gradient boosting ROC-AUC | 0.92–0.95 | 0.8987 | −0.021 under band — see below |
| SMOTE vs weights within 0.02 PR-AUC | ±0.02 | max \|delta\| 0.0091 | pass |
| Accuracy | 0.90–0.93 | 0.9042 (RF), 0.8883 (XGB) @ 0.50 | pass / marginal |
| Leakage tripwire ROC-AUC > 0.97 | must not trip | max 0.8987 | pass, comfortably |

The two band misses were investigated rather than accepted:

- **No feature approaches determining the target.** Univariate ROC-AUC is reported
  direction-adjusted as `max(auc, 1 - auc)`, because an AUC below 0.5 means an *inverse*
  association of that strength, not weakness — `Income` scores a raw 0.3198, which is really a
  0.6802-strength signal, and an inverted leak would print near 0.0 and look harmless. Adjusted
  values: `Age` 0.5238, `Income` 0.6802, `Home` 0.6536, `Emp_length` 0.5731, `Intent` 0.5793,
  `Amount` 0.5592, `Rate` 0.7034, `Percent_income` 0.7143, `Default` 0.5756, `Cred_length` 0.5159.
  Nothing above 0.72. (Categorical columns are target-encoded in sample here, so their values are
  optimistic — this is a leak screen, not an importance ranking.)
- **Duplicate rows: 8 of 8,145** (0.1%), present in the source CSV, not manufactured by SMOTE —
  SMOTE runs inside the imblearn pipeline on training folds only.
- **The deviation is a split-seed artifact.** Across 5 split seeds, XGBoost (scale_pos_weight)
  gives PR-AUC 0.8570 ± 0.0181 `[0.859, 0.870, 0.880, 0.827, 0.850]` and ROC-AUC 0.9195 ± 0.0134
  `[0.922, 0.927, 0.938, 0.899, 0.912]`. Seed 42 — the one reported — produces the lowest PR-AUC
  *and* the lowest ROC-AUC of the five. The mean ROC-AUC of 0.920 lands inside the expected band.
- **The two misses point in opposite directions** (PR-AUC high, ROC-AUC low). Leakage inflates
  both together and drives ROC-AUC toward 0.97+. It did not. This is a smaller, `loan_grade`-free
  subsample of a familiar credit-risk dataset behaving slightly differently from the full version,
  not a leak.
