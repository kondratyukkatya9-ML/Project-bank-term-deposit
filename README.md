# Bank term deposit subscription prediction
## **Problem definition**: 
The bank runs telemarketing campaigns, outbound phone calls offering a term deposit. Each call costs operator time. The bank's telemarketing campaigns are dialled without prioritisation: every client in
the base has an equal chance of being called. Call-centre
capacity is therefore spent largely on clients who will not convert.
Expected impact: the same number of subscriptions achieved with fewer calls, or more
subscriptions within the existing calling capacity.

## **What are the key business metrics?**
- Conversion rate per call 
- Calls required per subscription
- Lift in the top decile vs random dialling

Monetary impact cannot be estimated from this dataset: it requires the cost per call and
the margin per deposit, which are not available. These figures should be supplied by the
business before deployment.

## **What are the model performance metrics?**

**PR-AUC (Average Precision) — primary.**
Summarises ranking quality across all thresholds using only the positive class.
At π = 0.1127 a random ranker scores 0.1127, so every result is read against that
baseline.

**ROC-AUC — secondary.**
Reported for comparability with published results on this dataset. Inflated by the
89/11 imbalance: the false-positive rate has ~36,500 negatives in its denominator,
so several thousand wasted calls barely move it.

**Gini (2 × ROC-AUC − 1) — secondary.**
A rescaling of ROC-AUC, included because it is the conventional metric in banking.
Shares ROC-AUC's insensitivity to imbalance.

Precision and recall are threshold-dependent and therefore not used for modelselection. They are reported for the final model once the
threshold is chosen.

**F1 — comparison at a fixed threshold.**
Used to compare models once a threshold is chosen; not used for model selection.

## **How do the model performance metrics align with business goals?**

The model is used to rank the client base; operators call from the top down until capacity runs out. That makes ranking quality the property to optimise, which is why a ranking metric PR-AUC is primary.

Lift in the top decile is the point where this becomes measurable in the call centre's own terms. PR-AUC says the ranking improved; lift says by how much the best-scored clients outperform untargeted dialling.

## **What is the data cleaning process?**


The dataset is unusually clean: no nulls, no malformed values, no type inconsistencies.
Cleaning is therefore limited to three decisions, none of which removes rows.

**Masked missing values are kept as a category.**
`unknown` in `default`, `education`, `housing`, `loan`, `job` and `marital` is a
recorded non-response, not a gap. `default = unknown` converts at ~5% versus ~12.5%
for `no`, so the non-response itself carries signal. Imputing it would destroy that.

**The `pdays` sentinel is re-encoded.**
999 means "never contacted before" rather than 999 days. Left as a number, a model
would read it as a genuine time distance. It is replaced by the binary flag
`was_contacted_before`; the original column is dropped as an exact duplicate
(correlation −1.00).

**Outliers are retained.**
`campaign` reaches 56, `previous` reaches 7, `age` reaches 98. All are plausible
records rather than data errors. The Tukey rule flags many points here only because
the distributions are right-skewed. The client who received 56 calls is one of the
most informative rows in the dataset — removing it would delete the signal about
hopeless contacts.

Skew is handled at the model level instead: trees use raw values, linear models and
kNN use `log1p` before scaling.

Duplicate rows are removed.
12 exact duplicates  were found, forming identical pairs. Call duration matches
to the second within each pair, which makes a data-loading artefact far more likely
than two distinct clients sharing a profile. 

**Target encoding.** `y` (`yes`/`no`) is converted to `y_bin` (1/0) so that a mean
over the column gives the conversion rate directly.

## **How will we prevent data leakage?**
`duration` is excluded from all models. It is recorded only after
the call ends, when the outcome is already known, so it cannot exist at scoring time.

## **What preprocessing is needed?**

Different model families need different preprocessing, so each gets its own pipeline
rather than sharing one transformed matrix.

**Trees (Decision Tree, XGBoost)** need categorical encoding and nothing else.
`OrdinalEncoder` assigns an arbitrary integer per category, which is harmless here
because trees split on thresholds. Numeric features pass through unchanged — scaling
and skew correction have no effect on threshold-based splits, and all three
macroeconomic features can be kept despite their mutual correlation.

**Linear models and kNN** need the full treatment. One-hot encoding, since ordinal
integers would imply a false ordering. `log1p` on `campaign` and `previous` to compress
the right tail. `StandardScaler` so that features on different scales contribute
comparably. Only one of `emp.var.rate`, `euribor3m` and `nr.employed` is kept — they
correlate at 0.91–0.97, which destabilises coefficients.

**Dropped from all models:** `duration` (leakage), `pdays` and `pdays_clean`
(correlations −1.00 and 0.87 with `was_contacted_before`). 19 features remain.

**Class imbalance** is handled by weighting rather than resampling:
`class_weight='balanced'` for sklearn models, `scale_pos_weight=7.88` for XGBoost.

## **What is the validation schema?**

Stratified random split 70/15/15: 28,823 / 6,176 / 6,177 rows, positive rate 0.1127 in
all three parts. Hyperparameters are selected by 5-fold StratifiedKFold CV inside the
training set; validation compares model families; test is opened once, at the end.

The reconstructed monthly timeline shows a change in call-centre behaviour rather than
client behaviour. Through May 2009 the bank dialled thousands per month at 6.7%
conversion; from June 2009 volumes collapsed to a few hundred while conversion rose to
44.5%. The first regime is 36,214 rows, the second only 4,962.

This makes a temporal split unusable: a chronological test block would consist almost
entirely of the second regime, which training would never see. That measures
extrapolation to an unseen operating mode, not ranking quality.

## **How do the two tuning methods compare?**

| Method | Best CV PR-AUC | Runtime |
|---|---|---|
| RandomizedSearchCV | 0.4675 | ~46s |
| Hyperopt (TPE) | 0.4711 | ~71s |

Same seven-parameter space, same CV scheme, 50 evaluations each. The difference is well
inside the CV standard deviation (~0.015), so the methods are equivalent here. They
converged on different parameter sets — random search picked `min_child_weight=3`,
Hyperopt picked 18 — which points to a flat objective surface rather than a sharp
optimum.

## **Which models were trained, and how do they compare?**

Four model families, plus tuning variants. All metrics on validation; test is held out.
| model                      | params                                   |   pr_auc_train |   pr_auc_val |   pr_auc_gap |   roc_auc_train |   roc_auc_val | comment                                                                                                          |
|:---------------------------|:-----------------------------------------|---------------:|-------------:|-------------:|----------------:|--------------:|:-----------------------------------------------------------------------------------------------------------------|
| XGBoost + Hyperopt         | lr=0.021, depth=8, n_est=353, mcw=18     |         0.5745 |       0.4735 |       0.101  |          0.8863 |        0.8023 | Best on validation. Selected as final model.                                                                     |
| XGBoost + RandomizedSearch | lr=0.013, depth=7, n_est=252, gamma=3.03 |         0.5568 |       0.4724 |       0.0845 |          0.8579 |        0.8034 | Equivalent to Hyperopt within CV noise (0.4675 vs 0.4711). Smaller gap.                                          |
| Logistic Regression        | balanced, L2, C=1.0                      |         0.4389 |       0.4374 |       0.0015 |          0.7902 |        0.7918 | Near-zero gap. Strong for its simplicity — one-hot recovers the non-monotonic age effect through job categories. |
| Decision Tree              | max_depth=6, balanced                    |         0.4528 |       0.4318 |       0.021  |          0.7916 |        0.7853 | Interpretable baseline. Depth beyond 6 degrades validation.                                                      |
| kNN                        | k=25, euclidean                          |         0.4763 |       0.4305 |       0.0458 |          0.8481 |        0.773  | On par with the tree despite no class weighting. Slow at inference.                                              |
| XGBoost, no macro          | same params, macro dropped               |         0.5115 |       0.4254 |       0.0861 |          0.8519 |        0.7718 | Diagnostic, not a candidate. Isolates client-level signal: −10% PR-AUC.                                          |
| XGBoost                    | defaults, spw=7.88                       |         0.7519 |       0.4174 |       0.3345 |          0.9421 |        0.7606 | Defaults overfit badly (gap 0.34) and lose to a depth-6 tree. Shows what tuning is for.                          |





