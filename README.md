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

The model is used to rank the client base; operators call from the top down untilcapacity runs out. That makes ranking quality the property to optimise, which is why a ranking metric PR-AUC is primary.

Lift in the top decile is the point where this becomes measurable in the callcentre's own terms. PR-AUC says the ranking improved; lift says by how much thebest-scored clients outperform untargeted dialling.

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






