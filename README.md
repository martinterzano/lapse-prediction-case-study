# Early Lapse Prediction: Case Study

> Weekly LightGBM classifier that scores an insurance portfolio for early policy
> lapse within a 6-month horizon. Production system on Google Cloud: training and
> validation in Vertex AI Workbench, weekly batch scoring orchestrated by Cloud
> Composer, results landed in BigQuery for downstream retention actions. In
> production, uninterrupted since deployment.

**Sector:** insurance · **Role:** end-to-end (data, model, production, monitoring) · **Stack:** LightGBM · Python · SQL · GCP (BigQuery, Vertex AI, Cloud Composer, Cloud Storage, Looker) · **Status:** in production, weekly cadence

*Writeup prepared September 2026; the implementation is client property.*

The client owns the business metrics and internal identifiers. This case study
describes how the problem was framed, how the model was built, and how the system
reasons, without publishing client data, feature names, or exact performance
numbers. Every metric below is expressed as a range or order of magnitude, and
predictive features are described by concept and group rather than by column name.

---

## Contents

1. [The problem](#1-the-problem)
2. [Why a 6-month prediction window](#2-why-a-6-month-prediction-window)
3. [Iteration journey](#3-iteration-journey)
4. [Feature engineering: how I think about signal](#4-feature-engineering-how-i-think-about-signal)
5. [Model choice: LightGBM, and why not the alternatives](#5-model-choice-lightgbm-and-why-not-the-alternatives)
6. [Probability calibration: isotonic over Platt](#6-probability-calibration-isotonic-over-platt)
7. [Explainability with SHAP: a "why" per customer](#7-explainability-with-shap-a-why-per-customer)
8. [Results and decile-based segmentation](#8-results-and-decile-based-segmentation)
9. [Validation strategy](#9-validation-strategy)
10. [External benchmark: Vertex AI AutoML](#10-external-benchmark-vertex-ai-automl)
11. [Monitoring, drift and retraining triggers](#11-monitoring-drift-and-retraining-triggers)
12. [Model card: scope, assumptions and limitations](#12-model-card-scope-assumptions-and-limitations)
13. [Scoping decisions and repository contents](#13-scoping-decisions-and-repository-contents)

---

## 1. The problem

An insurance business measures retention through **lapse rate**, the share of
policies that stop paying premiums and eventually cancel. Early lapses, those
happening in the first months after conversion, are the most damaging. The
acquisition cost has already been spent, and the policy has not been active long
enough to reach profitability.

Insurance products also have a **grace period**, a window after a missed payment
during which the policy is still technically active but administratively flagged
as at risk. By the time a policy is in grace period, the retention team is already
in reactive mode and the client is often already mentally out.

The retention team needed a way to rank active policies by their probability of
lapsing in the next 6 months *before* the grace period, so they could target
proactive retention actions at the top of the risk distribution instead of
reacting to policies already in trouble.

Constraints going in:

- Data volume in the millions of active policies. The pipeline had to be batch-oriented, not real-time.
- Retention actions run on a weekly rhythm. The model had to score weekly to stay useful.
- The team owned the actions. The model had to be a ranking tool with an explanation, and calibration mattered for prioritisation and for setting realistic team expectations, not for automatic firing of a workflow.

## 2. Why a 6-month prediction window

The prediction horizon is a business decision, not a technical one. Four reasons
converged on 6 months.

1. **Data availability for a robust label.** A shorter window (say 1 or 2 months) produces a sparse positive class. The prevalence in a monthly window is around 0.5%, which makes training unstable and evaluation noisy. Six months brings the prevalence up to a workable range (~3% in OOT) without diluting the "early lapse" concept.
2. **Retention team capacity.** The team can absorb a certain volume of proactive contacts per month. A 6-month horizon lets the model surface enough at-risk policies to fill the team's contact capacity for the whole quarter without over- or under-generating leads.
3. **Coverage of both recently converted and mature policies.** Very short windows bias the model toward policies close to their conversion date, where risk is naturally concentrated. Very long windows blur the "early lapse" concept. Six months covers the recently-converted cohort at its highest-volatility phase and still catches mature policies that have started deteriorating.
4. **Actionability.** A 6-month horizon leaves the retention team enough runway to try several actions (call, offer, communication cadence) and observe results before the policy actually lapses. Shorter windows leave no time for a second attempt.

Alternatives considered and dropped:

- **Variable-horizon labels** (*"did this lapse in the next N months, N ∈ {1, 3, 6, 12}"*) produce richer signal but are harder to communicate to the retention team, harder to calibrate, and harder to monitor. Rejected on operational grounds.
- **Survival models** (Cox, DeepSurv) are technically appropriate for time-to-event data, but the team's action is triggered by "is this policy risky in the next window?", not by an expected time-to-lapse curve. The additional complexity was not going to change the operational output.

## 3. Iteration journey

Seven iterations. Each one traded off leakage risk, signal richness, and
interpretability. The progression is more informative than the final result.

| Iter | Goal | AUC | Comment |
|---|---|---|---|
| 1 | Full model, everything on the table | ~1.00 | **Leakage-heavy.** Perfect performance flags a trap. The model was detecting policies already deep in the lapse process, not future risk. |
| 2 | Cut back highly correlated features and the 12m payment window | ~0.94 | Partial fix. Still leaking through subtler payment variables. |
| 3 | Clean baseline with 7 static features | ~0.86 | Realistic and interpretable. Answered "how much signal is really there without behavioural features?" |
| 4 | Add 6-month payment behaviour | ~0.93 | Signal jump from behavioural window features. |
| 5–6 | Refine behavioural and owner-level features | *(intermediate)* | Iteration on features and label windows. |
| 7 | Final baseline: leakage-audited, relative ratios, owner data, riders | ~0.90 | Candidate for production. Pilot deployed. |

Two decisions during the journey drove the final shape.

### 3.1 Discovering conceptual leakage

The initial leakage suspicion was purely on temporal grounds, features constructed
from data close to or after the snapshot date. After fixing that, a small group of
variables were still producing suspiciously high performance. They described the
policy's most recent payment status: how many days since the last failed payment,
whether that last payment cleared, how long until the next scheduled payment.

They were temporally clean, computed strictly from data before the snapshot, but
conceptually too close to `lapse_flag`. A policy that failed its last payment 3
days ago is essentially already in the lapse process, not at risk of lapsing in 6
months. The model was being asked to predict something that had, in effect,
already started.

Removing them dropped performance but changed what the model was actually
predicting. It moved from "which policies are currently entering the grace period"
(useful only for reactive teams) to "which policies will lapse in the next 6
months" (useful for proactive retention). This is the case study's most important
design decision. The version that scored best in cross-validation was not the
version that solved the problem.

### 3.2 From absolute to relative variables

Iterations 1 to 3 used absolute payment counts (total positive payments, total
failed payments, total months covered). These are systematically biased by policy
maturity. A policy converted 5 years ago has more of everything than one converted
6 months ago. The model was learning "old policies have more payments", and doing
so at the expense of the behavioural signal.

Iteration 4 onward replaced absolute counters with relative ratios and windowed
metrics: coverage per month elapsed, failure rate over positive attempts, and
recent-window versions of both. The general recipe was to divide behavioural
counts by an exposure metric (elapsed maturity or attempts made), or to constrain
them to a fixed recent window that is comparable across all policies regardless of
age.

The model now captures behavioural patterns rather than stock volumes. A
6-month-old policy with a 20% failure rate is correctly flagged as riskier than a
5-year-old policy with a 5% failure rate, even though the older one has more
failures in absolute terms.

## 4. Feature engineering: how I think about signal

The final feature set has around 20 features across 8 groups. The full list and
the exact selection are client property. What I can share is how I chose the
groups and what I was looking for in each.

### Feature group taxonomy

| Group | What it captures |
|---|---|
| **Static (at conversion)** | Initial risk profile the policy was born with. |
| **Payments, all history** | Accumulated behaviour, normalised by maturity. |
| **Payments, window 6m** | Recent behaviour. Isolates current trend from lifetime stock. |
| **Payments, window 12m** | Broader behavioural context. |
| **Last payment** | Specific character of the most recent payment. |
| **Riders** | Product configuration signals. |
| **Owner** | Demographic and cross-policy signals. |
| **Maturity** | Where the policy is in its lifecycle. |

### Signals I explored, tested, and dropped

Feature engineering matters as much for what you keep out as for what you keep
in. A few examples of variables I built, evaluated, and dropped from the final
set, all of them because they either leaked, were redundant with something
better, or added noise:

- **Recent payment status indicators** (last-payment cleared flag, days since last failure, days to next scheduled payment). Dropped for conceptual leakage, see §3.1.
- **Very short window payment ratios** (3-month equivalents of the 6-month behavioural features). Dropped in favour of the 6-month window, since the 3-month version was too noisy at the individual-policy level to matter over the 6-month label window.
- **12-month payment behaviour ratios.** Dropped in favour of 6-month equivalents. Highly correlated, and the 6-month version was more responsive to recent shifts.
- **Rollup risk-method flags.** Engineered as a rollup of individual "ever used method X" indicators. Redundant with the individual flags once trees can interact them.

Anyone can list the features that ended up in the model. That's a table of
contents, not a technical signal. What separates a case study from a leaderboard
entry is showing the shape of the search. What signals I hypothesised, what I
built to test them, what the tests said, and why the dropped features were
dropped. The final model has around 20 features. The exploration touched around 50.

### The payment method change example

One feature I want to call out because it worked, but not for the reason I
expected.

**Hypothesis.** In a subscription-like product, changes in payment method are a
friction signal. The client had to interact with billing, which correlates with
"this is not on autopilot anymore".

**What I found.** Changes from a low-friction method (automatic debit) to a
high-friction method (manual charge) were strongly predictive. Changes in the
other direction (manual to automatic) were only mildly predictive of retention.
Asymmetric signal.

**How I encoded it.** A count of method changes over the policy's history,
combined with the current payment method and the historical menu of methods this
policy has been on (a small set of "ever used method X" indicators). The tree
model interacts them. Concretely, "this policy has 2 method changes AND is
currently on manual charge AND was on automatic debit originally" is a specific
interaction the model can learn without me pre-computing it.

## 5. Model choice: LightGBM, and why not the alternatives

LightGBM as production model. Two comparisons matter.

### LightGBM vs. deep-learning tabular alternatives

- **Data shape.** Around a million rows, around 20 mixed-type features, moderate missingness. This is the regime where gradient boosting has been repeatedly shown to match or beat tabular deep learning while being an order of magnitude faster to train.
- **Interpretability.** The retention team needs a "why this customer is high risk" answer per contact, see §7. SHAP on trees is fast, canonical, and produces answers the business team can read. SHAP or attention on tabular deep models exists but is slower and less standardised.
- **Iteration speed.** With 7 iterations and continuous feature engineering, training runs happened many times per week. LightGBM trains in minutes on this data. A transformer would have slowed the loop by 10x with no clear performance case.

### LightGBM vs. XGBoost

The two are close cousins and either would work. LightGBM was chosen on three
practical grounds.

1. **Native categorical handling, no dimensionality explosion.** Several features are true categoricals with 5 to 20 levels each. XGBoost cannot consume categoricals directly and requires an explicit encoding. One-hot encoding would inflate the feature space (four categoricals with an average of ~10 levels each add ~40 binary columns for information already carried by 4 columns). This dilutes tree splits and slows training. LightGBM sidesteps it by finding the optimal split over category subsets directly, at the same time as it grows the tree. Zero encoding, zero extra preprocessing state, no leakage surface, no inflated feature matrix.
2. **Missing values.** LightGBM's default missing-value handling is empirically clean on this feature mix and did not require the imputation shims that XGBoost sometimes benefits from.
3. **Training speed on wide-and-tall tabular data.** LightGBM's leaf-wise growth is consistently faster than XGBoost's level-wise on the shape of this dataset. Not a dealbreaker, but multiplied by 7 iterations and by regular retraining runs, it adds up.

Despite XGBoost's slight edge in raw performance on some tabular benchmarks,
LightGBM remains the better fit here on the three grounds above.

The head-to-head comparison was not run empirically inside this project. Both
would have landed in the same performance ballpark. The Vertex AI AutoML benchmark
(§10) played the "am I missing a much better modelling approach?" role.

### Class imbalance handling: `scale_pos_weight` over resampling

With OOT prevalence around 3% (and training prevalence in a similar range), the
minority class is heavily under-represented. Left untreated, LightGBM's loss would
be dominated by the majority class and the model would learn to almost always
predict "not lapse", technically accurate and operationally useless.

Three standard options exist.

| Technique | What it does | Trade-off |
|---|---|---|
| **`scale_pos_weight`** (adopted) | Multiplies the gradient contribution of positive samples by `neg_count / pos_count`, rebalancing the loss in-place. | Preserves the full training set. No data thrown away. |
| **Random undersampling of the majority class** | Drops a random subset of negatives until classes are roughly balanced. | Simple, but throws away information. On a class this rare it also increases the variance of every retraining run. |
| **SMOTE / synthetic oversampling** | Synthesises new minority-class samples in feature space. | On mixed categorical and numeric features (this project) the synthetic samples are semantically meaningless and can push the model toward overfitting. Not appropriate here. |

`scale_pos_weight` was adopted. The full training set is preserved (this matters
at the tail of the score distribution, where evidence is already scarce), no
artefactual samples are generated, and the calibration step (§6) is applied
*after* the reweighting so the final probabilities are still on the natural
prevalence scale, not on the rebalanced-training-set scale.

The Vertex AI AutoML benchmark (§10) uses manual undersampling instead. Two
independent balancing techniques on the same dataset converging on the same
high-risk population is additional external validation that the signal is in the
data, not in the modelling choice.

## 6. Probability calibration: isotonic over Platt

LightGBM outputs a real-valued score that is a good *ranker* but not a good
*probability estimator*. In the raw score distribution, the top of the range
piles up at 0.95+ with very few actual lapses. The score is over-confident.

For a ranking-only use case this doesn't matter (any monotonic transformation
preserves ranking). It matters here for two reasons.

- The retention team communicates internally in probability language ("this policy has an 80% chance of lapsing"). A well-calibrated probability is a first-class deliverable, not an afterthought.
- The monitoring layer uses probability distributions to detect model drift, see §11. If raw scores were uncalibrated, drift signals would trip on calibration noise rather than on real distributional change.

Calibration is applied via `sklearn.calibration.CalibratedClassifierCV` as a
separately trained wrapper around the base LightGBM model. It is persisted as its
own artefact (`calibrated_model.joblib`) alongside the base model, so the ranking
model and the probability mapping can evolve independently.

### Isotonic over Platt scaling

Two standard options exist for post-hoc calibration.

| Method | Assumes | When it fits | When it fails |
|---|---|---|---|
| **Platt scaling** (sigmoid) | A sigmoid relationship between raw score and true probability. Two-parameter fit. | Small validation sets, near-sigmoid miscalibration. | Non-sigmoid miscalibration patterns (piecewise, monotonic-but-not-sigmoid). |
| **Isotonic regression** | Only that the mapping is monotonic. Non-parametric, piecewise-constant fit. | Larger validation sets (>1k positive samples). Handles arbitrary monotonic miscalibration. | Small samples (overfits). |

Isotonic was the choice. Two supporting facts.

1. **Sample size.** The training set has enough positive lapse examples that isotonic's flexibility is worth the additional parameters. This is well above the empirical threshold where isotonic starts to beat Platt.
2. **Observed miscalibration shape.** The base model's reliability diagram was not sigmoid-shaped. It had a plateau in the mid-range and a spike near 1.0. Platt would have flattened both artefacts into one smooth curve. Isotonic reproduces the shape piecewise, which the reliability diagram then confirms is closer to identity.

The calibration is refit whenever the base model is retrained, never in isolation.

## 7. Explainability with SHAP: a "why" per customer

**SHAP (SHapley Additive exPlanations)** is used at two levels.

### Global feature importance

Post-training, the SHAP summary tells us which feature *groups* move the score for
the population as a whole. In this model the drivers rank in the following order
of impact.

1. **Maturity.** Dominant driver by a wide margin. Very early-stage policies have the highest volatility. This is expected in an *early*-lapse model.
2. **Current payment method.** Clients on higher-friction methods (manual charge, mailed billing) lapse more than clients on autopilot (automatic debit).
3. **Recent payment coverage (windowed).** Recent coverage ratio, the clearest current-behaviour signal.
4. **Owner demographics.** Non-monotonic effect, with drop-offs at both extremes of the age range.
5. **Product characteristics** (product family and premium bracket). Product-level baseline risk.
6. **Payment reliability ratios (windowed).** Failure-over-attempt ratios in the recent and broader windows.

The maturity dominance is worth flagging because it means the model would
collapse to "just use maturity" if we let it. The relative-features work (§3.2)
exists specifically to keep the other signals audible against maturity's
information value.

### Per-customer explanation for the retention team

The single most operationally useful part of SHAP in this project. For every
scored policy, the model can emit the top 3 to 5 features that pushed this
specific policy up or down the score, expressed in natural language the retention
team can read.

> **Policy [ID], flagged high-risk (top decile), because:**
> · converted 5 months ago (early maturity)
> · currently paying by manual charge
> · 2 failed payments out of 6 attempts in the last 6 months
> · owner in the older bracket of the portfolio

This turns the model output from a black-box prioritised list into a
**conversation starter**. The retention agent picks up the phone knowing not only
*that* the policy is at risk but *what the strongest signal is*, which shapes
the pitch:

- **Payment friction signal** → offer to help set up automatic debit or automatic renewal.
- **Product-fit signal** (product characteristics unusual for owner demographic) → offer a product review conversation.
- **Multiple missed payments** → offer a payment plan or premium restructuring.
- **Very early maturity with several signals firing** → escalate to a senior retention agent.

The strategy per customer is not decided by the model. The model provides the
reason, and the human picks the response.

## 8. Results and decile-based segmentation

*All numbers below are approximate. The client owns the exact metrics.*

### Which metric drives decisions, and which doesn't

Before showing numbers, it matters *which* numbers we optimise for, because with
a ~3% positive class, standard metrics carry unequal weight.

**ROC-AUC, reported but not optimised.** ROC-AUC is included by convention
because reviewers expect it, but it is not the metric used to compare model
candidates in this project. With ~3% prevalence, the confusion matrix is
dominated by True Negatives. Roughly 97% of the population is correctly
classified as "not lapse" almost for free. This inflates ROC-AUC toward high
values even for mediocre models and makes it insensitive to differences in how
well the model handles the *minority* class, which is the entire point of
building a lapse predictor. ROC-AUC of 0.90 sounds strong but tells us
surprisingly little about operational performance on the 3% that actually matter.

**PR-AUC (precision-recall), the quality metric.** PR-AUC is computed *only* over
the ranking of the positive class and is not diluted by TNs. It is the correct
model-selection metric under class imbalance, and it is the one that moved
between iterations (§3). The leakage-heavy Iteration 1 scored ~0.75 PR-AUC on
OOT versus Iteration 7's ~0.5, even though ROC-AUC differed less between them.
PR-AUC is what told us Iteration 7 was actually solving the right problem.

**Recall at top-K, the operational metric.** In production, the retention team
has a fixed contact capacity per week (top 5%, 10%, or 15% of the ranked list,
see the three strategies below). Given that budget, the question is how many of
the actual future lapses did we manage to reach. That is recall, evaluated at
the specific cutoff the team can operate on. Precision matters too, but the
operational cost of a false negative (missing a client who was about to lapse,
irrecoverable revenue) is asymmetrically larger than the cost of a false positive
(a courtesy retention call to a customer who was staying anyway). Under that
asymmetry, the model is tuned and monitored on recall@top-K, with precision as a
secondary constraint that limits how deep into the ranked list the team can go
before the marginal contact stops being worth the effort.

The full picture, then:

- **ROC-AUC** is a sanity-check that the model can distinguish classes at all.
- **PR-AUC** is the internal metric for selecting between model versions.
- **Recall@top-K** is the operational metric the business tracks and the model is judged on.

### Overall performance (out-of-time)

| Metric | Range | Use | What it means |
|---|---|---|---|
| ROC-AUC | **~0.90** | Sanity check | The model orders policies by risk well. Inflated by TNs, not used for decisions. |
| PR-AUC | **~0.5** | Model selection | Given the ~3% prevalence, this is strong. Random would be ~0.03. |
| Recall @ top 10% | **~0.78** | **Operational** | Given the team's default contact budget (decile 1), the model captures ~78% of real lapses. |
| Logloss (calibrated) | **~0.08** | Calibration check | After isotonic calibration, probabilities are reliable. |
| Prevalence (OOT) | **~3%** | Context | 3 in 100 active policies lapse in the label window. |

Metrics on the out-of-time set (a strictly later snapshot than training) are as
strong as or stronger than on the training-time test set. The model generalises,
it is not overfitted to a specific period. This holds across multiple months of
OOT evaluation, see §9.

### Decile-based segmentation and retention action bands

Ranking policies by predicted probability into deciles is the operational
interface. The distribution of lapses across deciles is heavily concentrated at
the top.

| Decile | Approx. share of total lapses | Approx. lift vs. random | Recommended retention action |
|---|---|---|---|
| **1** (highest risk) | **~78%** | **~8x** | **Proactive intensive contact.** Personal call, tailored offer based on SHAP top drivers, senior agent for the sub-segment with multiple signals firing. This is where the model earns its keep. |
| **2** | ~7% | ~4x | **Standard proactive contact.** Templated but customised outreach (email or SMS with a callback path). Volume is high, automation-friendly. |
| **3–4** | ~7% combined | ~2–3x | **Passive monitoring and light touch.** Newsletter, general product-value communication, conversion-anniversary reminders. Not worth the cost of a call. |
| **5+** | ~8% combined | ≤2x | **No proactive action.** These policies contain a real but small share of lapses. Contacting them is not cost-efficient. Keep them in monitoring only. |

**Why segmentation by decile and not by threshold.** A probability threshold (say
"contact everyone above 0.15") is unstable across retraining. A slightly different
calibration shifts the threshold's meaning. Decile ranks are inherently
self-normalising. Decile 1 is always the top 10% of the current run, regardless
of how the score distribution has shifted. The retention team's capacity is
stable in headcount, and decile is the right unit for that.

### The three operational strategies (top-%)

Decile bands are the strategic decomposition. Within them, three concrete
operational runs are pre-computed for the retention team based on their current
contact capacity.

| Strategy | Contacts (of the portfolio) | Approx. recall (lapses captured) | Approx. precision | When to use |
|---|---|---|---|---|
| **Focused** | Top 5% | ~70% | ~40% | Small team month or low-capacity period. Cover most lapses with least noise. |
| **Balanced** | Top 10% (= decile 1) | ~78% | ~22% | Default. Team's standard operational cadence. |
| **Max reach** | Top 15% | ~83% | ~16% | Post-launch of a new product, or a period where retention capacity is expanded. |

The three strategies share the same underlying model. They are three different
cut points on the same ranked list, letting the business trade *efficiency*
(precision) against *coverage* (recall) without retraining.

## 9. Validation strategy

Three layers of validation, each catching a different failure mode.

### 9.1 Snapshot-based training with a temporal split

Training uses a snapshot of the active portfolio at date `T`, with the label
computed over `[T, T+6 months]`. The snapshot logic ("which policies were active
at `T` and only using data known before `T`") is enforced in the SQL that builds
the training set, not in the model code. This makes temporal leakage a
query-level invariant that survives model changes.

**Why one snapshot was sufficient.** Before committing to single-snapshot
training, I evaluated data drift across several snapshots taken at different
points in time. The population characteristics (feature distributions, prevalence,
maturity mix) were stable across those checks. Combined with the very large
training volume already available in a single snapshot, this made multi-snapshot
windowing unnecessary. Windowing is a technique to compensate for population
non-stationarity by pooling data across time; with a stationary population and
abundant volume per snapshot, it would have added complexity without adding
signal.

Cross-validation folds are stratified by lapse label but not by snapshot date
(there is only one snapshot per training run). This is deliberate, since we're
not testing robustness across time here. Time robustness is tested by the OOT
step below.

### 9.2 Out-of-time validation

A completely independent, later snapshot `T'` with a label window of `[T', T'+6
months]` is used as the primary generalisation test. Not a random holdout, since
retention patterns can drift over time and a random split would hide that drift.

The OOT window matches the training label window (6 months), so that ranking
performance and prevalence are measured on the same task the model was trained
for. This makes the OOT metrics directly comparable to training-time metrics,
which is what "stability of predictions" requires.

*(An earlier iteration used a shorter 2-month OOT window and applied a
conservative correction to compare against training. That was the pilot-phase
setup and is not the final validation regime.)*

### 9.3 Stability across months

The model is scored month by month over the OOT window. Two patterns are checked.

- **Ranking stability by month (PR-AUC by month).** PR-AUC is stable and high across all months. Slight natural attenuation appears for months further from the snapshot date, which is expected in any retention model. PR-AUC is used here (and not ROC-AUC) for the same reason it drives model selection in §8: it isolates minority-class behaviour and doesn't hide degradation under a mass of True Negatives.
- **Recall stability at fixed contact volume (recall@top10 by month).** Decreasing with distance from the snapshot, as expected. Near-term lapses are easier to spot than far-term ones. This is monitored, and it is not a failure signal.

The month-by-month view is what turns "the model works" into "the model works at
the cadence I need to operate it on".

## 10. External benchmark: Vertex AI AutoML

A second modelling approach was trained independently using **Vertex AI AutoML
Tabular** on the same training set, using manual undersampling of the majority
class to handle the class imbalance. This is deliberately a different technique
than the `scale_pos_weight` used in LightGBM, see §5. It makes the benchmark a
stronger test, since if both models rank the same customers as high risk despite
handling imbalance in different ways, the signal is robust to the balancing
choice.

Three things this benchmark answered.

1. *"Am I missing a much better modelling approach?"* No. Vertex AI produces comparable but slightly lower performance across ROC-AUC, PR-AUC, and recall at every contact tier. The LightGBM approach was not leaving obvious signal on the table.
2. *"Are the predictive patterns real, or am I overfitting to something specific to LightGBM?"* Vertex AI ranks the same customers as high risk. Two independent modelling approaches converging on the same population is external validation of the dataset, which is what actually matters for long-term robustness.
3. *"Is the result an artefact of `scale_pos_weight`?"* No. Manual undersampling arrives at the same high-risk population. The imbalance-handling choice affects efficiency, and it does not affect the identity of the customers flagged.

The benchmark also produced a slightly flatter score distribution than LightGBM,
with more overlap in the middle range between actual lapses and non-lapses. At
the same recall target, Vertex AI requires more contacts to hit it. LightGBM was
operationally cheaper to run, and the difference was not just marginal in AUC.

The Vertex AI benchmark model is retained as a challenger and retrained
periodically alongside the production model. If it ever surpasses the production
model on OOT performance, the retraining process migrates the benchmark to
production and the old production becomes the new benchmark.

## 11. Monitoring, drift and retraining triggers

Monitoring is a first-class part of the pipeline, not an afterthought. Two
routines run right after each weekly scoring run and land results in dedicated
tables in BigQuery.

### 11.1 Run summary: aggregate distribution health

For each weekly run, the summary checks:

- **Volume checks.** Row count, distinct-entity count, duplicate rate. Catches upstream pipeline breakage. An ETL job silently dropping half the portfolio does not surface as a model error, and it surfaces as a row-count anomaly.
- **Data quality checks.** Null rate on entity ID and probability, out-of-range probability count. Catches artefact corruption or schema drift.
- **Score distribution snapshot.** Mean, std, p50, p95, p99 of the score. Long-term this is the fingerprint of healthy scoring behaviour.
- **Score drift vs. previous run.** Absolute delta of mean, p95, and top-decile share.
- **Aggregated semaphore.** `OK` / `WARN` / `CRIT` based on all of the above.

### 11.2 Entity-level jump detection

Aggregate stats can look flat while a subset of policies is moving sharply. A
companion routine surfaces every policy whose probability changed by more than a
threshold delta between two consecutive runs. This produces a small, targeted
table that retention can pull to see who jumped from low risk to high risk this
week. These are often the policies where a real change happened (payment method
change, first missed payment), and are the most valuable prospects for
intervention.

### 11.3 Drift detection with PSI

Two flavours of Population Stability Index (PSI) run monthly, not per-scoring-run.

- **Data drift (PSI on features).** For each key feature, PSI is computed between the training-period distribution and the current-month scored distribution. Values above ~0.1 are `WARN`; above ~0.2 are `CRIT`. Catches distributional shift in inputs, for example a new product family with different payment behaviour becoming a larger share of the portfolio.
- **Model drift (PSI on scored probability).** Same statistic on the score distribution itself. Catches the case where feature distributions look fine individually but their joint effect on the score has shifted.

### 11.4 Retraining triggers

The model is retrained on a scheduled cadence (quarterly by default). The
monitoring layer can trigger an early retrain if any of the following fires.

- Data drift PSI on any top-5 feature exceeds `CRIT` for two consecutive months.
- Model drift PSI on the score distribution exceeds `CRIT` for one month.
- Recall on the top decile (measured against the previous OOT window as new labels mature) drops below the training-time confidence band.
- A structural business change (new product launch, changes to the payment method mix, regulatory change on grace periods) is flagged by the business team. This bypasses the metric-based triggers.

The retraining process itself is not automated end-to-end. A human review of the
drift analysis is required before a new model artefact is promoted. I have seen
enough incidents from "automatic retraining without human review" to scope it out
here deliberately.

## 12. Model card: scope, assumptions and limitations

### Intended use

Rank active policies of an existing insurance portfolio by 6-month lapse
probability, to feed a proactive retention workflow with a human agent in the
loop.

### Out of scope

- **Automatic actions on the client** (cancelling, upgrading, offering a discount) without human review. The model outputs risk, not decisions.
- **New products with no history in the training data.** The model has no signal on features it has never seen. Scores for a brand-new product family should be treated as noise until enough conversions accumulate and the model is retrained.
- **Reactive lapse recovery** (policies already in grace period). This model is by design not built for that. A separate reactive workflow handles them.
- **Individual-policy causal claims.** SHAP explains what pushed a *score*, not what is *causing* the underlying risk. "The model says this policy is risky because of maturity" does not translate to "shortening maturity would reduce risk". That would require a causal model.

### Known limitations

1. **Maturity dominance.** Maturity has the highest information value by a wide margin. The relative-features work (§3.2) mitigates it and does not eliminate it. If the retention team wants to isolate signal from behaviour (independent of maturity), a maturity-stratified version of the model would be needed. Noted, and out of the current scope.
2. **Prevalence sensitivity.** The 3% prevalence gives a healthy PR-AUC, and calibration in the very-low-probability range is noisier than in the mid-range. Isotonic calibration handles this reasonably. A monotonic-only assumption on such a sparse regime is the right conservative choice.
3. **Recall degradation for far-term lapses.** The model is strongest at the near end of the 6-month window and weaker at the far end. This is expected and monitored. A use case that specifically prioritises policies lapsing in month 5 or 6 would need a separate horizon-specific model.
4. **No feature-drift detection at ingestion time.** Drift is detected monthly, not at scoring time. A drastic upstream schema change would only surface after the run. This is mitigated by the run-summary null-rate and out-of-range flags, and is not fully eliminated.

### Fairness and ethical scope

The model uses owner age as a feature. Age is a protected characteristic in many
jurisdictions. In this deployment, the model output is used to prioritise
retention outreach, not to price policies or refuse coverage. That is why age is
retained as a feature. If the output were ever to feed into a pricing or
underwriting decision, the age feature would have to be removed and the model
retrained and re-audited from scratch.

## 13. Scoping decisions and repository contents

### What was deliberately left out of the production system

- **No idempotency or backfill** in the scoring script. The DAG is scheduled and the business is fine with re-running the previous week manually if needed. Adding proper backfill was scoped out as phase 2, since it would have delayed the first production run by weeks.
- **No structured logging inside the scoring script.** Cloud Composer's task-level logs are enough at the current volume. Structured logging is in the phase 2 backlog.

The general pattern is to ship the production loop first, monitor the outputs,
and harden the parts that actually cause incidents in that order.

### What is in this repo

```
lapse-prediction-case-study/
├── README.md                       ← this document
└── snippets/
    ├── preprocessing_dict.py       ← pattern: versioned transformation state
    └── calibration_and_deciles.py  ← pattern: isotonic wrapper and decile lift analysis
```

The full training code, business feature engineering, and production scoring
scripts are the client's property and are not published here. The snippets
illustrate transferable technique (versioned preprocessing state, calibration
wrapping, decile lift computation), and they do not contain client logic.

### What this case study is meant to show

- End-to-end delivery on GCP: framing, data, training, production, monitoring.
- Discipline around leakage, both temporal and conceptual.
- Absolute-to-relative feature engineering as a maturity signal (§3.2).
- Ranking model separated from probability calibration as versioned artefacts.
- Segmentation by decile with a per-band retention action, not just a threshold.
- Monitoring, drift and retraining triggers designed alongside the model, not bolted on.
- A model card that names its own limits.

---

Martín Terzano · [helliumlab.com](https://helliumlab.com) · [LinkedIn](https://www.linkedin.com/in/martinterzano) · martin@helliumlab.com
