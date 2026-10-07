# ELAPS: A Landmark-Based Audit Protocol for Label-Anchoring Bias in Retail-Banking Propensity Scoring

Official code and reproducibility materials accompanying the manuscript:

> **ELAPS: A Landmark-Based Audit Protocol for Label-Anchoring Bias in Retail-Banking Propensity Scoring**  
> **Authors:** Ngo Thi Thu Trang, Tran Thu Trang, and Ha-Nam Nguyen

---

## 1. Purpose of this repository

This repository contains the notebooks used to implement and audit **ELAPS (Event–Landmark Audit for Propensity Scoring)**, a reproducible protocol for detecting and quantifying **label-anchoring bias** in retail-banking propensity models.

The problem addressed by ELAPS is subtle. In a retrospective extract, adopters are often described using information available up to their adoption date, while non-adopters are described using information available up to the end of the extract. Every feature can still be point-in-time correct relative to its own customer-specific cutoff, yet the two outcome classes are observed over systematically different time windows. Ordinary banking aggregates can then encode observation duration, onboarding stage, or other temporal structure that would not be available at a common deployment decision time.

ELAPS is therefore **an audit and evaluation protocol, not a new classifier**. The classifiers used in the experiments are established models such as XGBoost, LightGBM, logistic regression, random forest, extra trees, AdaBoost, and gradient-boosted trees. The methodological contribution lies in:

1. constructing an auditable event-anchored historical reference;
2. rebuilding the prediction problem at a common landmark through a **Forward-Looking Deployment Cohort (FLDC)**;
3. comparing alternative feature-window rules on the **same customers, labels, and data splits**;
4. separating the paired window contrast from the broader retrospective-to-deployment gap; and
5. reporting a nine-item audit battery with explicit completion status.

Two datasets are used:

- **VIB Datathon retail-banking data** — full ELAPS audit instance;
- **Santander Product Recommendation data** — directional first-year replication using monthly product snapshots.

---

## 2. Research questions implemented by the code

The notebooks are organized around the three research questions in the manuscript:

- **RQ1 — Label-anchoring bias:** On identical customers, labels, and splits, how much does event anchoring of feature windows change discrimination and campaign-ranking metrics?
- **RQ2 — Deployment-time performance:** What does a correctly posed model achieve at a common decision time, in terms of both ranking and probability quality?
- **RQ3 — Directional replication:** Does the direction of the paired window contrast persist on a second public retail-banking dataset with different event measurement?

---

## 3. Core ELAPS design

### 3.1 Step 1 — Event-anchored historical reference

For customer `c`, let:

- `r(c)` = relationship start;
- `e(c)` = first observed credit-card opening/event;
- `T_end` = administrative extract end;
- `tau(c)` = feature cutoff.

The event-anchored retrospective rule is:

```text
if an event is observed before T_end:
    tau(c) = e(c)
    y(c)   = 1
else:
    tau(c) = T_end
    y(c)   = 0
```

Only records satisfying

```text
r(c) <= time(record) < tau(c)
```

are admissible for feature construction.

This rule enforces predictor-level point-in-time correctness, but positive and negative customers can still have systematically different admissible history lengths. The resulting model is treated as a **historical reference**, not as a prospective deployment estimate.

### 3.2 Step 2 — Forward-Looking Deployment Cohort (FLDC)

A common landmark is imposed at relationship day `L`.

For the primary VIB analysis:

```text
L = 60 days
```

A customer is eligible only if:

- the customer is observable at the landmark; and
- the customer has not already opened/held the target card on or before day 60.

Predictors use only information available before the common landmark. The target is subsequent adoption after day 60 within the available follow-up.

VIB landmark sensitivity is also evaluated at:

```text
30, 60, 90, and 180 days
```

### 3.3 Step 3 — Same-population A/B/C decomposition

Branches A, B, and C use the **same FLDC-eligible customers, the same post-landmark labels, the same train/holdout assignments, and separately refitted models**.

- **Branch A — fixed 60-day window**  
  Uses only the first 60 relationship days for every customer.

- **Branch B — event-anchored window**  
  Uses the same customers and labels as A, but positive histories end at the positive event and negative histories end at the applicable observation endpoint.

- **Branch C — B + explicit cutoff tenure**  
  Uses Branch B predictors and adds elapsed time from relationship start to the Branch-B cutoff.

The two main paired contrasts are:

```text
D_window   = AUC_B - AUC_A
D_explicit = AUC_C - AUC_B
```

`D_window` is the principal same-population estimate of the performance advantage associated with the event-anchored window rule and its corresponding refitted model. It is **not a causal effect of window length**.

The broader retrospective-to-deployment contrast is:

```text
D_deploy = AUC_retrospective - AUC_FLDC
```

`D_deploy` is a **composite deployment gap**. It mixes changes in cohort, target, follow-up, feature-window construction, and model-selection conditions and must not be described as a pure leakage estimate.

### 3.4 Step 4 — Nine-item audit battery

ELAPS records the following diagnostics:

1. point-in-time traceability;
2. 1,000-run label permutation control;
3. window-length probe;
4. dominant-feature / demographic ablation;
5. placebo-cutoff sensitivity;
6. same-population A/B/C decomposition;
7. FLDC reconstruction;
8. calibration assessment; and
9. sampling robustness within FLDC.

The diagnostics target different mechanisms. Their AUC changes are **not additive**.

---

## 4. Main results reproduced by the repository

### 4.1 VIB retrospective reference

The retrospective VIB cohort contains:

```text
127,460 customers
19,714 adopters
107,746 non-adopters
15.47% adoption prevalence
```

The leading retrospective configurations cluster near AUC 0.95. The historical XGBoost + ROS reference has AUC approximately `0.9512`, while XGBoost without resampling reaches approximately `0.9519`.

The historical reference achieves:

```text
Precision@10% = 0.782
Lift@10%      = 5.055
Capture@10%   = 0.506
Capture@20%   = 0.819
Capture@30%   = 0.963
```

These are historical ranking results and are not interpreted as deployment performance.

### 4.2 VIB day-60 FLDC

The canonical VIB day-60 FLDC contains:

```text
94,453 eligible customers
4,374 post-landmark adopters
4.63% prevalence
```

Within the prespecified XGBoost pair, the unresampled model is selected using training-set cross-validation.

Primary random-holdout results:

```text
AUC             = 0.9102
KS              = 0.6733
Precision@10%   = 0.2968
Lift@10%        = 6.4084
Capture@10%     = 0.6411
Capture@20%     = 0.8343
Capture@30%     = 0.9097
Brier score     = 0.0346
Brier Skill     = 0.2163
ECE             = 0.0039
```

The ROS comparator has similar ranking but substantially worse probability quality, with BSS approximately `-1.40`.

### 4.3 VIB same-population A/B/C decomposition

The canonical holdout contains `18,891` identical customers for all three branches.

| Branch | Feature-window rule | AUC | KS | BSS | Precision@10% | Lift@10% | Capture@10% |
|---|---|---:|---:|---:|---:|---:|---:|
| A | fixed first 60 days | 0.9102 | 0.6733 | 0.2163 | 0.2968 | 6.4084 | 0.6411 |
| B | event-anchored | 0.9607 | 0.8137 | 0.4264 | 0.3937 | 8.4988 | 0.8503 |
| C | B + cutoff tenure | 0.9669 | 0.8202 | 0.4817 | 0.4011 | 8.6587 | 0.8663 |

Paired AUC contrasts:

```text
D_window   = B - A = +0.0504   95% CI [0.0428, 0.0582]
D_explicit = C - B = +0.0062   95% CI [0.0039, 0.0085]
C - A              = +0.0566   95% CI [0.0490, 0.0645]
```

All paired DeLong tests reported for these contrasts are `p < 0.001`.

### 4.4 VIB split/refit variability

The A/B comparison is repeated over `20` independent stratified 80/20 split/refit runs.

```text
Completed runs                = 20
Mean AUC, Branch A            = 0.909488
Mean AUC, Branch B            = 0.962463
Mean D_window                 = 0.052975
Empirical 2.5th percentile    = 0.046268
Empirical 97.5th percentile   = 0.058488
Runs with D_window > 0        = 20 / 20
```

This analysis captures split and refitting variability. The percentile range is not presented as a confidence interval for the mean.

### 4.5 VIB fixed post-landmark follow-up

A fixed follow-up analysis uses:

```text
Landmark L            = 60 days
Follow-up H            = 180 days
Common endpoint        = day 240
Eligible customers     = 31,149
Events                 = 1,454
Holdout customers      = 6,230
Holdout events         = 291
```

Results:

```text
AUC_A                  = 0.891002
BSS_A                  = 0.139893
AUC_B                  = 0.906161
D_window^H             = 0.015158
95% paired-bootstrap CI= [0.004543, 0.026398]
```

The contrast remains positive under fixed follow-up, although the smaller magnitude cannot be attributed only to follow-up length because the eligible cohort and target also change.

### 4.6 Santander first-year directional replication

The canonical Santander first-year S0 cohort contains:

```text
153,399 customers
916 positive post-day-60 monthly card statuses
122,719 training customers / 733 positives
30,680 holdout customers / 183 positives
```

Shared S0 holdout results:

| Branch | AUC | KS | BSS | Precision@10% | Lift@10% | Capture@10% |
|---|---:|---:|---:|---:|---:|---:|
| A — fixed 60-day | 0.885617 | 0.647980 | 0.016537 | 0.031617 | 5.300546 | 0.530055 |
| B — event-anchored | 0.972259 | 0.819746 | 0.434479 | 0.053455 | 8.961749 | 0.896175 |
| C — B + tenure | 0.998651 | 0.968654 | 0.781478 | 0.059648 | 10.000000 | 1.000000 |

Paired S0 contrasts:

```text
B - A = +0.086642   95% CI [0.072011, 0.101842]
C - B = +0.026391   95% CI [0.019052, 0.034094]
C - A = +0.113033   95% CI [0.097854, 0.128813]
```

The S1 sensitivity rule excludes S0 customers with no post-day-60 snapshot in the available first-year source while preserving retained labels and original split membership.

```text
S1 customers       = 139,067
S1 positives       = 916
S1 training        = 111,283 / 733 positives
S1 holdout         = 27,784 / 183 positives
```

Refitted S1 AUCs:

```text
A = 0.876453
B = 0.976517
C = 0.998368
B - A = +0.100064   95% CI [0.083936, 0.116179]
```

Santander is used as a **directional replication**. Its monthly positive-status proxy is coarser than the VIB event dates, and its effect size is not assumed to transfer to another institution.

---

## 5. Repository structure

```text
ELAPS/
├── README.md
├── requirements.txt
├── Notebooks/
│   ├── 01_VIB_Dataprocessing.ipynb
│   ├── 02_VIB_ELAPS.ipynb
│   ├── 03_VIB_Placebo.ipynb
│   ├── 04_VIB_FLDC_30D.ipynb
│   ├── 05_VIB_FLDC_60D.ipynb
│   ├── 06_VIB_FLDC_90D.ipynb
│   ├── 07_VIB_FLDC_180D.ipynb
│   ├── 08_VIB_LightGBM_ROS.ipynb
│   ├── 09_Santander_ELAPS.ipynb
│   ├── 10_Santander_Placebo.ipynb
│   ├── 11_Santander_FLDC_60D.ipynb
│   ├── 12_Santander_FLDC_90D.ipynb
│   ├── 13_VIB60ABC.ipynb
│   ├── 14_VIB_FLDC_60D_B.ipynb
│   ├── 15_VIB_FLDC_60D_C.ipynb
│   ├── 16_SantanderABC.ipynb
│   ├── 17_VIB_Dwindow_Refit_20Seeds.ipynb
│   └── 18_VIB_FixedHorizon_H180_A60_B240.ipynb
└── data/
```

The repository contains both **canonical manuscript notebooks** and several **historical/exploratory notebooks** retained for transparency. The sections below distinguish them explicitly.

---

## 6. Detailed notebook guide

### `01_VIB_Dataprocessing.ipynb`

**Role:** canonical VIB data construction and temporal-admissibility preparation.

**Main tasks**

- load and link VIB source tables;
- parse and validate relationship-start, event, and source timestamps;
- reject impossible date orderings such as recorded card opening before relationship start;
- prepare the fields needed by retrospective and landmark analyses;
- maintain customer IDs for linkage/audit while excluding identifiers from the predictor matrix;
- implement the **point-in-time traceability check** by verifying that the maximum contributing source timestamp is strictly before the applicable cutoff for each customer/feature vector.

**Manuscript/Supplement links**

- Step 1 data preparation;
- main-text VIB data description;
- ELAPS audit item 1: point-in-time traceability;
- Supplementary Table S3 predictor construction support;
- contributes to the VIB cohort-description statistics used to motivate the day-60 landmark.

**Important:** the traceability check establishes predictor-level temporal admissibility. It does not by itself eliminate label-anchoring bias.

---

### `02_VIB_ELAPS.ipynb`

**Role:** canonical VIB retrospective ELAPS experiment and the full retrospective learner/sampling screen.

**Main tasks**

- run the **35-configuration retrospective screen**:
  - 7 classifiers;
  - 5 sampling strategies;
- generate the retrospective XGBoost reference results;
- compute retrospective AUC, KS, threshold metrics, and campaign-ranking metrics;
- run the **1,000-run label-permutation test without resampling**;
- run the **1,000-run label-permutation test with ROS applied after label permutation**;
- compare the matched permutation pipelines using the same permutation seeds;
- run feature ablations:
  - remove `AVG_CA_BALANCE`;
  - remove `CLIENT_SEX`;
  - remove `AGE`;
  - remove both `AGE` and `CLIENT_SEX`;
- generate retrospective SHAP analyses.

**Canonical outputs linked to the paper**

- Supplementary Table S1 — full 35-configuration retrospective screen;
- Supplementary Table S7(a) — permutation results;
- Supplementary Table S7(b) — ablation results;
- Supplementary Table S12 — leading retrospective configurations;
- Supplementary Table S13 — VIB retrospective ranking;
- Figure S1 — retrospective SHAP dependence for `AGE`;
- Figure S2 — retrospective SHAP interaction `AGE × COUNT_CA_ACCT`;
- Figure S4(a) — global SHAP summary for retrospective XGBoost + ROS;
- main-text retrospective results in Section IV-B.

**Permutation details**

The permutation control shuffles **training labels only**. The holdout labels remain unchanged. In the ROS pipeline, oversampling is performed **after** labels are permuted. Matched seeds are used when comparing ROS with no sampling.

Reported null summaries include:

```text
No sampling:
observed AUC = 0.951855
null mean    = 0.505034
null SD      = 0.044816
0 / 1,000 null AUCs >= observed
plus-one p   = 0.000999

ROS:
observed AUC = 0.951179
null mean    = 0.522290
null SD      = 0.045503
0 / 1,000 null AUCs >= observed
plus-one p   = 0.000999

Matched ROS - no-sampling null shift:
mean shift   = +0.0173
95% CI       = [0.0162, 0.0183]
ROS higher   = 851 / 1,000 matched permutations
```

The null-pipeline shift is reported as a pipeline sensitivity, not as a leakage estimate.

---

### `03_VIB_Placebo.ipynb`

**Role:** canonical VIB placebo-cutoff sensitivity diagnostic.

**Main tasks**

- assign alternative pseudo-cutoffs designed to reduce class-dependent window asymmetry;
- repeat the placebo construction five times;
- retrain/evaluate the model under each placebo construction;
- summarize AUC, KS, Precision@10%, Lift@10%, and Capture@30%.

**Canonical output**

- Supplementary Table S8 — VIB placebo-cutoff sensitivity;
- contributes to ELAPS audit item 5 and main-text Table 7.

Reported result:

```text
AUC          = 0.868 ± 0.001
KS           = 0.559 ± 0.001
Lift@10%     = 4.170 ± 0.028
Capture@30%  = 0.767 ± 0.002
```

The decline is treated as **bounding/sensitivity evidence**, not as a causal leakage estimate, because the pseudo-cutoffs can also discard legitimately available temporal information.

---

### `04_VIB_FLDC_30D.ipynb`

**Role:** VIB 30-day landmark sensitivity.

**Main tasks**

- rebuild eligibility at day 30;
- reconstruct predictors using only the admissible pre-landmark history;
- rebuild the post-landmark outcome;
- refit and evaluate the FLDC model.

**Manuscript/Supplement links**

- contributes the 30-day row of Supplementary Table S11;
- contributes to Figure S5.

Reported landmark summary:

```text
Eligible customers = 106,271
Prevalence          = 6.51%
AUC                 = 0.897
Lift@10%            = 6.04
```

---

### `05_VIB_FLDC_60D.ipynb`

**Role:** **primary deployment-aligned VIB experiment**.

**Main tasks**

- construct the canonical day-60 FLDC;
- compare the prespecified XGBoost pair:
  - XGBoost without resampling;
  - XGBoost + ROS;
- select between the pair using training-set five-fold CV AUC;
- evaluate the selected model on the untouched random holdout;
- compute AUC, KS, threshold metrics, and top-K campaign metrics;
- compute Brier score, Brier Skill Score, and ECE;
- run subgroup monitoring by sex and age group;
- generate the canonical FLDC global SHAP analysis;
- provide Branch A inputs used later in the paired A/B/C decomposition.

**Canonical outputs linked to the paper**

- primary random-split columns of main-text Table 4;
- Supplementary Table S6 — VIB subgroup monitoring;
- 60-day row of Supplementary Table S11;
- Figure S3 — subgroup AUC and within-group Capture@20%;
- Figure S4(b) — global SHAP summary for canonical FLDC XGBoost without resampling;
- contributes to Figure S5;
- provides canonical Branch A for Table 5 / the A/B/C experiment.

**Model-selection rule**

The holdout is not used to select between XGBoost with and without ROS. Selection is based on mean cross-validated AUC in the training partition.

**Important distinction**

Ranking and calibration are assessed separately. ROS can change recall at threshold `0.5` without improving ranking and can substantially degrade probability calibration because the training class prior is altered.

---

### `06_VIB_FLDC_90D.ipynb`

**Role:** VIB 90-day landmark sensitivity.

**Outputs**

- 90-day row of Supplementary Table S11;
- contributes to Figure S5.

Reported summary:

```text
Eligible customers = 80,100
Prevalence          = 4.10%
AUC                 = 0.912
Lift@10%            = 6.65
```

---

### `07_VIB_FLDC_180D.ipynb`

**Role:** VIB 180-day landmark sensitivity.

**Outputs**

- 180-day row of Supplementary Table S11;
- contributes to Figure S5.

Reported summary:

```text
Eligible customers = 38,337
Prevalence          = 3.61%
AUC                 = 0.949
Lift@10%            = 8.16
```

The higher AUC at day 180 is **not interpreted as proof that day 180 is a better operating landmark**, because eligibility, prevalence, follow-up, cohort composition, and decision time change simultaneously.

---

### `08_VIB_LightGBM_ROS.ipynb`

**Role:** standalone/auxiliary LightGBM + ROS reproduction and robustness notebook.

The **full 35-configuration screening grid reported in Supplementary Table S1 is generated in `02_VIB_ELAPS.ipynb`**. Notebook 08 is retained as a focused LightGBM + ROS run and is not required as a separate source for Table S1.

It is useful for checking the leading non-XGBoost retrospective learner/sampling configuration reported for context in the manuscript.

---

### `09_Santander_ELAPS.ipynb`

**Role:** **historical/exploratory Santander retrospective experiment**, not the canonical first-year paired replication.

This notebook corresponds to the earlier Santander historical model with a different eligibility rule and target. The manuscript notes an earlier run with approximately:

```text
255,627 customers
1,591 positives
AUC ≈ 0.963
```

Because that experiment does not use the canonical first-year Fixed60D population/target, its score **must not be paired with the canonical Branch A score** and must not be used to compute the paper's Santander `D_window`.

Figure S4(c), which shows an earlier exploratory Santander historical SHAP model, belongs to this historical/exploratory lineage and is explicitly not the S0 or S1 first-year canonical model.

---

### `10_Santander_Placebo.ipynb`

**Role:** legacy/exploratory Santander placebo notebook.

**Critical reporting rule:** the current manuscript reports **no verified placebo result for the canonical first-year Fixed60D Santander cohort**.

Therefore:

- outputs from this notebook are **not** reported as the Santander placebo result in the final audit table;
- the Santander placebo status in main-text Table 7 and Supplementary Table S8 is **Not run**;
- the older placebo value from an unverified cohort is excluded from the manuscript.

This notebook is retained only for historical/reproducibility transparency.

---

### `11_Santander_FLDC_60D.ipynb`

**Role:** Santander day-60 FLDC-style preparation/checking notebook.

It supports construction and checking of a day-60 comparator for the first-year Santander workflow. The **canonical final manuscript A/B/C, S0/S1, ranking, bootstrap, and refitting results are generated in `16_SantanderABC.ipynb`**.

Use notebook 11 as a preparation/development step, not as the authoritative source of Tables 6 and S14-S17.

---

### `12_Santander_FLDC_90D.ipynb`

**Role:** exploratory Santander 90-day landmark analysis.

This notebook is retained for exploratory reproducibility but **its 90-day results are not reported in the current manuscript or Supplementary Tables S1-S19**.

Do not substitute its outputs for the canonical first-year day-60 replication.

---

### `13_VIB60ABC.ipynb`

**Role:** canonical VIB same-population A/B/C audit and several downstream robustness analyses.

**Inputs**

- Branch A prepared from the canonical day-60 FLDC workflow;
- Branch B prepared by `14_VIB_FLDC_60D_B.ipynb`;
- Branch C prepared by `15_VIB_FLDC_60D_C.ipynb`.

**Main tasks**

1. fit and compare A/B/C on identical FLDC customers, labels, and splits;
2. calculate `D_window`, `D_explicit`, and `C-A`;
3. compute paired bootstrap confidence intervals;
4. run paired DeLong tests;
5. report branch-specific CV AUC, KS, BSS, and ranking metrics;
6. run the **window-length probe** by adding explicit tenure/history duration;
7. run the **relationship-start cohort-ordered robustness analysis**;
8. run/assemble the FLDC **sampling robustness comparison** between XGBoost without resampling and XGBoost + ROS.

**Canonical outputs linked to the paper**

- main-text Table 5 — VIB A/B/C same-population decomposition;
- VIB panel of main-text Figure 3;
- Supplementary Table S10 — relationship-start cohort robustness;
- contributes to ELAPS audit item 3 — window-length probe;
- contributes to ELAPS audit item 6 — same-population decomposition;
- contributes to ELAPS audit item 9 — FLDC sampling robustness;
- cohort-ordered column of main-text Table 4.

**Window-length probe**

Reported retrospective sensitivity:

```text
AUC: 0.9512 -> 0.974
KS:  0.793  -> 0.843
```

The explicit duration variable is excluded from the principal models. Its effect demonstrates that observation duration carries label-related information, but the larger A/B paired contrast shows that ordinary aggregates can already encode much of the window information even without an explicit tenure variable.

---

### `14_VIB_FLDC_60D_B.ipynb`

**Role:** canonical data-construction notebook for VIB Branch B.

**Main tasks**

- preserve the canonical day-60 FLDC customer IDs and labels;
- construct event-anchored features on those same customers;
- use the positive event cutoff for adopters;
- use the applicable observation endpoint for non-adopters;
- validate the Branch-B feature matrix before the comparative model fitting in notebook 13.

This notebook performs **feature-matrix construction**, not the final A/B/C statistical comparison.

---

### `15_VIB_FLDC_60D_C.ipynb`

**Role:** canonical data-construction notebook for VIB Branch C.

**Main tasks**

- start from the Branch-B feature construction;
- add explicit elapsed time from relationship start to the Branch-B cutoff;
- validate the Branch-C matrix;
- provide the input used by notebook 13 for `D_explicit = AUC_C - AUC_B`.

As with notebook 14, the final comparative modelling and paired inference are performed in notebook 13.

---

### `16_SantanderABC.ipynb`

**Role:** **canonical Santander first-year replication notebook**.

This notebook is the authoritative source for the final S0/S1 Santander results reported in the manuscript and Supplementary Material.

**Main tasks**

- construct/use the canonical first-year S0 cohort;
- fit Branches A, B, and C independently on the shared train split;
- evaluate all branches on the same S0 holdout IDs and labels;
- compute AUC, KS, Brier score, BSS, threshold metrics, and top-K ranking;
- compute five-fold CV AUC where reported for the S0 A/B/C comparison;
- run paired 1,000-resample bootstrap intervals for branch AUC contrasts;
- run paired DeLong tests;
- build the S1 sensitivity subset by excluding customers without later post-day-60 snapshots;
- preserve retained labels and original train/test membership;
- evaluate saved S0 models on the retained S1 holdout (`S0_on_S1_holdout`);
- refit A/B/C using the retained S1 training IDs (`S1_refit`);
- compare S1-refit predictions with the S0 models on the **identical S1 holdout**;
- generate the Santander panel of the main A/B/C figure.

**Canonical outputs linked to the paper**

- main-text Table 6 — Santander first-year A/B/C;
- Santander panel of main-text Figure 3;
- Supplementary Table S14 — S0 top-K ranking;
- Supplementary Table S15(a) — S0/S1 cohort counts;
- Supplementary Table S15(b) — S0, S0-on-S1, and S1-refit AUCs;
- Supplementary Table S15(c) — KS, Brier, BSS, and Capture@10%;
- Supplementary Table S16 — paired AUC contrasts;
- Supplementary Table S17 — refitting effects on the identical S1 holdout;
- Santander rows of main-text Table 7 where the diagnostic was actually completed.

**S0/S1 interpretation**

S1 improves follow-up observability by requiring at least one later snapshot, but it does not establish complete-year follow-up or prove every retained negative label is a true non-event.

---

### `17_VIB_Dwindow_Refit_20Seeds.ipynb`

**Role:** VIB split/refit robustness for the primary A/B contrast.

**Main tasks**

- repeat the matched A/B experiment on `20` independent stratified 80/20 splits;
- keep customer IDs, labels, and the split assignment matched within each A/B run;
- refit Branch A and Branch B independently for every seed;
- compute `D_window = AUC_B - AUC_A` for every run;
- summarize the empirical distribution across completed runs.

**Canonical output**

- Supplementary Table S18 — repeated split and refitting variability;
- main-text statement that all 20 contrasts are positive and mean `D_window ≈ 0.0530`.

This analysis addresses refitting/split variability that is not captured by bootstrapping a single fixed set of predictions.

---

### `18_VIB_FixedHorizon_H180_A60_B240.ipynb`

**Role:** VIB fixed post-landmark follow-up sensitivity.

**Design**

```text
landmark                 = day 60
post-landmark follow-up  = 180 days
common endpoint          = day 240
```

Eligibility requires sufficient follow-up to day 240.

- Branch A uses the original first-60-day predictors;
- Branch B uses pre-event information for positives and pre-day-240 information for negatives;
- A and B are refitted on the same stratified 80/20 split.

**Canonical output**

- Supplementary Table S19 — fixed-horizon VIB A/B comparison;
- main-text fixed-follow-up robustness result.

This notebook evaluates whether the direction of `D_window` persists when all eligible customers have the same nominal post-landmark follow-up horizon.

---

## 7. Canonical execution order

### 7.1 VIB manuscript workflow

Recommended order:

```text
01_VIB_Dataprocessing.ipynb
        ↓
02_VIB_ELAPS.ipynb
        ↓
03_VIB_Placebo.ipynb
        ↓
04_VIB_FLDC_30D.ipynb
05_VIB_FLDC_60D.ipynb   ← primary FLDC
06_VIB_FLDC_90D.ipynb
07_VIB_FLDC_180D.ipynb
        ↓
14_VIB_FLDC_60D_B.ipynb
15_VIB_FLDC_60D_C.ipynb
        ↓
13_VIB60ABC.ipynb
        ↓
17_VIB_Dwindow_Refit_20Seeds.ipynb
18_VIB_FixedHorizon_H180_A60_B240.ipynb
```

`08_VIB_LightGBM_ROS.ipynb` is auxiliary and can be executed after preprocessing when a focused LightGBM + ROS check is desired.

### 7.2 Santander manuscript workflow

For the **canonical first-year replication**:

```text
11_Santander_FLDC_60D.ipynb   [preparation/checking]
        ↓
16_SantanderABC.ipynb         [authoritative S0/S1 manuscript results]
```

The following are **not sources of the canonical paired replication**:

```text
09_Santander_ELAPS.ipynb       historical/exploratory retrospective run
10_Santander_Placebo.ipynb     legacy unverified placebo run
12_Santander_FLDC_90D.ipynb    exploratory 90-day analysis, not reported
```

---

## 8. Manuscript-to-notebook traceability

### 8.1 Main-text tables and figures

| Main-text item | Content | Notebook source(s) |
|---|---|---|
| Fig. 1 | Conceptual four-step ELAPS workflow | manuscript schematic; not generated by a model notebook |
| Table 1 | Related-work methodological comparison | literature synthesis; not a model experiment |
| Table 2 | VIB cohort structure motivating landmark choice | VIB preprocessing/cohort summaries; primarily `01_VIB_Dataprocessing.ipynb` |
| Fig. 2 | Conceptual A/B/C feature-window rules | manuscript schematic; not a model-result notebook |
| Table 3 | Study-design summary | manuscript synthesis across the canonical notebooks |
| Table 4 | VIB day-60 selected FLDC model, cohort-ordered robustness, ROS comparator | `05_VIB_FLDC_60D.ipynb` + `13_VIB60ABC.ipynb` |
| Table 5 | VIB same-population A/B/C decomposition | `13_VIB60ABC.ipynb`, using branch matrices from `05`, `14`, and `15` |
| Table 6 | Santander first-year A/B/C | `16_SantanderABC.ipynb` |
| Fig. 3(a) | VIB A/B/C AUC contrast | `13_VIB60ABC.ipynb` |
| Fig. 3(b) | Santander A/B/C AUC contrast | `16_SantanderABC.ipynb` |
| Table 7 | Nine-item audit battery | assembled from `01`, `02`, `03`, `05`, `13`, and `16` |
| Fixed-follow-up robustness statement | VIB H=180 A/B comparison | `18_VIB_FixedHorizon_H180_A60_B240.ipynb` |
| 20 split/refit robustness statement | repeated VIB A/B contrast | `17_VIB_Dwindow_Refit_20Seeds.ipynb` |

### 8.2 Nine-item ELAPS audit battery

| Audit item | VIB notebook source | Santander status/source |
|---|---|---|
| 1. Point-in-time traceability | `01_VIB_Dataprocessing.ipynb` | first-year cutoff rule only; no independent max-date trace in final audit |
| 2. Label permutation, 1,000 runs | `02_VIB_ELAPS.ipynb` | not run |
| 3. Window-length probe | `13_VIB60ABC.ipynb` | not run |
| 4. Dominant-feature / demographic ablation | `02_VIB_ELAPS.ipynb` | not run |
| 5. Placebo cutoff | `03_VIB_Placebo.ipynb` | canonical first-year result not verified; final status = Not run |
| 6. Same-population A/B/C decomposition | `13_VIB60ABC.ipynb` | `16_SantanderABC.ipynb` |
| 7. FLDC reconstruction | `05_VIB_FLDC_60D.ipynb` + retrospective reference from `02` | canonical first-year Branch A in `16`; comparison with older retrospective run is unpaired |
| 8. Calibration assessment | `05_VIB_FLDC_60D.ipynb` | Branch-A Brier/BSS in `16`; no canonical ECE reported |
| 9. Sampling robustness within FLDC | `05_VIB_FLDC_60D.ipynb` + `13_VIB60ABC.ipynb` | not run |

---

## 9. Supplementary Material-to-notebook mapping

This section is intentionally explicit so that each supplementary result can be traced back to its implementation.

| Supplement item | What it contains | Notebook source(s) |
|---|---|---|
| **S1** | Full 35-configuration VIB retrospective screen | `02_VIB_ELAPS.ipynb` |
| **S2** | Extended related-work comparison | literature synthesis; no model notebook |
| **S3** | VIB predictor definitions and point-in-time construction | `01_VIB_Dataprocessing.ipynb` + manuscript documentation |
| **S4** | Model and pipeline settings | consolidated documentation of settings implemented across `02`, `05`, `13`, and `16` |
| **S5** | Santander predictors and first-year window rules | canonical implementation in `16_SantanderABC.ipynb`; day-60 preparation supported by `11_Santander_FLDC_60D.ipynb` |
| **S6** | VIB subgroup monitoring | `05_VIB_FLDC_60D.ipynb` |
| **S7(a)** | Complementary VIB diagnostics | traceability: `01`; permutation: `02`; window-length probe: `13`; placebo: `03`; cohort-profile context: VIB preprocessing summaries |
| **S7(b)** | VIB ablation detail | `02_VIB_ELAPS.ipynb` |
| **S8** | Placebo-cutoff sensitivity | VIB: `03_VIB_Placebo.ipynb`; Santander canonical first-year row = Not run |
| **S9** | Metric definitions | manuscript/repository documentation; formulas used by evaluation notebooks |
| **S10** | VIB relationship-start cohort robustness | `13_VIB60ABC.ipynb` |
| **S11** | VIB landmark sensitivity | `04`, `05`, `06`, `07` |
| **S12** | Leading retrospective configurations | `02_VIB_ELAPS.ipynb` |
| **S13** | VIB retrospective ranking | `02_VIB_ELAPS.ipynb` |
| **S14** | Santander S0 top-K ranking | `16_SantanderABC.ipynb` |
| **S15(a)** | Santander S0/S1 cohort counts | `16_SantanderABC.ipynb` |
| **S15(b)** | Santander S0_full, S0_on_S1_holdout, S1_refit AUCs | `16_SantanderABC.ipynb` |
| **S15(c)** | Santander KS, Brier, BSS, Capture@10% | `16_SantanderABC.ipynb` |
| **S16** | Paired Santander AUC contrasts and DeLong tests | `16_SantanderABC.ipynb` |
| **S17** | Refitting effects on the identical S1 holdout | `16_SantanderABC.ipynb` |
| **S18** | VIB repeated split/refit variability | `17_VIB_Dwindow_Refit_20Seeds.ipynb` |
| **S19** | VIB fixed post-landmark follow-up, H=180 | `18_VIB_FixedHorizon_H180_A60_B240.ipynb` |

### Supplementary figures

| Figure | Content | Notebook source(s) |
|---|---|---|
| **Fig. S1** | Retrospective SHAP dependence for `AGE` | `02_VIB_ELAPS.ipynb` |
| **Fig. S2** | Retrospective SHAP interaction `AGE × COUNT_CA_ACCT` | `02_VIB_ELAPS.ipynb` |
| **Fig. S3** | VIB subgroup AUC and within-group Capture@20% | `05_VIB_FLDC_60D.ipynb` |
| **Fig. S4(a)** | Global SHAP — VIB retrospective XGBoost + ROS | `02_VIB_ELAPS.ipynb` |
| **Fig. S4(b)** | Global SHAP — canonical VIB day-60 FLDC XGBoost without resampling | `05_VIB_FLDC_60D.ipynb` |
| **Fig. S4(c)** | Earlier exploratory Santander historical SHAP model; not canonical S0/S1 | `09_Santander_ELAPS.ipynb` |
| **Fig. S5** | VIB landmark sensitivity | `04`, `05`, `06`, `07` |

---

## 10. Experiment groups

### 10.1 Retrospective model/sampling screen

Seven classifiers are crossed with five sampling configurations.

**Classifiers**

- Logistic Regression
- Random Forest
- Extra Trees
- AdaBoost
- Gradient-Boosted Trees
- XGBoost
- LightGBM

**Sampling configurations**

- None
- Random Oversampling (ROS)
- SMOTE
- Borderline-SMOTE
- ADASYN

Total:

```text
7 × 5 = 35 configurations
```

Resampling is applied only inside the training data / training folds.

The screen is exploratory. The retrospective holdout contributes to the historical reference designation, so those holdout results are descriptive rather than a clean model-selection estimate.

### 10.2 FLDC model selection

Only the two prespecified retrospective XGBoost configurations are re-evaluated inside FLDC:

```text
XGBoost, no resampling
XGBoost + ROS
```

Selection uses mean five-fold cross-validated AUC on the training partition. The final holdout does not participate in the selection decision.

### 10.3 Paired A/B/C inference

For A/B/C comparisons:

- holdout customers are identical across branches;
- labels are identical across branches;
- split assignments are identical across branches;
- each branch is fitted separately on its own feature matrix;
- AUC differences use paired customer-level bootstrap resampling;
- paired DeLong tests are used for AUC contrasts where reported.

The bootstrap intervals condition on the already fitted models unless the analysis explicitly refits models, as in notebook 17.

### 10.4 Cohort-ordered robustness

Customers are ordered by relationship start (`CLIENT_CREATE_DATE`):

```text
earliest 80%  -> training
latest 20%    -> testing
```

This test is not treated as a complete out-of-time validation. It changes cohort composition, prevalence, follow-up, and calendar ordering simultaneously.

### 10.5 Fixed-horizon robustness

The fixed-horizon analysis requires a common 180-day post-landmark observation period after day 60. It is intended to reduce heterogeneity in follow-up length when auditing the A/B window contrast.

### 10.6 Santander S0/S1 label sensitivity

- **S0:** customers with no later positive status are labeled 0 within the available first-year source, including customers with no later snapshot.
- **S1:** excludes S0 customers with no post-day-60 snapshot while preserving the retained labels and original train/test IDs.

The S1 experiment contains three distinct evaluations:

1. `S0_full` — original S0 models on the full S0 holdout;
2. `S0_on_S1_holdout` — saved S0 models evaluated only on retained S1 holdout IDs;
3. `S1_refit` — models retrained on retained S1 training IDs and evaluated on the same S1 holdout IDs.

This structure separates the effect of changing the evaluated population from the effect of refitting the model after the S1 exclusion.

---

## 11. Metrics reported

### Discrimination

- Area Under the ROC Curve (**AUC**)
- Kolmogorov–Smirnov statistic (**KS**)

### Capacity-constrained ranking

At `K = 10%, 20%, 30%`:

- **Precision@K** — positive share among the contacted top-K fraction;
- **Capture@K** — fraction of all observed positives contained in the top-K fraction;
- **Lift@K** — `Precision@K / holdout prevalence`.

### Probability quality

- **Brier score**;
- **Brier Skill Score (BSS)** relative to a constant holdout-prevalence forecast;
- **Expected Calibration Error (ECE)** using ten bins where reported.

### Threshold-0.5 classification metrics

- Recall
- Precision
- F1 score
- Accuracy

These threshold metrics are descriptive. A threshold of `0.5` is not the campaign-capacity decision rule, and resampling can alter the effective training prior.

---

## 12. Reproducibility rules

To reproduce the manuscript results correctly:

1. **Do not use customer identifiers as predictors.** IDs may be retained only for linkage, pairing, and audit checks.
2. **Do not use event dates, eligibility dates, or audit timestamps as predictors** unless the experiment explicitly defines an audit variable such as Branch-C tenure.
3. Fit encoders, imputers, category maps, and other data-dependent preprocessing **on training rows only**.
4. Treat previously unseen categorical levels as unknown rather than learning them from the holdout.
5. Apply ROS/SMOTE/ADASYN only to training data or within training folds.
6. Preserve the stated seeds and split definitions when reproducing reported values.
7. Keep the final holdout isolated from model and sampling-strategy selection.
8. For event-anchored features, use only records strictly before the applicable cutoff.
9. For the day-60 FLDC, use only information available before the common landmark and remove customers who already have the target card by the landmark.
10. For paired A/B/C experiments, keep IDs, labels, and split assignments identical across branches.
11. Refit each branch on its own feature matrix; do not reuse the fitted model from another branch.
12. Do not interpret Branch B or C as deployable day-60 forecasts; they are hindsight diagnostics.
13. Do not describe `D_deploy` as a pure leakage estimate.
14. Do not describe placebo-cutoff declines as causal estimates of leakage.
15. Preserve software/environment information printed by the notebooks alongside archived outputs.

---

## 13. Data availability

The manuscript uses two de-identified public datasets available through Kaggle, subject to their dataset-specific terms:

- **VIB Datathon dataset** — full ELAPS audit instance;
- **Santander Product Recommendation dataset** — first-year directional replication.

The repository does **not redistribute third-party datasets**. Users must obtain the data from the original source and update local/Colab paths before execution.

No additional live bank-system access or personally identifiable information is required by the manuscript workflow.

---

## 14. Interpretation cautions

The code should be interpreted with the following distinctions in mind:

- a strong retrospective AUC does not establish prospective deployment value;
- point-in-time feature correctness does not eliminate bias if the cutoff itself depends on future outcome status;
- `D_window` is a paired same-population performance contrast, not a causal estimate;
- `D_deploy` is a composite retrospective-to-deployment gap;
- Branches B and C are hindsight diagnostics;
- a higher AUC at a later landmark does not by itself imply a better operational decision time;
- ROS can preserve ranking while seriously degrading probability calibration;
- SHAP values describe fitted-model behavior and do not identify causal effects;
- subgroup analyses are monitoring diagnostics, not fairness guarantees;
- Santander monthly ownership status is only a proxy for exact opening time;
- Santander S1 improves follow-up observability but does not validate every negative label;
- the canonical Santander placebo diagnostic remains **Not run** because no verified first-year Fixed60D placebo result is available.

---

## 15. Installation

### 15.1 Clone the repository

```bash
git clone https://github.com/trang1981/ELAPS.git
cd ELAPS
```

### 15.2 Optional virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 15.3 Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 15.4 Launch Jupyter

```bash
jupyter notebook
```

The notebooks can also be run in Google Colab after mounting the required storage and updating input/output paths.

---

## 16. Quick reproduction checklist

For the principal manuscript claims, the minimum workflow is:

### VIB

```text
01  -> temporal preprocessing and traceability
02  -> retrospective reference, 35-config screen, permutation, ablation, retrospective SHAP
03  -> placebo diagnostic
05  -> canonical day-60 FLDC, calibration, subgroup monitoring, FLDC SHAP
14  -> Branch B construction
15  -> Branch C construction
13  -> paired A/B/C, window probe, cohort-ordered robustness, FLDC sampling robustness
17  -> 20 split/refit A/B runs
18  -> fixed H=180 follow-up A/B analysis
```

Use `04`, `06`, and `07` to reproduce the 30/90/180-day landmark sensitivity curve/table.

### Santander

```text
11  -> day-60 preparation/checking
16  -> canonical first-year S0/S1 A/B/C results
```

Treat `09`, `10`, and `12` as historical/exploratory notebooks, not as sources of the canonical Santander paired result.

---

## 17. Citation

Please cite the accompanying manuscript and the repository if you use this code.

Until the final journal record and DOI are available, the repository may be cited provisionally as:

```bibtex
@software{elaps2026repository,
  author = {Ngo Thi Thu Trang and Tran Thu Trang and Ha-Nam Nguyen},
  title  = {ELAPS: A Landmark-Based Audit Protocol for Label-Anchoring Bias in Retail-Banking Propensity Scoring},
  year   = {2026},
  url    = {https://github.com/trang1981/ELAPS},
  note   = {Code and reproducibility materials accompanying the ELAPS manuscript}
}
```

Replace or supplement this software entry with the final paper citation once the article is formally published.

---

## 18. Funding and responsible use

The manuscript reports **no external funding**.

The models estimate **observed product-adoption propensity**. They do not estimate the incremental causal effect of contacting a customer and must not be described as uplift or treatment-effect models.

---

## 19. Contact

**Tran Thu Trang**  
Faculty of Information Technology  
Dai Nam University, Hanoi, Vietnam  
Email: `trangtt@dainam.edu.vn`

For implementation or reproducibility questions, please open an issue in the repository.

