ELAPS — Auditing Outcome-Dependent Prediction Times in Retail-Banking Propensity Scoring

Reproducibility materials for the manuscript:

Auditing Outcome-Dependent Prediction Times in Machine Learning: Paired Feature-Window Contrasts for Retail-Banking Propensity Scoring
Ngo Thi Thu Trang, Tran Thu Trang, and Ha-Nam Nguyen
Submitted to PeerJ Computer Science (2026).


Data sources: VIB-labelled Datathon dataset · Santander Product Recommendation

Contents
What ELAPS investigates
The A / B-sym / B design
Datasets and cohorts
Repository structure
Where each result is produced
Principal results
Notebook guide
Execution order
Manuscript and Supplement mapping
Model and statistical settings
Reproduction invariants and integrity checks
Audit coverage
Data provenance and responsible use
Installation
Citation and contact
1. What ELAPS investigates

ELAPS (Event–Landmark Audit for Propensity Scoring) is an empirical audit protocol, not a new predictive classifier. It studies how a machine-learning evaluation changes when the time at which predictors are constructed depends on the subsequently observed outcome.

In retrospective banking records, a card adopter may be summarized up to the recorded adoption event, whereas a non-adopter may be summarized up to an administrative endpoint. Every source record can precede its assigned cutoff and therefore be point-in-time valid at the feature level; nevertheless, class-dependent cutoffs can create evaluation conditions that do not exist at a common prospective decision time.

The study measures paired differences between separately refitted pipelines under different cutoff rules. These differences are descriptive; they are not causal effects, universal leakage magnitudes, or estimates of marketing uplift.

Research questions

RQ1 — Paired window contrast. With identical customers, labels, and train/holdout assignments, how much does the feature-window rule change discrimination and campaign ranking?
RQ2 — Symmetrization and follow-up. What remains of this contrast when negative pseudo-durations are drawn from positive training-event delays, and when outcome follow-up is fixed?
RQ3 — Robustness and transport of direction. Does the direction of the contrast persist across learner families, 50 split/refit realizations, and a second public banking source with a different event proxy?

Four audit stages

Historical reference — event-anchored features and validity of contributing source timestamps.
Day-60 landmark — eligibility, feature availability, target, and follow-up defined from a common prediction time (the Forward-Looking Deployment Cohort, FLDC).
Paired A / B-sym / B comparison — pipelines refitted on the same IDs, labels, and split under three cutoff rules.
Evaluation audit — AUC, KS, top-K ranking, probability quality, paired bootstrap uncertainty, refit sensitivity, and limitations.
2. The A / B-sym / B design

For customer c: r(c) relationship start; e(c) first recorded credit-card adoption (VIB) or first later card-positive monthly status (Santander proxy); L = 60 days landmark; T(c) admissible observation endpoint of the cohort.

Feature records must satisfy

text
r(c) <= timestamp(record) < cutoff(c)

Santander exception. Santander Branch A includes the monthly snapshot dated on day 60. Because the Santander target is defined strictly after day 60 and customers card-positive through day 60 are excluded, this snapshot cannot contain the target event. VIB event streams use the strict-before rule above.

Branch	Positives	Negatives	Role
A — fixed day 60	First 60 relationship days	First 60 relationship days	Deployment-aligned reference (not prospectively validated)
B-sym — pseudo-cutoff	Event cutoff e(c)	min(r(c) + P(c), T(c)), with P(c) sampled with replacement from positive training event delays e(c) − r(c) in the same split	Symmetrized diagnostic
B — event-anchored	Event cutoff e(c)	Cohort endpoint T(c)	Historical diagnostic
Holdout-positive delays never contribute to the B-sym sampling distribution.
B-sym is generated with a fixed seed and a fixed customer order, because the sampled values depend on the order in which customers are drawn.
B-sym names the construction; it does not guarantee matched realized durations, because capping at T(c) can break balance. The realized-duration KS statistic and truncation rate are therefore reported with it (Section 6.1).
B and B-sym rely on outcome-dependent cutoffs; they are diagnostics and cannot be scored at day 60 as deployable models.

Reported contrasts (same holdout, separately fitted models):

text
D_window   = AUC(B)     - AUC(A)
D_matched  = AUC(B-sym) - AUC(A)
D_endpoint = AUC(B)     - AUC(B-sym)

D_window = D_matched + D_endpoint      # exact arithmetic identity

D_matched and D_endpoint are descriptive pipeline contrasts, not separately identified causal mechanisms.

Branch C (supplementary only). Branch C adds TENURE_AT_CUTOFF to Branch B. Because this variable is derived from the outcome-dependent cutoff, it can nearly encode the labelling mechanism. C is reported only as a near-label leakage stress test (Supplementary Table S20) and is not part of the primary comparison.

3. Datasets and cohorts
3.1 VIB-labelled dataset

Public Kaggle dataset labelled Datathon_VIB_VN. The analyses use 18 predictors (Supplementary Table S3). CUSTOMER_NUMBER is used only for linkage and paired evaluation; COUNT_CREDITCARD defines the outcome and is not a predictor.

Cohort	Customers	Positives	Train / holdout	Interpretation
Historical 2019 reference	127,460	19,714 (15.47%)	101,968 / 25,492	Retrospective reference
Day-60 FLDC	94,453	4,374 (4.63%)	75,562 / 18,891	Day-60 features; follow-up to the end of 2019
Day-60 + H180	31,149	1,454 (4.67%)	24,919 / 6,230	Complete follow-up to relationship day 240

The day-60 FLDC excludes adopters on or before day 60 and follows remaining customers to the common administrative endpoint, so later entrants have shorter follow-up. H180 addresses this by requiring observation to day 240 and defining the outcome as first card opening in (day 60, day 240]. Because H180 also changes eligibility, differences between the two cohorts are not a controlled follow-up-only effect.

3.2 Santander Product Recommendation

Monthly product-status snapshots rather than exact card-opening timestamps. Target: first observed card-positive monthly status strictly after day 60 and before the first relationship anniversary. Customers card-positive in any available snapshot through day 60 are excluded. The analyses use 20 non-target predictors (Supplementary Table S5), including age and antiguedad unless removed in the ablation.

Cohort	Customers	Positives	Train (positives)	Holdout (positives)
S0	153,399	916	122,719 (733)	30,680 (183)
S1	139,067	916	111,283 (733)	27,784 (183)
S0: customers without an observed positive status after day 60 are labelled 0, including those without a later snapshot.
S1: keeps only S0 customers with at least one later first-year snapshot (removes 14,332 negatives, keeps all 916 positives, preserves labels and split membership). S1 improves follow-up observability but does not guarantee complete-year follow-up.
4. Repository structure

File names below are exactly as stored in the repository.

text
ELAPS/
├── README.md
└── Notebooks/
    ├── 01_VIB_Dataprocessing.ipynb
    ├── 02_VIB_ELAPS.ipynb
    ├── 03_VIB_Placebo.ipynb
    ├── 04_VIB_FLDC_30D.ipynb
    ├── 05_VIB_FLDC_60D.ipynb
    ├── 06_VIB_FLDC_90D.ipynb
    ├── 07_VIB_FLDC_180D.ipynb
    ├── 08_VIB_LightGBM_ROS.ipynb
    ├── 09_Santander_ELAPS.ipynb
    ├── 10_Santander_Placebo.ipynb
    ├── 11_Santander_FLDC_60D.ipynb
    ├── 12_Santander_FLDC_90D.ipynb
    ├── 13_VIB60ABC.ipynb
    ├── 14_VIB_FLDC_60D_B.ipynb
    ├── 15_VIB_FLDC_60D_C.ipynb
    ├── 16_SantanderABC.ipynb
    ├── 17_VIB_Dwindow_Refit_20Seeds.ipynb
    └── 18_VIB_FixedHorizon_H180_A60_B240.ipynb

Naming notes. Notebook names reflect the development history of the project. In particular, 13_VIB60ABC.ipynb now contains the A / B-sym / B analyses, not only the earlier A/B/C design. Paths are case-sensitive (Notebooks/).

5. Where each result is produced

The current paired analyses are concentrated in two canonical notebooks:

13_VIB60ABC.ipynb — VIB day-60 A / B-sym / B, the B-sym realized-duration audit, H180 A / B-sym / B, four-learner R50, cohort-ordered and oversampling sensitivities, and the VIB Branch C stress test.
09_Santander_ELAPS.ipynb — Santander S0/S1 A / B-sym / B, paired bootstrap contrasts, four-learner R50 with Kendall tau-b, and the age / antiguedad ablation.

Upstream inputs come from 01 (VIB preprocessing), 05 (VIB Branch A), 14 and 15 (VIB Branch B and C matrices), and 11 (Santander first-year cohort preparation).

Result	Manuscript / Supplement	Notebook
VIB day-60 A / B-sym / B, 5,000 paired bootstrap	Table 5	13_VIB60ABC.ipynb
VIB B-sym realized-duration audit	S7(b)	13_VIB60ABC.ipynb
VIB H180 A / B-sym / B	Table 6, S19	13_VIB60ABC.ipynb
VIB four-learner R50	Table 7, S18	13_VIB60ABC.ipynb
VIB cohort-ordered and ROS sensitivity	Table 4, S10	13_VIB60ABC.ipynb (Branch A from 05)
VIB Branch C stress test	S20	13_VIB60ABC.ipynb (matrices from 14, 15)
Santander S0/S1 A / B-sym / B	Table 8, S15(a–b), S16	09_Santander_ELAPS.ipynb
Santander four-learner R50 + Kendall tau-b	S17(a–b)	09_Santander_ELAPS.ipynb
Santander age / antiguedad ablation (12 refits)	Figure 3, S21(a–c)	09_Santander_ELAPS.ipynb
Santander Branch C stress test	S20	16_SantanderABC.ipynb
Figure 3 (five analysis settings)	Figure 3	13 (VIB panels) + 09 (Santander panels)
6. Principal results

Compare results within a cohort; AUC differences between cohorts are not paired comparisons. Reported differences are computed from unrounded AUCs, so differences recomputed from the six-decimal AUCs shown here can deviate by one unit in the last digit.

6.1 VIB day-60 A / B-sym / B — 13_VIB60ABC.ipynb

Shared holdout: 18,891 customers, 875 positives; XGBoost without oversampling; split seed 42.

Branch or contrast	AUC / ΔAUC	95% paired bootstrap interval
A — fixed day 60	0.910242	[0.9017, 0.9185]
B-sym — pseudo-cutoff	0.923461	[0.9155, 0.9308]
B — event-anchored	0.960688	[0.9548, 0.9661]
B − A (D_window)	0.050446	[0.0427, 0.0582]
B-sym − A (D_matched)	0.013219	[0.0084, 0.0182]
B − B-sym (D_endpoint)	0.037227	[0.0304, 0.0444]

Intervals use 5,000 paired, customer-level, non-stratified bootstrap resamples of the holdout with the fitted models held fixed; they do not include refit variability (see Section 6.3).

Branch A additionally reports KS 0.6733, Brier score 0.0346, BSS 0.2163, Lift@10% 6.4084, Capture@10% 0.6411, Capture@20% 0.8343, and Capture@30% 0.9097. An XGBoost + ROS comparator gives similar ranking but a Brier skill score of about −1.40, because oversampling shifts the training class prior.

B-sym realized-duration audit (S7(b))

Before capping, negative pseudo-durations match training-positive delays (two-sample KS = 0.005).
36,155 of 90,079 negatives (40.1%) have pseudo-cutoffs capped at the administrative endpoint.
After capping, the median realized duration is 106 days for negatives and 136 days for positives (KS = 0.189).
B-sym in the primary cohort is therefore not duration-balanced, and D_matched should be read with this in mind.

Sensitivity to the B-sym draw. The reported B-sym model uses seed 42 and the customer order fixed in 13_VIB60ABC.ipynb. A different customer order produces a different pseudo-duration draw; one such draw gives B-sym AUC 0.921921 (vs 0.923461). Variation due to resampling the pseudo-durations is covered systematically by the R50 study, in which B-sym is rebuilt for every split (Section 6.3).

6.2 VIB fixed H180 — 13_VIB60ABC.ipynb

Shared holdout: 6,230 customers, 291 positives; follow-up complete to day 240; 1,000 paired bootstrap resamples.

Quantity	Value
AUC A / B-sym / B	0.891002 / 0.903415 / 0.906161
B − A	0.015158 [0.004543, 0.026398]
B-sym − A	0.012412 [0.002897, 0.023012]
B − B-sym	0.002746 [−0.003178, 0.008347]
BSS A / B-sym / B	0.139893 / 0.187516 / 0.215304
Capture@10% A / B-sym / B	0.539519 / 0.584192 / 0.618557
Negative B-sym cutoffs capped	0

In this cohort, where no negative pseudo-cutoff is capped, the B − B-sym interval includes zero. Because H180 changes eligibility as well as follow-up, its smaller B − A contrast is not attributable to the fixed horizon alone.

6.3 Four learners × 50 split/refit runs — 13 (VIB) and 09 (Santander)

Fifty stratified 80:20 splits (seeds 42–91). For every split, all three branches are refitted and the B-sym pseudo-durations are redrawn from that split's positive training customers. Ranges across runs are empirical refit ranges, not confidence intervals for the mean.

Learner	VIB mean B − A	Santander S0 mean B − A
Logistic regression	0.062945	0.031812
Random forest	0.064160	0.154028
LightGBM	0.053272	0.096721
XGBoost	0.052219	0.092896
VIB: for all four learners, all 50/50 runs show positive values of all three contrasts.
Santander S0: all 50/50 runs show positive values of each contrast.
Santander R50 uses all 20 predictors, including antiguedad; the effect of removing it is examined separately in Section 6.5.

Full decompositions and empirical ranges: S17(a–b) and S18.

6.4 Santander S0/S1 — 09_Santander_ELAPS.ipynb
Cohort	AUC A	AUC B-sym	AUC B
S0	0.885617	0.897839	0.972259
S1	0.876453	0.894911	0.976517

Paired contrasts, 1,000 bootstrap resamples (95% intervals):

Contrast	S0	S1
B − A	0.086642 [0.071196, 0.103525]	0.100064 [0.084668, 0.115961]
B-sym − A	0.012222 [0.001313, 0.024441]	0.018458 [0.006499, 0.030354]
B − B-sym	0.074420 [0.063273, 0.086647]	0.081606 [0.068869, 0.094768]

Within each cohort, all branches share IDs, labels, and the original train/holdout membership. Santander provides directional cross-dataset evidence under a monthly status proxy.

6.5 Santander S0 age / antiguedad ablation — 09_Santander_ELAPS.ipynb

All branches refitted on the S0 split with identical preprocessing and XGBoost settings; 1,000 shared bootstrap resamples.

Feature set	AUC A	AUC B-sym	AUC B	B − A (95% interval)
All 20	0.885617	0.897839	0.972259	0.0866 [0.0712, 0.1035]
Without age	0.872851	0.887020	0.970561	0.0977 [0.0795, 0.1160]
Without antiguedad	0.885729	0.891982	0.927917	0.0422 [0.0292, 0.0550]
Without both	0.873982	0.882427	0.923855	0.0499 [0.0343, 0.0644]
Removing antiguedad reduces B − A by about 0.0445, roughly half of its full-feature value.
Without both variables, B − A = 0.0499 and its interval excludes zero; B-sym − A = 0.0084 [−0.0014, 0.0186] includes zero.
The ablation was performed on the S0 split only; it was not repeated for S1 or within R50.
6.6 Historical reference and Branch C stress test
The VIB historical models reach AUC ≈ 0.95. This is a retrospective reference, not a day-60 deployment estimate. The common-holdout historical comparison (n = 25,492; 3,943 positives) is reported in S12 (02_VIB_ELAPS.ipynb).
Branch C (S20 only): VIB day-60 AUC A 0.9102, B 0.9607, C 0.9669; Santander C 0.998651 (S0) and 0.998368 (S1). These values show how explicit cutoff information can approximate the labelling mechanism; they are not deployment results.
7. Notebook guide
#	Notebook	Status	Role in the current manuscript
01	01_VIB_Dataprocessing.ipynb	Canonical	VIB source joins, timestamps, eligibility, point-in-time trace; Table 2, S3
02	02_VIB_ELAPS.ipynb	Canonical (context)	Historical reference, 1,000-run label permutation, feature ablation, SHAP; S7(a), S12, Figs S1, S2, S4
03	03_VIB_Placebo.ipynb	Canonical (context)	Historical VIB placebo cutoffs; S8
04	04_VIB_FLDC_30D.ipynb	Descriptive	Day-30 landmark; Fig S5
05	05_VIB_FLDC_60D.ipynb	Canonical	Day-60 FLDC and Branch A; probability quality; subgroup monitoring; Table 4, S6, Fig S3
06	06_VIB_FLDC_90D.ipynb	Descriptive	Day-90 landmark; Fig S5
07	07_VIB_FLDC_180D.ipynb	Descriptive	Day-180 landmark (distinct from the day-60 + H180 cohort); Fig S5
08	08_VIB_LightGBM_ROS.ipynb	Auxiliary	Retrospective LightGBM + ROS check; not used for reported tables
09	09_Santander_ELAPS.ipynb	Canonical — Santander	S0/S1 A / B-sym / B, bootstrap, R50 + Kendall tau-b, age/antiguedad ablation; Table 8, S5, S15–S17, S21
10	10_Santander_Placebo.ipynb	Exploratory	Earlier placebo exploration; not used for reported results
11	11_Santander_FLDC_60D.ipynb	Preparation	Santander first-year day-60 cohort preparation
12	12_Santander_FLDC_90D.ipynb	Exploratory	Day-90 exploration; not used for reported results
13	13_VIB60ABC.ipynb	Canonical — VIB	Day-60 A / B-sym / B, S7(b) duration audit, H180 A / B-sym / B, four-learner R50, cohort-ordered and ROS sensitivity, Branch C; Tables 4–7, S10, S18–S20
14	14_VIB_FLDC_60D_B.ipynb	Builder	VIB Branch B feature matrix, used by 13
15	15_VIB_FLDC_60D_C.ipynb	Builder	VIB Branch C matrix for the S20 stress test
16	16_SantanderABC.ipynb	Supplementary	Santander A/B/C; source of the Santander Branch C stress test (S20)
17	17_VIB_Dwindow_Refit_20Seeds.ipynb	Superseded	Earlier 20-seed XGBoost A/B refits, retained for transparency; replaced by R50 in 13
18	18_VIB_FixedHorizon_H180_A60_B240.ipynb	Superseded	Earlier H180 A/B analysis, retained for transparency; the A / B-sym / B version is in 13

Sections of 09_Santander_ELAPS.ipynb that pre-date the first-year S0/S1 design are exploratory and are not used for any reported result.

8. Execution order
VIB
text
01_VIB_Dataprocessing.ipynb
        ↓
02_VIB_ELAPS.ipynb              (historical context)
03_VIB_Placebo.ipynb            (historical context)
        ↓
05_VIB_FLDC_60D.ipynb           → Branch A
04 / 06 / 07                    (landmark sensitivity, Fig S5)
        ↓
14_VIB_FLDC_60D_B.ipynb         → Branch B matrix
15_VIB_FLDC_60D_C.ipynb         → Branch C matrix (S20 only)
        ↓
13_VIB60ABC.ipynb               → Tables 4–7, S7(b), S10, S18–S20
Santander
text
11_Santander_FLDC_60D.ipynb    → first-year cohort preparation
        ↓
09_Santander_ELAPS.ipynb        → Table 8, S15–S17, S21
16_SantanderABC.ipynb           → Branch C stress test (S20)
9. Manuscript and Supplement mapping
Main manuscript
Item	Content	Source
Figure 1	Four-stage ELAPS workflow	Conceptual
Table 1	Prior temporal-evaluation guidance vs ELAPS	Literature
Table 2	VIB historical / day-60 / H180 denominators	01, 05, 13
Table 3	Cutoffs, outcomes, interpretation	Conceptual
Figure 2	A / B-sym / B feature windows	Conceptual
Table 4	VIB Branch A, cohort-ordered and oversampling sensitivity	05 + 13
Table 5	VIB day-60 A / B-sym / B, paired contrasts	13
Table 6	VIB H180 A / B-sym / B	13
Table 7	VIB four-learner R50 means	13
Table 8	Santander S0 and S1 A / B-sym / B	09
Figure 3	AUC across five analysis settings	13 + 09
Table 9	Diagnostic status matrix	01, 02, 03, 05, 13, 09
Supplementary Material
Item	Content	Source
S3	VIB 18 predictors and point-in-time rules	01
S4	Model settings, split seed, B-sym order, uncertainty settings	13, 09
S5	Santander 20 predictors, event proxy, window rules	09
S6	VIB subgroup monitoring	05
S7(a)	Permutation, point-in-time, window, ablation, placebo, onboarding diagnostics	01, 02, 03, 13
S7(b)	VIB B-sym realized-duration matching and truncation	13
S8	Historical VIB placebo; Santander first-year placebo not run	03
S9	Metric definitions	Documentation
S10	VIB cohort-ordered sensitivity	13
S12	Historical common-holdout comparison	02
S15(a–b)	Santander S0/S1 denominators and three-branch metrics	09
S16	Santander paired contrasts and bootstrap intervals	09
S17(a–b)	Santander four-learner R50 and Kendall tau-b	09
S18	VIB four-learner R50 decomposition	13
S19	VIB H180 A / B-sym / B	13
S20(a–d)	Branch C stress test	VIB 13 (+ 15); Santander 16
S21(a–c)	Santander age / antiguedad ablation, 12 refits	09
Figs S1, S2, S4	Retrospective VIB SHAP	02
Fig S3	VIB Branch A subgroup monitoring	05
Fig S5	VIB landmark sensitivity (descriptive)	04–07

Supplementary table numbering follows the manuscript and is not contiguous.

10. Model and statistical settings
Setting	Specification
Main split	Stratified 80:20, seed 42
XGBoost (canonical)	300 trees, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8; no oversampling
Santander XGBoost	tree_method="hist", max_bin=256
Logistic regression (R50)	StandardScaler, class_weight="balanced", solver="lbfgs", max_iter=2000
Random forest (R50)	200 trees; no oversampling
LightGBM (R50)	500 trees, num_leaves=31, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8
R50	Split/refit seeds 42–91; B-sym redrawn per split
Paired bootstrap	VIB day-60: 5,000 resamples (seed 42); VIB H180 and Santander: 1,000 resamples; predictions fixed
VIB B-sym	Seed 42 with the customer order fixed in 13_VIB60ABC.ipynb
Software	Python; NumPy, pandas, SciPy, scikit-learn, XGBoost, LightGBM, imbalanced-learn, SHAP; DuckDB for Santander snapshot processing

Settings not listed here (including learner random states) are recorded in the notebooks and in S4. In LightGBM, row subsampling via subsample is active only when subsample_freq > 0; see the notebook for the exact configuration.

Inference. Bootstrap intervals apply the same resampled customer indices to all branches and are conditional on the fitted pipelines; they exclude training variability. R50 quantifies split/refit variability separately; its 2.5–97.5% range is not a confidence interval.

Metrics. ROC AUC and Kolmogorov–Smirnov (KS); Precision@K, Capture@K, Lift@K at fixed contact fractions; Brier score and Brier skill score against the holdout-prevalence forecast; Kendall tau-b for ranking concordance in the Santander R50 study. Definitions are in S9.

11. Reproduction invariants and integrity checks

Invariants

Customer identifiers are used for linkage and paired scoring only, never as predictors.
The target, target-derived proxies, eligibility dates, audit timestamps, and card-status fields are excluded from predictors.
Each branch is built from records admissible under its own cutoff, with identical predictor definitions across A, B-sym, and B.
Imputers, encoders, and other learned preprocessing are fitted on training data only.
Each learner is refitted separately per branch, with the same hyperparameters and split.
B-sym pseudo-durations are drawn only from positive training delays within each split.
The random seed and customer order are fixed for exact B-sym reproduction.
Realized negative durations and truncation rates are reported after capping.
VIB uses strict-before cutoffs; the Santander day-60 snapshot rule is handled explicitly (Section 2).
Retrospective and Branch C outputs are not reported as prospective performance.

Integrity checks

Cohort counts and splits match Table 2 and S15(a).
Within each cohort, all branches share evaluation IDs, labels, and splits.
D_window = D_matched + D_endpoint holds to floating-point tolerance before rounding.
VIB day-60 B-sym AUC = 0.923461 under the documented seed, customer order, and package versions (other library versions can change results in the last digits).
VIB day-60 B-sym truncation = 40.1%; H180 truncation = 0.
Santander full-feature ablation refits reproduce the S15(b) AUCs (absolute differences < 1e-6).
Santander B − A without age and antiguedad = 0.0499, with an interval excluding zero.
12. Audit coverage

Status matrix corresponding to manuscript Table 9.

Diagnostic	VIB	Santander
Matched A / B-sym / B IDs, labels, splits	Reported	Reported
Paired AUC bootstrap	Day-60: 5,000; H180: 1,000	S0/S1: 1,000
Four-learner 50 split/refit	Reported	Reported
Fixed H180 follow-up	Reported	Not reported
B-sym realized-duration balance	Day-60 and H180	Not reported
Independent maximum-source-timestamp trace	Reported	Not reported
Age/tenure exclusion ablation	Not applicable as a matched tenure ablation	S0 split
Kendall tau-b ranking concordance	Not reported	S0 R50
Conditional B-sym avoiding capping	Not run	Not run
Train on B, score with day-60 features	Not run	Not run

Historical VIB permutation, placebo, and retrospective diagnostics are provided as context in the Supplement. A Santander first-year placebo analysis was not run.

13. Data provenance and responsible use
VIB-labelled source: a publicly accessible third-party Kaggle dataset. The analysed fields use coded customer identifiers; no direct identifiers such as names, telephone numbers, or account numbers were observed in them. Users should review the dataset page for its terms before reuse.
Santander source: public Kaggle competition data, subject to the competition rules.
No raw data are redistributed in this repository. Download both datasets from their original pages and update the data paths in each notebook (local environment or Google Colab).
Identifiers are used only for linkage, ordering, and audit.
Interpretation: the models estimate observed adoption/status propensity. Neither the paired AUC contrasts nor SHAP explanations establish causality, marketing uplift, fairness, profit, or production readiness.
14. Installation

Download and extract the archive from the Zenodo record, then from the extracted folder:

bash
cd <extracted-folder>

python -m venv .venv
source .venv/bin/activate              # Linux / macOS
# .venv\Scripts\Activate.ps1           # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn xgboost lightgbm \
    imbalanced-learn shap duckdb jupyter

jupyter notebook

Notebooks can also be executed non-interactively, for example:

bash
jupyter nbconvert --to notebook --execute Notebooks/01_VIB_Dataprocessing.ipynb

The notebooks print package versions alongside their outputs; keep these together with seeds and saved predictions when archiving results.

15. Citation and contact

If you use this code, please cite the code archive and, once published, the article.

bibtex
@software{ngo2026elaps_code,
  author    = {Ngo Thi Thu Trang and Tran Thu Trang and Ha-Nam Nguyen},
  title     = {ELAPS: Code for "Auditing Outcome-Dependent Prediction Times in
               Machine Learning: Paired Feature-Window Contrasts for
               Retail-Banking Propensity Scoring"},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24297946},
  url       = {https://doi.org/10.5281/zenodo.24297946}
}

@misc{ngo2026elaps,
  author = {Ngo Thi Thu Trang and Tran Thu Trang and Ha-Nam Nguyen},
  title  = {Auditing Outcome-Dependent Prediction Times in Machine Learning:
            Paired Feature-Window Contrasts for Retail-Banking Propensity Scoring},
  year   = {2026},
  note   = {Manuscript submitted to PeerJ Computer Science}
}

The licence for the code is stated on the Zenodo record.

Corresponding author: Ha-Nam Nguyen — International School, Vietnam National University, Hanoi — namnhvn@vnu.edu.vn

Code and reproducibility contact: Tran Thu Trang — Faculty of Information Technology, Dai Nam University, Hanoi — trangtt@dainam.edu.vn

Summary. Even when every individual input is time-admissible, assigning feature cutoffs by the recorded outcome can materially alter retrospective performance estimates. The A / B-sym / B gaps reported here are design-specific, descriptive comparisons; they are not causal leakage estimates and do not imply that event-anchored models can be deployed at day 60.
