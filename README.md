# ELAPS — Auditing Outcome-Dependent Prediction Times in Retail-Banking Propensity Scoring

Code and reproducibility materials for:

> **Auditing Outcome-Dependent Prediction Times in Machine Learning: Paired Feature-Window Contrasts for Retail-Banking Propensity Scoring**
> Ngo Thi Thu Trang, Tran Thu Trang and Ha-Nam Nguyen. Submitted to *PeerJ Computer Science* (2026).

**Code archive (Zenodo):** https://doi.org/10.5281/zenodo.24302412
**Data:** [VIB-labelled Datathon dataset](https://www.kaggle.com/datasets/onamihoang/datathon-vib-vn) · [Santander Product Recommendation](https://www.kaggle.com/competitions/santander-product-recommendation) (not redistributed here)

---

## 1. What ELAPS is

ELAPS (*Event–Landmark Audit for Propensity Scoring*) is an **audit protocol, not a new classifier**. It checks how much an evaluation changes when the time at which predictors are built depends on the outcome: adopters summarised up to adoption, non-adopters up to the end of the extract. Every record can precede its own cutoff and still the two classes are observed under different rules.

On identical customers, labels and train/holdout splits, ELAPS refits the same learner under different feature-window rules and reports paired differences. These differences are **descriptive**, not causal effects or universal leakage estimates.

| Branch | Positives (adopters) | Negatives | Role |
|---|---|---|---|
| **A** | first 60 days after relationship start | first 60 days | deployment-aligned reference |
| **B** | up to the event e(c) | up to the cohort endpoint T(c) | event-anchored historical rule (diagnostic) |
| **B-sym** | up to e(c) | up to min(r(c) + P(c), T(c)), P(c) drawn from **training** positives' delays | duration-matched diagnostic |
| **B-sym-rs** | up to e(c) | as B-sym, P(c) drawn within the same relationship-start decile | entry-time matched diagnostic |
| **lag-k** | up to e(c) − k | entry-time matched to the shifted delays | probe for event-proximate information |
| **B→A** | model fitted on B, holdout scored with A features | | what a historically trained model gives at day 60 |

```text
D_window = AUC(B) − AUC(A) = [AUC(B-sym) − AUC(A)] + [AUC(B) − AUC(B-sym)] = D_sym + D_res
```

All branches must be built with **one feature pipeline** that reproduces the stored Branch A matrix exactly (Algorithm 1, step 5). Records satisfy `r(c) <= timestamp < cutoff(c)`; for Santander, Branch A includes the snapshot dated on day 60 (the target is defined strictly after day 60).

---

## 2. Repository structure

```text
ELAPS/
├── README.md
├── requirements.txt
├── Notebooks/          analysis notebooks (Google Colab, Python 3)
└── Simulation/         synthetic positive control and audit extensions
    ├── elaps_sim.py          simulation (S0 null, S1 pre-adoption ramp, S2 signal)
    ├── fig_sim.py            Figure 4 from the simulation output
    ├── results_xgb.csv       simulation output used in Table 10 / Figure 4
    ├── elaps_extensions.py   B-sym-rs, lag-k, B→A, paired bootstrap, balance report
    └── test_extensions.py    self-test of elaps_extensions on simulated data
```

| Notebook | Status | Produces |
|---|---|---|
| `01_VIB_Dataprocessing.ipynb` | canonical | VIB joins, timestamps, eligibility, point-in-time trace (Table 2, S3) |
| `02_VIB_ELAPS.ipynb` | context | historical reference (AUC ≈ 0.95), permutation, ablation, SHAP (S7(a), S12, Figs S1, S2, S4) |
| `03_VIB_Placebo.ipynb` | context | historical VIB placebo cutoffs (S8) |
| `04`, `06`, `07` `_VIB_FLDC_*` | descriptive | landmark sensitivity at days 30, 90, 180 (Fig S5) |
| `05_VIB_FLDC_60D.ipynb` | canonical | day-60 cohort and **Branch A**; probability quality, subgroups (Table 4, S6, Fig S3) |
| `08_VIB_LightGBM_ROS.ipynb` | auxiliary | retrospective check, not reported |
| `09_Santander_ELAPS.ipynb` | canonical | Santander age/tenure ablation (S21) and Santander probes: B-sym-rs, lag curve, B→A (last cell) |
| `10_Santander_Placebo.ipynb`, `12_Santander_FLDC_90D.ipynb` | exploratory | not reported |
| `11_Santander_FLDC_60D.ipynb` | preparation | Santander first-year day-60 cohort (S0/S1) |
| `13_VIB60ABC.ipynb` | **canonical** | VIB day-60 A / B-sym / B (Table 5), Branch B rebuild, VIB probes, four-learner R50 (Table 7, S18), H180 (Table 6, S19); guide in the first cell |
| `14_VIB_FLDC_60D_B.ipynb`, `15_VIB_FLDC_60D_C.ipynb` | legacy | earlier Branch B / C matrices from the historical extract; not used in reported results |
| `16_SantanderABC.ipynb` | supplementary | Santander Branch C stress test (S20) |
| `17_VIB_Dwindow_Refit_20Seeds.ipynb`, `18_VIB_FixedHorizon_H180_A60_B240.ipynb` | superseded | earlier refits and H180 A/B, kept for transparency |

Notebook comments are partly in Vietnamese. Paths in the notebooks point to Google Drive; edit them before running elsewhere.

---

## 3. How to reproduce

**VIB-labelled data**

1. `01` → `05`: build the cohorts and Branch A.
2. `13_VIB60ABC.ipynb`, Section A, in order:
   - *RUN_AUDIT CHO VIB* with `RUN = ()` loads the data and feature builder; with `RUN = ('H180', 'ADMIN')` it also reproduces B-sym-rs, the lag curve and B→A for both cohorts.
   - *BƯỚC 1* and *BƯỚC 1b* rebuild Branch B with the Branch A pipeline and write `VIB_FLDC_60D_B_REBUILT.csv` (check: AUC 0.922914, 0 mismatches against stored Branch A).
   - *VIB60ABC — A / B / B-sym + R50 + H=180* and `run_all()`: four-learner R50.
   - *BƯỚC 3: TABLE 5*: Table 5.
3. `13_VIB60ABC.ipynb`, Section B: H180 A / B-sym / B.

**Santander**

1. `11_Santander_FLDC_60D.ipynb`: first-year cohort (S0, S1).
2. `09_Santander_ELAPS.ipynb`: age/tenure ablation and the Santander probes.

**Synthetic positive control** (no third-party data needed)

```bash
cd Simulation
python elaps_sim.py --reps 5 --n 60000 --model xgb --out results_xgb.csv   # about 9 min on 2 CPU cores
python fig_sim.py results_xgb.csv figure4.png
python test_extensions.py . .                                                # self-test of elaps_extensions
```

**Environment**

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt duckdb
```

Real-data runs used Google Colab (Python 3.13, scikit-learn 1.6.1, XGBoost 3.4.1, LightGBM 4.6.0, NumPy 2.4.6, pandas 2.2.3, SciPy 1.16.3; 2 CPU cores, 13.6 GB RAM). Other library versions can change results in the last digits.

---

## 4. Main results

Compare branches **within** a cohort; differences between cohorts are not paired. Intervals are 95% customer-level paired bootstrap intervals (1,000 resamples, seed 42) with fitted models held fixed.

**VIB day-60, common administrative endpoint** — 94,453 customers; holdout 18,891 (875 events); XGBoost; split seed 42 (Table 5)

| | AUC / ΔAUC | 95% interval |
|---|---:|---|
| A | 0.910242 | 0.9018 – 0.9188 |
| B-sym | 0.923461 | 0.9159 – 0.9307 |
| B | 0.922914 | 0.9153 – 0.9303 |
| B − A | 0.012672 | 0.0078 – 0.0178 |
| B-sym − A | 0.013219 | 0.0083 – 0.0181 |
| B − B-sym | −0.000547 | −0.0025 – 0.0014 |

- 40.1% of negative B-sym pseudo-cutoffs are truncated at the endpoint (KS 0.189); B-sym-rs: 17.2% (KS 0.175), AUC 0.923381.
- Lag curve, contrast with A: lag-7 0.0107 (0.0061 – 0.0158), lag-30 0.0067 (0.0019 – 0.0115), lag-60 0.0026 (−0.0042 – 0.0096).
- B→A: AUC 0.905515, BSS 0.188 vs 0.216 for A.
- **Feature-pipeline equivalence.** The legacy Branch B matrix (`VIB_FLDC_60D.csv`) came from the historical extract and used different aggregation rules (no lower bound at relationship start, non-adopter window ending one month earlier, different transaction mode, latest-balance rule, missing-level coding and age). It gave AUC 0.960688 and B − A = 0.050446. Rebuilt with the Branch A pipeline, Branch B loses 0.0378 AUC (0.0312 – 0.0448).

**VIB H180, equal 180-day follow-up** — 31,149 customers; holdout 6,230 (291 events) (Table 6)

| | Value |
|---|---|
| AUC A / B-sym / B | 0.891002 / 0.903415 / 0.906161 |
| B − A | 0.0152 (0.0045 – 0.0264) |
| B − B-sym | 0.0027 (−0.0032 – 0.0083) |
| Lag-30 − A | 0.0019 (−0.0075 – 0.0103) |
| B→A | AUC 0.8858, BSS 0.086 vs 0.140 for A |

No negative pseudo-cutoff is truncated in this cohort.

**Four learners × 50 split/refit runs, VIB day-60** (Table 7, S18)

| Learner | mean B − A | mean B − B-sym (runs > 0) |
|---|---:|---:|
| Logistic regression | 0.0111 | −0.0048 (0/50) |
| Random forest | 0.0187 | −0.0018 (9/50) |
| LightGBM | 0.0120 | −0.0014 (5/50) |
| XGBoost | 0.0116 | −0.0016 (4/50) |

B − A and B-sym − A are positive in all 50 runs for every learner. XGBoost Kendall tau-b: 0.780 (A, B), 0.812 (A, B-sym).

**Santander first year** — S0: 153,399 customers, 916 events; S1: 139,067 customers, 916 events (Table 8)

| | A | B-sym | B | B − A |
|---|---:|---:|---:|---|
| S0 | 0.885617 | 0.897839 | 0.972259 | 0.0866 (0.0712 – 0.1035) |
| S1 | 0.876453 | 0.894911 | 0.976517 | 0.1001 (0.0847 – 0.1160) |

- Removing tenure (`antiguedad`) halves B − A on S0 (0.0866 → 0.0422); without age and tenure B − A = 0.0499 (0.0343 – 0.0644) and B-sym − A includes zero (S21).
- Lag one month − A: 0.0004 (−0.0107 – 0.0132); B→A: AUC 0.8417 and BSS −14.99 vs 0.8856 and 0.0165 for A.
- In four learners × 50 refits, B − A is positive in all runs (S17).

**Synthetic positive control** (Table 10, Figure 4): with no signal, Branch B reaches AUC 0.99 under equal follow-up; B-sym and B-sym-rs restore 0.50 only when follow-up is fixed; a 30-day pre-adoption ramp survives duration matching but disappears at lag 30.

---

## 5. Settings

| Setting | Value |
|---|---|
| Split | stratified 80:20, seed 42 |
| XGBoost (canonical) | 300 trees, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, no oversampling |
| Logistic regression (R50) | `StandardScaler`, `class_weight="balanced"`, `lbfgs`, `max_iter=2000` |
| Random forest (R50) | 200 trees |
| LightGBM (R50) | 500 trees, `num_leaves=31`, `max_depth=6`, `learning_rate=0.05` |
| R50 | split/refit seeds 42–91; B-sym redrawn from each split's training positives |
| Bootstrap | 1,000 customer-level paired resamples, seed 42 |
| BSS reference | holdout prevalence (real data); training prevalence (simulation) |
| B-sym draw | seed 42 **and** fixed customer order (draws depend on row order) |

Identifiers, eligibility dates, outcomes and audit timestamps are never predictors. Preprocessing is fitted on training data only, and each branch is refitted with the same learner and split.

---

## 6. Audit coverage (manuscript Table 9)

| Diagnostic | VIB | Santander |
|---|---|---|
| Same IDs, labels and splits across branches | yes | yes |
| One feature pipeline reproducing stored Branch A | yes (0 mismatches) | yes (0 mismatches) |
| Paired bootstrap | day-60 and H180 | S0, S1 |
| Four learners × 50 refits | day-60 | S0 |
| Fixed-horizon (H180) cohort | yes | no |
| B-sym duration balance | day-60 and H180 | no |
| B-sym-rs, lag curve, B→A | day-60 and H180 | S0 |
| Age/tenure ablation | not applicable (no tenure predictor) | S0 |
| Independent source-timestamp trace | yes | no |

A diagnostic that was not run is not evidence that the risk is absent.

---

## 7. Data and responsible use

- Both datasets are public on Kaggle and are **not redistributed**; download them and update the paths in the notebooks.
- The VIB-labelled data are a third-party upload whose sampling is undocumented; they are not a representative sample of the bank's customers.
- Santander targets are a monthly ownership-status proxy, not verified card-opening dates.
- The models estimate observed adoption propensity. Nothing here establishes causal effects, marketing uplift or production readiness.

---

## 8. Citation and contact

```bibtex
@software{ngo2026elaps_code,
  author    = {Ngo Thi Thu Trang and Tran Thu Trang and Ha-Nam Nguyen},
  title     = {ELAPS: Code for "Auditing Outcome-Dependent Prediction Times in Machine Learning:
               Paired Feature-Window Contrasts for Retail-Banking Propensity Scoring"},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24302412},
  url       = {https://doi.org/10.5281/zenodo.24302412}
}
```

The licence is stated on the Zenodo record.

**Corresponding author:** Ha-Nam Nguyen, International School, Vietnam National University, Hanoi — namnhvn@vnu.edu.vn
**Code contact:** Tran Thu Trang, Faculty of Information Technology, Dai Nam University, Hanoi — trangtt@dainam.edu.vn
