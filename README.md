# ELAPS
Official implementation of ELAPS: an event-based leakage-aware propensity-scoring framework for credit-card cross-selling in retail banking.
## Experiments and Evaluation Metrics

This repository contains the complete experimental workflow used to develop, audit, and evaluate the ELAPS framework. The experiments cover data preprocessing, conventional propensity modeling, leakage diagnostics, deployment-faithful cohort construction, horizon sensitivity, external validation, explainability, fairness, and cross-model robustness.

---

## 1. VIB Dataset Experiments

### 1.1 `VIB_Dataprocessing.ipynb`

This notebook prepares the internal VIB retail-banking dataset for subsequent modeling experiments.

#### Main procedures

- Integration of customer-level data from multiple banking sources
- Customer identifier reconciliation
- Date parsing and temporal consistency checks
- Removal of invalid records
- Detection of cases in which the credit-card opening date precedes the customer relationship start date
- Definition of customer relationship start date
- Identification of the first credit-card opening event
- Identification of the last observed transaction date
- Construction of customer-specific temporal cutoffs
- Event-anchored feature construction
- Aggregation of account, loan, transaction, and digital-banking variables
- Encoding of categorical variables
- Missing-value handling
- Construction of the final modeling table
- Target-label generation
- Class-distribution analysis

#### Label definition

A customer is labeled as positive when at least one credit card is opened during the study period.

A customer is labeled as negative when no credit card is opened during the study period.

For adopter \(c\), the feature cutoff is the first credit-card opening date:

\[
\tau(c)=t_c^{\mathrm{card}}.
\]

For non-adopter \(c\), the feature cutoff is the last observed transaction date in the study year:

\[
\tau(c)=t_c^{\mathrm{last\ transaction}}.
\]

#### Main output

- Final customer-level analytical dataset
- Event-anchored predictors
- Binary credit-card adoption target
- Dataset dimensions
- Positive and negative class counts
- Credit-card adoption prevalence

---

### 1.2 `VIB_ELAPS.ipynb`

This notebook contains the main ELAPS experiment on the VIB dataset.

#### Data-splitting and validation procedures

- Stratified 80/20 train-test split
- Five-fold cross-validation
- Random-state control
- Holdout evaluation
- Class-imbalance handling
- Random Oversampling
- Comparison of alternative sampling strategies where applicable

#### Machine-learning models evaluated

- Logistic Regression
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- XGBoost
- LightGBM

#### Sampling methods evaluated

- No resampling
- Random Oversampling
- SMOTE
- Borderline-SMOTE
- ADASYN

#### Main selected configuration

- XGBoost
- Random Oversampling
- Stratified holdout evaluation

#### Discrimination metrics

- Area Under the ROC Curve — AUC
- Receiver Operating Characteristic curve — ROC
- Kolmogorov-Smirnov statistic — KS
- Cross-validation mean AUC
- Cross-validation standard deviation
- DeLong comparison between selected models, where implemented
- Bootstrap confidence intervals, where implemented

#### Classification metrics

- Accuracy
- Precision
- Recall
- F1-score
- Specificity, where reported
- Confusion matrix
- True positives
- False positives
- True negatives
- False negatives
- Classification report

#### Probability-quality metrics

- Brier score
- Calibration analysis
- Predicted-probability distribution
- Calibration curve, where generated

#### Ranking and campaign metrics

- Precision@10%
- Precision@20%
- Precision@30%
- Capture@10%
- Capture@20%
- Capture@30%
- Lift@10%, also denoted Lift@D1
- Lift@20%
- Lift@30%
- Number of adopters captured in each score band
- Decile-level response rate
- Cumulative gains
- Cumulative lift

#### Explainability analyses

- SHAP global feature importance
- SHAP summary plot
- Mean absolute SHAP value
- Feature ranking
- SHAP dependence analysis, where generated
- Direction of feature effects
- Interpretation of customer demographics
- Interpretation of account ownership
- Interpretation of account balances
- Interpretation of loan activity
- Interpretation of digital-banking activity
- Interpretation of transaction behavior

#### Leakage-control procedures

- Exclusion of post-adoption information
- Per-customer temporal cutoff enforcement
- Feature-level point-in-time correctness
- Label-shuffling sanity check
- Dominant-feature inspection
- Feature-family ablation

#### Ablation experiments

- Full feature set
- Removal of `COUNT_CA_ACCT`
- Removal of the complete account-feature group
- Comparison of AUC after feature removal
- Comparison of lift after feature removal
- Comparison of feature importance after ablation

#### Observation-window sensitivity

- Construction of tenure at cutoff
- Comparison with and without tenure-related information
- AUC comparison
- KS comparison
- F1 comparison
- Brier-score comparison
- Lift@10% comparison

---

### 1.3 `VIB_LIGHTGBM_ROS.ipynb`

This notebook tests whether the leakage-audit conclusions are robust to the choice of learning algorithm.

#### Main experiment

- LightGBM classifier
- Random Oversampling
- Stratified train-test split
- Holdout evaluation
- Comparison with XGBoost + Random Oversampling

#### Cross-model performance ladder

The notebook evaluates LightGBM under progressively stricter leakage-control configurations.

- L0: conventional full-cohort evaluation
- L1: deployed event-anchored ELAPS configuration
- L2: removal of dominant account-feature information
- L3: placebo-cutoff evaluation
- L4: FLDC-60D deployment evaluation

#### Metrics

- AUC
- Mean AUC across placebo seeds
- Standard deviation of AUC
- KS statistic, where computed
- Accuracy
- Precision
- Recall
- F1-score
- Brier score
- Precision@10%
- Capture@10%
- Lift@10%
- Base rate
- Cohort size
- Cross-model comparison with XGBoost

---

### 1.4 `VIB_PLACEBO.ipynb`

This notebook implements the placebo-cutoff experiment used to diagnose performance inflation caused by class-dependent observation-window geometry.

#### Main procedures

- Construction of placebo temporal cutoffs
- Equalization or randomization of customer observation windows
- Recalculation of temporal features
- Repeated model estimation across random seeds
- Comparison with conventional ELAPS performance
- Quantification of performance loss after removing window-related information

#### Repeated-seed evaluation

- Multiple random seeds
- Mean metric across seeds
- Standard deviation across seeds
- Stability assessment

#### Metrics

- AUC for each seed
- Mean AUC
- Standard deviation of AUC
- Precision@10%
- Mean Precision@10%
- Standard deviation of Precision@10%
- Capture@10%
- Capture@20%
- Capture@30%, where reported
- Lift@10%
- Mean Lift@10%
- Standard deviation of Lift@10%
- Cohort size
- Positive count
- Base rate
- Difference from the conventional full-cohort result

---

### 1.5 `VIB30cutoffngay.ipynb`

This notebook constructs a Forward-Looking Deployment Cohort using a 30-day observation window.

#### FLDC design

- Relationship start used as cohort anchor
- Fixed first 30 days used for feature construction
- Customers adopting within the first 30 days excluded
- Target defined as adoption strictly after day 30
- Same feature horizon applied to positive and negative customers
- Deployment-faithful train-test evaluation

#### Metrics

- Cohort size
- Positive count
- Negative count
- Base rate
- AUC
- Cross-validation AUC
- Cross-validation standard deviation
- KS statistic
- Accuracy
- Precision
- Recall
- F1-score
- Brier score
- Confusion matrix
- Precision@10%
- Capture@10%
- Lift@10%
- Additional top-decile or cumulative-gain measures
- SHAP feature importance, where generated
- Fairness metrics, where generated

---

### 1.6 `VIB60cutoffngay.ipynb`

This notebook implements the principal deployment-faithful evaluation used in the manuscript.

#### FLDC-60D design

- Fixed 60-day feature-observation window
- Exclusion of customers adopting during the first 60 days
- Target defined as credit-card adoption strictly after day 60
- Equal feature-window duration for all customers
- Stratified model development and holdout testing
- XGBoost + Random Oversampling
- Comparison with conventional event-anchored ELAPS

#### Cohort metrics

- Number of eligible customers
- Number of excluded first-window adopters
- Positive count
- Negative count
- Post-window adoption prevalence
- Train-set size
- Test-set size

#### Predictive-performance metrics

- AUC
- Five-fold cross-validation AUC
- Cross-validation standard deviation
- KS statistic
- Accuracy
- Precision
- Recall
- F1-score
- Brier score
- Confusion matrix
- Classification report

#### Ranking metrics

- Precision@10%
- Capture@10%
- Lift@10%
- Precision@20%, where reported
- Capture@20%
- Lift@20%, where reported
- Precision@30%, where reported
- Capture@30%
- Cumulative gains
- Score-decile performance

#### Explainability

- SHAP global importance
- SHAP summary plot
- Ranking of deployment-window predictors
- Comparison with full-cohort SHAP importance
- Assessment of whether dominant account and tenure effects persist

#### Fairness evaluation

Fairness is evaluated by sex and age group.

##### Groups

- Male
- Female
- Age below 30
- Age 30–45
- Age 46 and above

##### Group-level metrics

- Group sample size
- Group positive count, where reported
- Group base rate
- Group-specific AUC
- Group-specific Brier score
- True-positive rate at the classification threshold
- Selection rate at the classification threshold
- True-positive rate among the top 20% scored customers
- Group-level calibration quality, where generated

##### Fairness-summary metrics

- Disparate-impact ratio
- Selection-rate gap
- True-positive-rate gap
- AUC gap
- Brier-score gap
- Top-20% capture or TPR gap
- Comparison across sex groups
- Comparison across age groups

---

### 1.7 `VIB_cutoff90ngay.ipynb`

This notebook evaluates the FLDC framework using a 90-day feature-observation window.

#### Main procedures

- Fixed first 90 days for feature construction
- Exclusion of customers adopting during the first 90 days
- Strictly future target after day 90
- Cohort reconstruction
- Model retraining
- Holdout evaluation
- Cross-validation
- Comparison with the 30-day, 60-day, and 180-day horizons

#### Metrics

- Cohort size
- Positive count
- Negative count
- Base rate
- AUC
- Cross-validation mean AUC
- Cross-validation standard deviation
- KS statistic
- Accuracy
- Precision
- Recall
- F1-score
- Brier score
- Confusion matrix
- Precision@10%
- Capture@10%
- Lift@10%
- SHAP feature importance, where generated
- Fairness metrics, where generated

---

### 1.8 `VIB180cutoffngay.ipynb`

This notebook evaluates a longer 180-day FLDC observation horizon.

#### Main procedures

- Fixed first 180 days for feature construction
- Exclusion of all first-window adopters
- Strictly future adoption target after day 180
- Evaluation of the trade-off between longer observation and cohort retention
- Comparison with shorter FLDC horizons

#### Metrics

- Cohort size
- Number and proportion of excluded customers
- Positive count
- Negative count
- Base rate
- AUC
- KS statistic
- Accuracy
- Precision
- Recall
- F1-score
- Brier score
- Confusion matrix
- Precision@10%
- Capture@10%
- Lift@10%
- Cross-validation performance, where generated

---

## 2. Santander Dataset Experiments

### 2.1 `Santander_ELAPS.ipynb`

This notebook evaluates the transferability of ELAPS on the public Santander Product Recommendation dataset.

#### Main procedures

- Santander data preprocessing
- Customer-month ordering
- Product-adoption event identification
- Credit-card product target construction
- Event-anchored customer cutoff construction
- Customer-level feature aggregation
- Class-imbalance treatment
- XGBoost + Random Oversampling
- Holdout evaluation
- SHAP interpretation
- External comparison with VIB

#### Dataset metrics

- Number of customers
- Number of positive customers
- Number of negative customers
- Credit-card adoption prevalence
- Number of input features

#### Predictive metrics

- AUC
- KS statistic
- Accuracy
- Precision
- Recall
- F1-score
- Brier score
- Confusion matrix
- Classification report
- Cross-validation AUC, where generated

#### Ranking metrics

- Precision@10%
- Capture@10%
- Lift@10%
- Precision@20%
- Capture@20%
- Lift@20%, where reported
- Precision@30%
- Capture@30%
- Lift@30%, where reported
- Number of adopters captured in top-ranked segments

#### Explainability

- SHAP global importance
- SHAP summary plot
- Mean absolute SHAP value
- Feature ranking
- Analysis of age
- Analysis of customer tenure
- Analysis of product holdings
- Analysis of account behavior

---

### 2.2 `Santander_PLACEBO.ipynb`

This notebook transfers the placebo-cutoff leakage diagnostic to the Santander dataset.

#### Main procedures

- Construction of equalized or randomized placebo cutoffs
- Removal of class-dependent observation-window advantage
- Repeated experiments across random seeds
- Comparison with the original Santander ELAPS result
- Evaluation of cross-dataset reproducibility of leakage inflation

#### Metrics

- AUC for each seed
- Mean AUC
- Standard deviation of AUC
- Precision@10%
- Mean Precision@10%
- Standard deviation of Precision@10%
- Capture@10%
- Capture@20%
- Capture@30%
- Mean Capture@30%
- Standard deviation of Capture@30%
- Lift@10%
- Mean Lift@10%
- Standard deviation of Lift@10%
- Base rate
- Cohort size
- Difference relative to original event-anchored ELAPS

---

### 2.3 `Santander60cutoff.ipynb`

This notebook evaluates the FLDC framework on Santander using the same 60-day horizon selected for VIB.

#### FLDC design

- Fixed first 60 days for feature construction
- Exclusion of customers adopting within the initial 60-day period
- Target defined as adoption strictly after day 60
- Equal observation-window geometry
- XGBoost + Random Oversampling
- Deployment-faithful holdout evaluation
- Direct comparison with original Santander ELAPS

#### Predictive metrics

- Cohort size
- Positive count
- Negative count
- Base rate
- AUC
- KS statistic
- Accuracy
- Precision
- Recall
- F1-score
- Brier score
- Confusion matrix
- Classification report
- Precision@10%
- Capture@10%
- Lift@10%
- Additional cumulative-gain metrics

#### Performance comparison

- Difference in AUC between original ELAPS and FLDC-60D
- Difference in KS
- Difference in accuracy
- Difference in precision
- Difference in recall
- Difference in F1-score
- Difference in Brier score
- Difference in Precision@10%
- Difference in Capture@10%
- Difference in Lift@10%
- Change in cohort construction
- Change in target definition
- Change in class prevalence

#### Fairness evaluation

##### Sex groups

- Male
- Female

##### Age groups

- Age below 30
- Age 30–45
- Age 46 and above

##### Group metrics

- Sample size
- Base rate
- AUC
- Brier score
- TPR at the classification threshold
- Selection rate at the classification threshold
- TPR among the top 20% scored customers

##### Fairness-summary metrics

- Disparate-impact ratio
- Selection-rate gap
- TPR gap
- AUC gap
- Brier-score gap
- Top-20% TPR gap
- Sex-group comparison
- Age-group comparison

---

### 2.4 `Santander90cutoff.ipynb`

This notebook evaluates whether Santander FLDC performance changes under a longer 90-day observation window.

#### Main procedures

- Fixed first 90 days for feature construction
- Exclusion of first-window adopters
- Strictly future adoption target
- Cohort reconstruction
- Model retraining
- Comparison with Santander FLDC-60D
- Horizon-sensitivity assessment

#### Metrics

- Cohort size
- Positive count
- Negative count
- Base rate
- AUC
- KS statistic
- Accuracy
- Precision
- Recall
- F1-score
- Brier score
- Confusion matrix
- Precision@10%
- Capture@10%
- Lift@10%
- Fairness metrics by sex and age, where generated
- Differences relative to the 60-day horizon

---

## 3. Cross-Experiment Analyses

### 3.1 Horizon-Sensitivity Analysis

The VIB FLDC notebooks jointly evaluate four onboarding horizons:

- 30 days
- 60 days
- 90 days
- 180 days

The Santander notebooks evaluate at least:

- 60 days
- 90 days

#### Comparison dimensions

- Eligible cohort size
- Number of excluded first-window adopters
- Positive count
- Base rate
- AUC
- KS statistic
- Accuracy
- Precision
- Recall
- F1-score
- Brier score
- Precision@10%
- Capture@10%
- Lift@10%
- Cross-validation stability
- Fairness variation
- Deployment-population coverage

---

### 3.2 Leakage-Audit Performance Ladder

The repository supports a graded performance ladder that progressively excludes potential sources of optimistic performance inflation.

#### Ladder levels

- **L0:** Conventional full-cohort model
- **L1:** Event-anchored ELAPS with feature-level point-in-time correctness
- **L2:** Dominant-feature or account-feature ablation
- **L3:** Placebo-cutoff evaluation
- **L4:** FLDC-60D deployment-faithful evaluation

#### Metrics compared across levels

- Base rate
- AUC
- Precision@10%
- Capture@10%
- Lift@10%
- Cohort definition
- Inflation channels excluded
- XGBoost performance
- LightGBM performance
- Cross-model consistency

---

### 3.3 Cross-Model Robustness

The principal conclusions are evaluated using:

- XGBoost + Random Oversampling
- LightGBM + Random Oversampling

#### Cross-model metrics

- AUC at each performance-ladder level
- Lift@10% at each ladder level
- Difference between XGBoost and LightGBM
- Stability of placebo results
- Stability of FLDC results
- Consistency of the estimated deployable performance range

---

### 3.4 Statistical Validation

Across the notebooks, the following statistical procedures are used where applicable:

- Stratified holdout testing
- Five-fold cross-validation
- Mean cross-validation AUC
- Cross-validation standard deviation
- Bootstrap confidence intervals
- DeLong test for AUC comparison
- Repeated placebo experiments
- Mean across random seeds
- Standard deviation across random seeds
- Label-shuffling sanity check

---

## 4. Complete Metric Inventory

The complete set of metrics generated across the repository includes:

### Dataset and cohort statistics

- Total number of customers
- Number of positive cases
- Number of negative cases
- Base rate
- Train-set size
- Test-set size
- Number of excluded first-window adopters
- Deployment-cohort retention rate
- Number of features

### Discrimination metrics

- AUC
- ROC curve
- KS statistic
- Cross-validation AUC
- Mean AUC
- Standard deviation of AUC
- Bootstrap confidence interval
- DeLong test statistic and p-value, where implemented

### Classification metrics

- Accuracy
- Precision
- Recall
- Sensitivity
- Specificity, where implemented
- F1-score
- Confusion matrix
- True positives
- False positives
- True negatives
- False negatives
- Classification report

### Probability and calibration metrics

- Brier score
- Calibration curve
- Group-specific Brier score
- Predicted-probability distribution

### Ranking and campaign metrics

- Precision@10%
- Precision@20%
- Precision@30%
- Capture@10%
- Capture@20%
- Capture@30%
- Lift@10%
- Lift@20%
- Lift@30%
- Decile response rate
- Cumulative gains
- Cumulative lift
- Number of positives captured in top-ranked segments

### Fairness metrics

- Group sample size
- Group base rate
- Group-specific AUC
- Group-specific Brier score
- TPR at a fixed classification threshold
- Selection rate at a fixed classification threshold
- TPR among the top 20% scored customers
- Disparate-impact ratio
- Selection-rate gap
- TPR gap
- AUC gap
- Brier-score gap
- Top-20% TPR gap

### Explainability outputs

- SHAP global importance
- SHAP summary plot
- Mean absolute SHAP value
- Feature ranking
- Feature-effect direction
- SHAP dependence plots, where generated

### Leakage-audit indicators

- Label-shuffling AUC
- Performance difference after feature ablation
- Performance difference after account-feature removal
- Performance difference after placebo cutoff
- Performance difference after FLDC construction
- Performance difference across observation horizons
- Cross-model agreement in the performance ladder
