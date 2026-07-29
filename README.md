# ELAPS
Official implementation of ELAPS: an event-based leakage-aware propensity-scoring framework for credit-card cross-selling in retail banking.
# ELAPS: Event-Based Leakage-Aware Propensity Scoring

This repository contains the experimental implementation of the study:

> **ELAPS: A Leakage-Audited Machine Learning Framework for Credit-Card Cross-Sell Propensity Prediction in Retail Banking**

ELAPS is designed to evaluate credit-card cross-sell propensity models under deployment-faithful conditions. The framework combines event-based feature construction, leakage auditing, placebo diagnostics, Forward-Looking Deployment Cohort evaluation, external validation, model interpretation, and fairness analysis.

---

## 1. Research Objective

The main objective of this study is to develop and evaluate a leakage-aware machine learning framework for credit-card cross-sell propensity prediction in retail banking.

Conventional propensity models may report overly optimistic performance when feature windows, customer lifecycles, or cohort definitions differ between adopters and non-adopters. ELAPS addresses this problem through a structured evaluation process that distinguishes feature-level point-in-time correctness from cohort-level performance inflation.

The study has five main objectives:

1. Construct customer-level predictors using event-anchored temporal cutoffs.
2. Prevent post-adoption information from entering the feature set.
3. Diagnose potential performance inflation using label-shuffling, feature ablation, observation-window sensitivity, and placebo-cutoff experiments.
4. estimate deployment-faithful performance through the Forward-Looking Deployment Cohort, or FLDC.
5. Assess the transferability of the framework using both an internal VIB banking dataset and the Santander Product Recommendation dataset.

The main experimental components are:

- event-based feature construction;
- point-in-time correct preprocessing;
- XGBoost and LightGBM modeling;
- random oversampling and alternative imbalance treatments;
- feature and feature-family ablation;
- label-shuffling sanity checks;
- placebo-cutoff analysis;
- FLDC evaluation at multiple observation horizons;
- SHAP-based interpretation;
- fairness evaluation by sex and age group;
- cross-dataset validation;
- cross-model performance-ladder analysis.

---

## 2. Repository Structure

```text
.
├── README.md
├── requirements.txt
└── Notebooks/
    ├── VIB_Dataprocessing.ipynb
    ├── VIB_ELAPS.ipynb
    ├── VIB_LIGHTGBM_ROS.ipynb
    ├── VIB_PLACEBO.ipynb
    ├── VIB30cutoffngay.ipynb
    ├── VIB60cutoffngay.ipynb
    ├── VIB_cutoff90ngay.ipynb
    ├── VIB180cutoffngay.ipynb
    ├── Santander_ELAPS.ipynb
    ├── Santander_PLACEBO.ipynb
    ├── Santander60cutoff.ipynb
    └── Santander90cutoff.ipynb
```

### Notebook descriptions

| Notebook | Purpose |
|---|---|
| `VIB_Dataprocessing.ipynb` | Cleans and integrates the internal VIB data, constructs event-anchored features, defines labels, and produces the analytical dataset. |
| `VIB_ELAPS.ipynb` | Runs the main ELAPS experiment on VIB, including model comparison, resampling, evaluation, SHAP analysis, ablation, and ranking metrics. |
| `VIB_LIGHTGBM_ROS.ipynb` | Evaluates LightGBM with Random Oversampling and reproduces the cross-model performance ladder. |
| `VIB_PLACEBO.ipynb` | Runs placebo-cutoff experiments to diagnose observation-window-related performance inflation. |
| `VIB30cutoffngay.ipynb` | Runs FLDC with a fixed 30-day observation window. |
| `VIB60cutoffngay.ipynb` | Runs the main FLDC-60D evaluation used as the default deployment-faithful configuration. |
| `VIB_cutoff90ngay.ipynb` | Runs FLDC with a fixed 90-day observation window. |
| `VIB180cutoffngay.ipynb` | Runs FLDC with a fixed 180-day observation window. |
| `Santander_ELAPS.ipynb` | Applies the ELAPS pipeline to the Santander Product Recommendation dataset. |
| `Santander_PLACEBO.ipynb` | Runs placebo-cutoff analysis on Santander. |
| `Santander60cutoff.ipynb` | Runs Santander FLDC with a fixed 60-day observation window. |
| `Santander90cutoff.ipynb` | Runs Santander FLDC with a fixed 90-day observation window. |

---

## 3. Installation

### 3.1 Recommended environment

- Python 3.10 or later
- Jupyter Notebook, JupyterLab, or Google Colab
- At least 8 GB RAM recommended
- Additional memory may be required for the full Santander dataset

### 3.2 Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

Replace `<repository-url>` and `<repository-name>` with the actual GitHub repository information.

### 3.3 Create a virtual environment

Using `venv`:

```bash
python -m venv elaps-env
```

Activate the environment.

On Windows:

```bash
elaps-env\Scripts\activate
```

On macOS or Linux:

```bash
source elaps-env/bin/activate
```

### 3.4 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

A suitable `requirements.txt` is:

```text
numpy>=1.24
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
imbalanced-learn>=0.11
shap>=0.45
matplotlib>=3.8
seaborn>=0.13
joblib>=1.3
tqdm>=4.66
notebook>=7.0
jupyterlab>=4.0
ipykernel>=6.0
openpyxl>=3.1
xlrd>=2.0
statsmodels>=0.14
optuna>=3.6
```

If the notebooks were originally executed in Google Colab, some packages may already be preinstalled. The installation cell should still be executed to ensure package compatibility.

---

## 4. Running the Notebooks

### 4.1 Run locally with Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open the `Notebooks` directory and run the notebooks in the sequence described below.

### 4.2 Run with JupyterLab

```bash
jupyter lab
```

### 4.3 Run with Google Colab

Each notebook can also be uploaded to Google Colab.

Recommended steps:

1. Open Google Colab.
2. Upload or open the selected notebook from GitHub.
3. Mount Google Drive if the input data are stored there.
4. Update all data paths before execution.
5. Run the notebook from the first cell to the final cell.
6. Save the executed notebook or exported result tables.

Example for mounting Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4.4 Recommended execution order

#### VIB experiments

Run the notebooks in the following order:

```text
1. VIB_Dataprocessing.ipynb
2. VIB_ELAPS.ipynb
3. VIB_LIGHTGBM_ROS.ipynb
4. VIB_PLACEBO.ipynb
5. VIB30cutoffngay.ipynb
6. VIB60cutoffngay.ipynb
7. VIB_cutoff90ngay.ipynb
8. VIB180cutoffngay.ipynb
```

#### Santander experiments

```text
1. Santander_ELAPS.ipynb
2. Santander_PLACEBO.ipynb
3. Santander60cutoff.ipynb
4. Santander90cutoff.ipynb
```

The VIB and Santander pipelines are independent after their respective data preparation steps.

---

## 5. Reproducing the Results

### 5.1 General reproducibility procedure

To reproduce the reported results:

1. Install the required Python packages.
2. Place the authorized input files in the expected data directory.
3. Update all local or Google Drive paths inside the notebooks.
4. Execute each notebook from top to bottom without skipping cells.
5. Use the same random seeds specified in the notebooks.
6. Preserve the same train-test split, preprocessing steps, feature definitions, and model parameters.
7. Record the final output tables and figures generated by each notebook.

### 5.2 Main experimental sequence

The study follows the sequence below:

```text
Raw data
   ↓
Data cleaning and temporal validation
   ↓
Event-anchored feature construction
   ↓
Conventional full-cohort evaluation
   ↓
Feature ablation and observation-window sensitivity
   ↓
Label-shuffling and placebo-cutoff diagnostics
   ↓
FLDC evaluation at multiple horizons
   ↓
Cross-model robustness
   ↓
External validation on Santander
   ↓
Fairness and explainability analysis
```

### 5.3 Reproducing the main VIB results

Run:

```text
VIB_Dataprocessing.ipynb
VIB_ELAPS.ipynb
VIB_PLACEBO.ipynb
VIB60cutoffngay.ipynb
VIB_LIGHTGBM_ROS.ipynb
```

These notebooks reproduce the main components of the study:

- event-anchored ELAPS performance;
- XGBoost + Random Oversampling results;
- LightGBM + Random Oversampling robustness results;
- label-shuffling sanity checks;
- placebo-cutoff analysis;
- dominant-feature and feature-family ablation;
- FLDC-60D deployment-faithful evaluation;
- SHAP feature importance;
- ranking metrics;
- fairness metrics.

### 5.4 Reproducing horizon sensitivity

Run:

```text
VIB30cutoffngay.ipynb
VIB60cutoffngay.ipynb
VIB_cutoff90ngay.ipynb
VIB180cutoffngay.ipynb
```

The notebooks compare:

- cohort size;
- excluded early adopters;
- positive count;
- base rate;
- AUC;
- KS statistic;
- accuracy;
- precision;
- recall;
- F1-score;
- Brier score;
- Precision@10%;
- Capture@10%;
- Lift@10%;
- cross-validation stability, where reported.

The 60-day configuration is used as the default FLDC setting in the manuscript because it preserves a larger eligible cohort while achieving discrimination close to the 90-day configuration.

### 5.5 Reproducing the performance ladder

The leakage-audit performance ladder is constructed from progressively stricter evaluation configurations:

| Level | Configuration | Main purpose |
|---|---|---|
| L0 | Conventional full-cohort evaluation | Baseline estimate before stricter leakage controls |
| L1 | Event-anchored ELAPS | Removes feature-level post-adoption leakage |
| L2 | Dominant-feature or account-feature ablation | Tests dependence on highly informative account variables |
| L3 | Placebo-cutoff evaluation | Reduces class-dependent observation-window advantage |
| L4 | FLDC-60D | Estimates deployment-faithful future-adoption performance |

The XGBoost and LightGBM notebooks provide the cross-model comparison for this ladder.

### 5.6 Reproducing Santander external validation

Run:

```text
Santander_ELAPS.ipynb
Santander_PLACEBO.ipynb
Santander60cutoff.ipynb
Santander90cutoff.ipynb
```

These notebooks reproduce:

- event-anchored Santander ELAPS results;
- placebo-cutoff results across repeated seeds;
- FLDC-60D performance;
- FLDC-90D horizon sensitivity;
- ranking performance;
- SHAP interpretation;
- fairness metrics by sex and age group.

### 5.7 Main evaluation metrics

The repository reports the following metrics where applicable.

#### Discrimination

- Area Under the ROC Curve
- ROC curve
- Kolmogorov-Smirnov statistic
- cross-validation mean AUC
- cross-validation standard deviation
- bootstrap confidence intervals
- DeLong comparison

#### Classification

- accuracy
- precision
- recall
- F1-score
- specificity
- confusion matrix
- classification report

#### Probability quality

- Brier score
- calibration curve
- group-specific Brier score

#### Ranking and campaign performance

- Precision@10%
- Precision@20%
- Precision@30%
- Capture@10%
- Capture@20%
- Capture@30%
- Lift@10%
- Lift@20%
- Lift@30%
- cumulative gain
- cumulative lift
- score-decile response rate

#### Leakage diagnostics

- label-shuffling AUC
- feature-ablation performance change
- placebo-cutoff performance change
- observation-window sensitivity
- FLDC performance change
- cross-model agreement

#### Fairness

- group sample size
- group base rate
- group-specific AUC
- group-specific Brier score
- true-positive rate at a fixed threshold
- selection rate at a fixed threshold
- true-positive rate among the top 20%
- disparate-impact ratio
- TPR gap
- selection-rate gap
- AUC gap
- Brier-score gap

#### Explainability

- SHAP global feature importance
- mean absolute SHAP value
- SHAP summary plots
- feature-effect direction
- feature ranking

### 5.8 Reproducibility limitations

Exact numerical reproduction of the internal VIB results requires:

- access to the same confidential source data;
- identical preprocessing rules;
- identical date fields and customer identifiers;
- identical software versions;
- identical random seeds;
- sufficient computational resources.

Small numerical differences may occur across operating systems, package versions, or hardware configurations, particularly for tree-based models and parallel execution.

---

## 6. Data Information

### 6.1 VIB dataset

The internal dataset contains customer-level information from a retail banking environment.

The source data include information related to:

- customer demographics;
- current-account relationships;
- term-deposit relationships;
- loan relationships;
- transaction activity;
- account balances;
- digital-banking usage;
- credit-card adoption events.

The final analytical dataset contains customer-level predictors constructed before the customer-specific temporal cutoff.

Examples of modeled feature groups include:

- age and sex;
- electronic-banking registration channel;
- SMS usage;
- verification method;
- login activity;
- current-account count and balance;
- term-deposit count and balance;
- loan count and loan amounts;
- transaction level and category;
- transaction amount statistics.

#### VIB target definition

A positive customer is a customer who opens at least one credit card during the study period.

A negative customer is a customer who does not open a credit card during the study period.

#### VIB temporal cutoff

For an adopter, the cutoff is the first credit-card opening date:

```text
cutoff = first credit-card opening date
```

For a non-adopter, the cutoff is the last observed transaction date in the study year:

```text
cutoff = last observed transaction date
```

All predictors are constructed using information available on or before the valid cutoff defined in the notebook.

### 6.2 Forward-Looking Deployment Cohort

The FLDC experiments use a fixed observation window after customer relationship start.

For a horizon of \(h\) days:

1. Features are constructed using only the first \(h\) days.
2. Customers adopting during the first \(h\) days are excluded.
3. The target is adoption strictly after day \(h\).
4. Positive and negative customers receive the same feature-window length.

The repository contains VIB FLDC experiments for:

- 30 days;
- 60 days;
- 90 days;
- 180 days.

The repository contains Santander FLDC experiments for:

- 60 days;
- 90 days.

### 6.3 Santander dataset

The external validation uses the Santander Product Recommendation dataset.

The dataset contains monthly customer records and banking-product ownership indicators. The notebooks identify the relevant credit-card adoption event and construct customer-level predictors using the temporal rules defined by ELAPS.

The public Santander data are not included in this repository. Users must download the dataset from its authorized source and update the input paths in the notebooks.

### 6.4 Data availability statement

The internal VIB dataset contains confidential banking and customer information and cannot be publicly released because of privacy, contractual, and institutional restrictions.

This repository therefore provides:

- the experimental notebooks;
- preprocessing logic;
- model-training procedures;
- leakage-audit procedures;
- FLDC cohort construction;
- evaluation code;
- fairness analysis;
- explainability analysis.

Reproduction of the internal VIB results requires authorized access to the original data or to an equivalent dataset with compatible fields.

No confidential customer-level data should be uploaded to the public repository.

---

## Citation

If this repository supports your research, please cite the associated paper:

```bibtex
@article{tran2026elaps,
  title   = {ELAPS: A Leakage-Audited Machine Learning Framework for Credit-Card Cross-Sell Propensity Prediction in Retail Banking},
  author  = {Tran, Thu Trang and co-authors},
  journal = {Submitted},
  year    = {2026}
}
```

The citation information should be updated after the paper is accepted or published.

---

## License

The source code and notebooks are provided for academic and non-commercial research purposes.

The banking data are not covered by this repository license and remain subject to the data owner's confidentiality and usage restrictions.

---

## Contact

**Tran Thu Trang**  
Faculty of Information Technology  
Dai Nam University  
Hanoi, Vietnam  

Email: `trangtt@dainam.edu.vn`
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
