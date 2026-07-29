# ELAPS: Event-Based Leakage-Aware Propensity Scoring

Official implementation of **ELAPS**, a leakage-aware machine learning framework for credit-card cross-sell propensity prediction in retail banking.

> **Paper:** *ELAPS: A Leakage-Audited Machine Learning Framework for Credit-Card Cross-Sell Propensity Prediction in Retail Banking*

---

## Overview

ELAPS is designed to develop and evaluate credit-card propensity models under **deployment-faithful conditions**. The framework minimizes temporal data leakage through event-based feature construction and systematically evaluates model robustness using leakage diagnostics, external validation, explainability, and fairness analysis.

### Key Features

- Event-based feature construction
- Leakage-aware preprocessing
- Feature ablation analysis
- Label-shuffling sanity checks
- Placebo-cutoff diagnostics
- Forward-Looking Deployment Cohort (FLDC)
- External validation (Santander dataset)
- SHAP explainability
- Fairness evaluation

---

## Workflow

```
Raw Data
    │
    ▼
Data Preprocessing
    │
    ▼
Event-Based Feature Construction
    │
    ▼
Leakage Audit
    │
    ▼
Model Training
    │
    ▼
FLDC Evaluation
    │
    ▼
External Validation
    │
    ▼
Explainability & Fairness
```

---

## Repository Structure

```text
ELAPS/
│
├── README.md
├── requirements.txt
├── notebooks/
│   ├── VIB_Dataprocessing.ipynb
│   ├── VIB_ELAPS.ipynb
│   ├── VIB_LIGHTGBM_ROS.ipynb
│   ├── VIB_PLACEBO.ipynb
│   ├── VIB30cutoffngay.ipynb
│   ├── VIB60cutoffngay.ipynb
│   ├── VIB_cutoff90ngay.ipynb
│   ├── VIB180cutoffngay.ipynb
│   ├── Santander_ELAPS.ipynb
│   ├── Santander_PLACEBO.ipynb
│   ├── Santander60cutoff.ipynb
│   └── Santander90cutoff.ipynb
└── data/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<username>/ELAPS.git
cd ELAPS
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## Quick Start

Run the notebooks in the following order:

### Internal VIB experiments

1. `VIB_Dataprocessing.ipynb`
2. `VIB_ELAPS.ipynb`
3. `VIB_PLACEBO.ipynb`
4. `VIB60cutoffngay.ipynb`
5. `VIB_LIGHTGBM_ROS.ipynb`

### External validation

1. `Santander_ELAPS.ipynb`
2. `Santander_PLACEBO.ipynb`
3. `Santander60cutoff.ipynb`
4. `Santander90cutoff.ipynb`

---

## Datasets

### VIB Dataset

The internal VIB dataset contains confidential retail banking customer information and **cannot be publicly released**.

### Santander Dataset

External validation is performed using the **Santander Product Recommendation** dataset. Users should download the dataset from its official source before running the notebooks.

---

## Reproducibility

To reproduce the reported results:

1. Install all required packages.
2. Prepare the input datasets.
3. Update local data paths.
4. Execute each notebook from top to bottom.
5. Use the same random seeds provided in the notebooks.

---

## Data Availability

The internal VIB dataset contains confidential customer information and cannot be shared publicly.

This repository provides:

- Source code
- Experimental notebooks
- Data preprocessing pipeline
- Leakage-audit procedures
- FLDC implementation
- Model evaluation scripts

A synthetic dataset will be released to facilitate reproducibility without exposing confidential customer information.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{tran2026elaps,
  title={ELAPS: A Leakage-Audited Machine Learning Framework for Credit-Card Cross-Sell Propensity Prediction in Retail Banking},
  author={Tran, Thu Trang and co-authors},
  journal={Submitted},
  year={2026}
}
```

---

## License

This repository is provided for academic and non-commercial research purposes only.

---

## Contact

**Tran Thu Trang**

Faculty of Information Technology

Dai Nam University, Hanoi, Vietnam

Email: trangtt@dainam.edu.vn
