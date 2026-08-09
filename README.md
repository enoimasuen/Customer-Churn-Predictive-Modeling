# Predicting Customer Churn at ABC Multinational Bank

### A Comparative Study of Logistic Regression, Boosted C5.0, and Random Forest

## 📊 [View the Full Interactive HTML Report](https://enoimasuen.github.io/Customer-Churn-Predictive-Modeling/Customer-Churn-ML.html)

![R](https://img.shields.io/badge/R-%E2%89%A54.2-276DC3?logo=r&logoColor=white)
![R Markdown](https://img.shields.io/badge/R%20Markdown-knit%20to%20PDF%20%2F%20HTML-1F65B7?logo=rstudio&logoColor=white)
![Models](https://img.shields.io/badge/models-Logistic%20%7C%20C5.0%20%7C%20Random%20Forest-success)
![Champion AUC](https://img.shields.io/badge/champion%20AUC-0.863-brightgreen)
![Course](https://img.shields.io/badge/Harvard%20Extension-CSCI%20E--106-A51C30)
![Status](https://img.shields.io/badge/status-complete-blue)

> Identifying the customers most likely to leave ABC Multinational Bank — and the factors that drive them out — so retention efforts can be aimed where they matter most.

**Team:** *Harry Plotter and the Chamber of Refits* · Harvard Extension School · CSCI E-106: Statistical Data Modeling

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [The Dataset](#the-dataset)
- [Methodology](#methodology)
- [Model Comparison](#model-comparison)
- [Key Findings from Exploratory Analysis](#key-findings-from-exploratory-analysis)
- [Limitations & Assumptions](#limitations--assumptions)
- [Ongoing Model Monitoring](#ongoing-model-monitoring)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Report Section Map](#report-section-map)
- [Authors & Contributions](#authors--contributions)
- [References](#references)
- [License](#license)

---

## Overview

Customer churn — a client closing their accounts and moving to a competitor — is one of the most expensive problems a retail bank faces, since acquiring a new customer costs far more than retaining an existing one. This project builds and validates a predictive model that flags customers at high risk of churning so the bank can intervene early with targeted retention strategies (personalized offers, improved engagement, proactive outreach).

Working from a dataset of **10,000 customers**, we trained and compared three classification models:

| Model | Role | Why |
| :--- | :--- | :--- |
| **Boosted C5.0** | 🏆 **Champion** | Best balance of discrimination, calibration, and generalization |
| **Random Forest** | Challenger | Strong performance but unstable on unseen data (overfit) |
| **Logistic Regression** | Benchmark | Retained for interpretability — odds ratios give regulatory transparency |

The headline question driving the work: **why do some clients leave ABC Multinational Bank while others stay, and can we predict who is at risk before they go?**

---

## Key Results

All metrics below are **out-of-sample** (30% held-out test set, 2,999 customers). The champion model is highlighted.

| Model | Accuracy | Sensitivity | Specificity | Precision | Recall | F1 | AUC | MSE | RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.7953 | 0.4354 | 0.8874 | 0.4972 | 0.4354 | 0.4642 | 0.7624 | 0.1380 | 0.3715 |
| **Boosted C5.0** 🏆 | **0.8469** | 0.6268 | **0.9033** | **0.6238** | 0.6268 | **0.6253** | **0.8629** | **0.1033** | **0.3215** |
| Random Forest | 0.8459 | **0.6432** | 0.8978 | 0.6170 | **0.6432** | 0.6298 | 0.8593 | 0.1161 | 0.3407 |

**Champion model (Boosted C5.0):** AUC **0.863**, accuracy **84.7%**, F1 **0.689** (train) / **0.625** (test).

The Boosted C5.0 model was selected because it achieved the highest AUC, the lowest probability error (Brier/RMSE — i.e. the best-calibrated probabilities), and a strong F1 score, all while generalizing far more reliably than Random Forest. Random Forest matched it on raw test accuracy and edged it on recall, but its train accuracy of 98.1% collapsing to 84.5% on test signaled meaningful overfitting and weaker probability calibration. Logistic Regression was kept as the interpretable benchmark.

---

## Repository 

```
.
├── README.md                            
├── Customer-Churn-ML.Rmd              # Full analysis source (R Markdown)
├── Customer-Churn-ML.pdf              # Rendered report (PDF)
├── Customer-Churn-ML.html             # Rendered report (HTML)
├── Bank Customer Churn Prediction.csv   # Raw dataset (10,000 customers)
└── refs.bib                             # BibTeX bibliography
```

> The `.Rmd` reads `Bank Customer Churn Prediction.csv` and `refs.bib` from the working directory. Keep all four files in the same folder before knitting.

---

## The Dataset

`Bank Customer Churn Prediction.csv` contains **10,000 rows (one per customer)** and is high quality: **no missing values and no duplicate rows**. The target variable is `churn`.

### Data Dictionary

| Variable | Type | Description |
| :--- | :--- | :--- |
| `customer_id` | ID | Account number (dropped before modeling) |
| `credit_score` | Continuous | Customer credit score |
| `country` | Categorical | France, Spain, or Germany |
| `gender` | Categorical | Female or Male |
| `age` | Continuous | Customer age |
| `tenure` | Continuous | Years as a client |
| `balance` | Continuous | Account balance |
| `products_number` | Continuous | Number of bank products held (1–4) |
| `credit_card` | Categorical | Holds a credit card (0 = no, 1 = yes) |
| `active_member` | Categorical | Active member status (0 = no, 1 = yes) |
| `estimated_salary` | Continuous | Estimated salary |
| **`churn`** | **Target** | **1 = left the bank, 0 = retained** |

### Class Imbalance

The target is imbalanced: **20.4% churned (2,037)** vs **79.6% retained (7,963)**. Because of this, **accuracy alone is not a sufficient metric** — a model that predicts "no churn" for everyone would still score ~80%. Evaluation therefore emphasizes sensitivity, specificity, precision, recall, F1, and AUC, and each model uses an imbalance-handling strategy (see below).

---

## Methodology

```
Raw CSV (10,000 customers)
        │
        ▼
Data quality checks ──► no missing values, no duplicates, drop customer_id
        │
        ▼
Exploratory data analysis ──► distributions + churn rates by predictor (Figs 1–9)
        │
        ▼
Feature prep ──► dummy variables (numeric pipeline) + factors (tree pipeline)
        │
        ▼
70 / 30 stratified train–test split (churn rate ≈ 20.4% in both)
        │
        ├──► Logistic Regression  (stepwise AIC + threshold tuning)
        ├──► Boosted C5.0          (25 boosting trials + threshold tuning)   ◄── Champion
        └──► Random Forest         (500 trees, class weighting)
        │
        ▼
Evaluation ──► AUC · accuracy · precision/recall · F1 · Brier/RMSE · train-vs-test
        │
        ▼
Champion + Benchmark selection ──► limitations, assumptions, monitoring plan
```

### Preprocessing

`customer_id` is dropped. Two parallel feature representations are built: a **dummy-encoded** dataset for logistic regression, and a **factor-encoded** dataset (`bank_factor`) for the tree-based models. The data is split **70% train / 30% test**, preserving the ~20.4% churn rate in both partitions.

### Models

- **Logistic Regression (benchmark).** Variables selected via **stepwise AIC** on the full model. The final model retained `credit_score`, `age`, `balance`, `active_member`, `country = Germany`, and `gender`. Multicollinearity was ruled out (all VIF < 2), and goodness-of-fit was assessed with the Hosmer–Lemeshow test (p = 0.27, fails to reject — adequate fit). An operating threshold of **0.35** was chosen over the default 0.50 to improve recall on churners.

- **Boosted C5.0 (champion).** A boosted decision-tree ensemble with **25 boosting iterations**. Probability thresholds from 0.30–0.50 were tested; **0.35** was selected for the best recall/precision trade-off. Inherently more robust to class imbalance than logistic regression, and threshold tuning sharpened minority-class detection further.

- **Random Forest (challenger).** **500 trees**, `mtry = 3`, with **class weights `No = 1, Yes = 4`** to penalize misclassifying churners under the 80/20 imbalance (`set.seed(1023)` for reproducibility). Out-of-bag error was **16.1%**. Variable importance ranked `age` and `products_number` highest by Mean Decrease Accuracy, and `age`, `balance`, `credit_score` highest by Mean Decrease Gini.

### Evaluation

Every model was scored on both train and test sets to expose overfitting. Threshold-independent probability metrics (**Brier score / RMSE / SSE**) measure calibration; **ROC curves and AUC** measure ranking ability; and the confusion-matrix family (**accuracy, sensitivity, specificity, precision, recall, F1, balanced accuracy**) measures classification quality at the chosen cutoff.

---

## Model Comparison

On the held-out test set, **Boosted C5.0 and Random Forest clearly outperform Logistic Regression**, with near-identical ROC curves (AUC 0.863 vs 0.859) sitting well above logistic regression's 0.762.

The deciding factor was **stability and calibration**, not raw test accuracy:

- **Random Forest** posted 98.1% train accuracy but only 84.5% on test — a large gap indicating overfitting — and its Brier score rose sharply out-of-sample.
- **Boosted C5.0** moved only from 87.8% (train) to 84.7% (test) and kept the lowest Brier/RMSE of the three, meaning its predicted probabilities are the most trustworthy.

Given the business objective — reliably ranking who is most at risk so retention spend is well-targeted — **Boosted C5.0 is the champion** and **Logistic Regression is retained as the interpretable benchmark** for regulatory contexts where odds ratios must be explained.

---

## Key Findings from Exploratory Analysis

The EDA (Figures 1–9 in the report) surfaced several actionable patterns in who churns:

- **Age** — Older customers churn more; the age distribution is right-skewed, with most clients aged 25–40.
- **Balance** — Higher account balances are associated with *higher* churn risk.
- **Geography** — Germany churns at **32%**, roughly double France (16%) and Spain (17%).
- **Activity** — Active members churn at **14%** vs **27%** for inactive members — nearly half the rate.
- **Gender** — Female customers churn at a higher rate than male customers.
- **Number of products** — A striking signal: **100% of 4-product customers and 83% of 3-product customers churned**, while 2-product customers were the *most* loyal (7.6%). (Caveat: only a small share of customers hold 3–4 products.)
- **Little to no signal** — `estimated_salary` and `credit_card` ownership showed minimal relationship with churn. Longer-tenured customers were *not* more loyal (churners had slightly higher, more variable tenure).

---

## Limitations & Assumptions

**Logistic Regression**
- The **linear log-odds assumption is violated** — a Box-Tidwell test found significant non-linear terms for `age`, `balance`, and `credit_score`. Polynomial or spline transformations are worth exploring in future work.
- Sensitive to influential observations; prone to overfitting in higher-dimensional, low-sample settings.

**Random Forest**
- **Black box** — individual predictions can't be cleanly explained to customers or regulators.
- **Required class-weight correction** (`No = 1, Yes = 4`); without it the model predicted almost entirely "No."
- **Cannot extrapolate** beyond the range of the training data, and showed the most overfitting (98.1% train → 84.5% test).

**Boosted C5.0**
- Boosting can overfit if unchecked.
- Lacks the simple, interpretable relationships (e.g. odds ratios) that make logistic regression easy to explain in a business/regulatory setting.
- Still carries some residual sensitivity to class imbalance.

**Dataset**
- A **static cross-sectional snapshot** — no time-series features such as balance trend or transaction frequency.
- **No competitor or macroeconomic data**, which may independently drive churn.

Note on assumptions: tree-based ensembles make **no linearity or distributional assumptions** and don't require normally distributed residuals (that's an OLS assumption that doesn't apply here). For logistic regression, 4 of 6 standard assumptions were fully met, with log-odds linearity the main concern.

---

## Ongoing Model Monitoring

A deployed churn model degrades silently as customer behavior, the economy, and product offerings shift. The plan below keeps the champion model honest in production.

### Monitoring Thresholds & Triggers

| Metric | Frequency | Alert Threshold | Action |
| :--- | :--- | :--- | :--- |
| AUC (ROC) | Monthly | Drop > 0.03 from baseline (below 0.81) | Re-validate, consider retraining |
| Population Stability Index (PSI) | Monthly | PSI > 0.20 | Investigate data drift, retrain |
| Accuracy | Monthly | Drop > 3% from baseline (below 0.80) | Flag for review |
| Actual vs Expected Churn Rate | Monthly | Divergence > 5 percentage points | Recalibrate probability outputs |
| Hosmer–Lemeshow p-value | Quarterly | p < 0.05 | Recalibrate model |

Baselines are set from actual test results: **AUC 0.863, accuracy 0.847**.

### Monitoring Assumptions

Deployment assumes **stable predictor distributions** (tracked via PSI), a **consistent data pipeline** (schema changes trigger review), a **stable business definition of churn**, and **no structural breaks** (major economic shocks, regulatory change, or mergers can invalidate the model).

### Replacement Triggers

The model should be retired and rebuilt if **AUC falls below 0.75 for two consecutive months**, **PSI exceeds 0.25 for two periods**, a **regulatory requirement** mandates a different approach, or a **challenger model proves materially superior** in the scheduled annual evaluation.

---

## Reproducing the Analysis

### Prerequisites

- **R ≥ 4.2** (and RStudio recommended)
- A **LaTeX distribution** for PDF output — `tinytex::install_tinytex()` is the easiest route
- Pandoc (bundled with RStudio)

### 1. Install dependencies

```r
install.packages(c(
  "readr", "knitr", "pander", "rmarkdown", "markdown",
  "MASS", "faraway", "olsrr", "caret", "fastDummies", "ISLR2",
  "dplyr", "car", "lmtest", "tree", "leaps", "ResourceSelection",
  "C50", "randomForest", "neuralnet", "tidyverse", "janitor",
  "skimr", "GGally", "pROC", "rpart", "rpart.plot", "pscl", "patchwork"
))
```

### 2. Place the data

Ensure `Bank Customer Churn Prediction.csv` and `refs.bib` are in the same directory as the `.Rmd`.

### 3. Knit the report

From RStudio, open the `.Rmd` and click **Knit** (PDF or HTML), or run from the console:

```r
rmarkdown::render("Customer-Churn-ML.Rmd")
```

The Random Forest uses `set.seed(1023)`, so results are fully reproducible.

---

## Report Section Map

The written report (PDF/HTML) follows the course template. Quick index:

| Section | Contents |
| :--- | :--- |
| Executive Summary | One-page summary of approach, champion model, and recommendation |
| I. Introduction | Problem framing and objective |
| II. Data Description & Quality | Data dictionary, quality checks, EDA (Figures 1–9) |
| III. Model Development | Logistic regression: stepwise AIC, odds ratios, VIF, Hosmer–Lemeshow, threshold tuning |
| IV. Model Performance Testing | Logistic regression train vs test generalization |
| V. Challenger Models | Boosted C5.0 and Random Forest, ROC comparison, performance table |
| VI. Limitations & Assumptions | Champion/benchmark selection, Box-Tidwell, model assumptions |
| VII. Ongoing Model Monitoring | Thresholds, monitoring assumptions, replacement triggers |
| VIII. Conclusion | Final model rationale |
| Bibliography & Appendix | Confusion matrices, ROC grids, additional plots, aggregated metrics |

---

## Authors & Contributions

**Team — *Harry Plotter and the Chamber of Refits***

| Author | Harvard ID |
| :--- | :--- |
| Andrew Smock | 41802707 |
| Enoghayin Imasuen | 31679354 |
| Lily Nguyen | 81814465 |
| Tim Bohlemann | 51818342 |

This was a collaborative effort across data preparation, modeling, evaluation, and writing. Section-level ownership:

| Section / Component | Contributor |
| :--- | :--- |
| Abstract | Andrew Smock |
| Executive Summary | Enoghayin Imasuen |
| Introduction | Andrew Smock |
| Data description, quality & EDA | Andrew Smock |
| Logistic Regression development | Tim Bohlemann |
| Logistic Regression testing | Tim Bohlemann |
| Boosted C5.0 challenger model | Lily Ngyuen |
| Random Forest challenger model | Enoghayin Imasuen |
| Model Limitations & Assumptions | Enoghayin Imasuen |
| Ongoing Model Monitoring Plan (VII) | Enoghayin Imasuen |
| Conclusion (contributing) | Enoghayin Imasuen |
| Conclusion (contributing) | Lily Ngyuen |
| Bibliography | Lily Nguyen |
| Additional analysis graphs (tenure, credit card) | Enoghayin Imasuen |

---

## References

Key sources cited in the report (full citations in `refs.bib`):

- Fawcett, T. (2006). *An Introduction to ROC Analysis.* Pattern Recognition Letters 27(8): 861–874.
- Hosmer, D. W., & Lemeshow, S. (2000). *Applied Logistic Regression* (2nd ed.). Wiley.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2014). *An Introduction to Statistical Learning with Applications in R.* Springer.
- Long, J. S. (1997). *Regression Models for Categorical and Limited Dependent Variables.* Sage.
- Perner, P. (ed.) (2018). *Machine Learning and Data Mining in Pattern Recognition.* Springer.
- Leung, K. (2021). *Assumptions of Logistic Regression, Clearly Explained.* Towards Data Science.
- Kennedy, W. B. (2024). *Achieve Better Classification Results with ClassificationThresholdTuner.* Towards Data Science.
- Sobolik, T., & Boudard, L. (2024). *Machine Learning Model Monitoring: Best Practices.* Datadog.

---

## License

Academic coursework for Harvard Extension School CSCI E-106. Shared for educational and portfolio purposes. The dataset is used under the terms of its original source.
