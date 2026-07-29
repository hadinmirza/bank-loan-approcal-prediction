# 🏦 Bank Loan Approval Prediction using Artificial Neural Networks (ANN)

## 📌 Project Overview

This project is an end-to-end Deep Learning classification project that predicts whether a bank customer is likely to accept a personal loan offer based on their demographic information, financial status, credit details, and existing banking relationships.

The project follows a rigorous Machine Learning workflow, beginning with business understanding and exploratory data analysis, followed by data cleaning, proper dataset splitting, feature scaling, systematic hyperparameter tuning of an Artificial Neural Network (ANN), and final unbiased model evaluation. It was developed as part of a Deep Learning coursework assignment to demonstrate a scientific, evidence-based approach to model architecture selection and hyperparameter optimization.

---

# 🎯 Problem Statement

Personal loans represent a significant revenue opportunity for commercial banks, but direct marketing campaigns are expensive and have low conversion rates when untargeted.

The objective of this project is to analyze bank customer characteristics, identify key financial and demographic drivers of loan acceptance, and build a Deep Learning model to accurately predict which customers are most likely to accept a personal loan offer, optimizing marketing effectiveness.

---

# ❓ Business Question

**Can we accurately predict whether a bank customer will accept a personal loan offer using their age, experience, income, family size, spending behavior, education level, and existing bank account portfolios?**

---

# 📊 Problem Type

**Binary Classification**

The target variable (`Personal.Loan`) has two possible classes (**0: No** and **1: Yes**), making this a supervised binary classification problem with moderate class imbalance (~90% non-acceptors vs. ~10% acceptors).

---

# 📂 Dataset Information

* **Source:** Universal Bank Personal Loan Dataset
* **Repository Name:** `bank-loan-approval-prediction`

**Original Dataset**

* 5,000 rows
* 14 columns

**Final Dataset Used for Modeling**

* 5,000 rows
* 11 features (after dropping non-predictive columns `ID` and `ZIP.Code` and separating the target `Personal.Loan`)

The dataset includes customer demographics (Age, Experience, Family), financial metrics (Income, CCAvg, Mortgage), education level, and current banking service participation (Securities Account, CD Account, Online Banking, Credit Card).

---

# 🎯 Target Variable

`Personal.Loan`

* `0`: Customer did not accept the personal loan offer.
* `1`: Customer accepted the personal loan offer.

---

# 🛠 Project Workflow

### 1. Business Understanding

* Defined the banking business problem and targeting strategy.
* Identified the prediction objective (Personal Loan conversion).
* Established `Personal.Loan` as the target variable.
* Defined the problem as a supervised Binary Classification task.

---

### 2. Data Understanding

Performed detailed initial data exploration examining:

* Dataset shape and dimensions ($5,000 \times 14$).
* Column names and data types.
* Summary statistics across numerical features.
* Target variable distribution (~90.4% class 0, ~9.6% class 1).

---

### 3. Data Quality Assessment

Inspected data quality and identified anomalies:

* No missing values (`NaN`/nulls) were found.
* No duplicate records were detected.
* 52 invalid negative values were identified in the `Experience` column (e.g., $-1$, $-2$, $-3$).

---

### 4. Data Cleaning

Applied necessary data cleaning steps:

* Converted negative values in `Experience` to positive values using `.abs()`, treating them as data entry recording errors.
* Dropped unnecessary columns:
  * `ID`: Unique record index with zero predictive power.
  * `ZIP.Code`: Location identifier not required for credit modeling in this scope.
* Saved the cleaned dataset to `cleaned_dataset.csv` ($5,000 \text{ rows} \times 12 \text{ columns}$).

---

### 5. Exploratory Data Analysis (EDA)

Generated visual distributions and correlation heatmaps to understand customer behavior:

* **Target Imbalance:** Visualized the ~90/10 target imbalance.
* **Correlation Analysis:** Identified strong positive correlation between `Age` and `Experience` ($r \approx 0.99$) as well as strong correlation between `Income` and `CCAvg` (Credit Card spending).
* **Feature Distributions:** Analyzed multi-modal and skewed distributions in `Income`, `CCAvg`, and `Mortgage`.
* **Outlier Visualizations:** Boxplots highlighted significant positive skewness/outliers in `Mortgage` and `Income`.

---

### 6. Feature Selection & Preparation

Separated input features and target variable:

* **Features ($X$):** 11 variables (`Age`, `Experience`, `Income`, `Family`, `CCAvg`, `Education`, `Mortgage`, `Securities.Account`, `CD.Account`, `Online`, `CreditCard`)
* **Target ($y$):** `Personal.Loan`

---

### 7. Train-Validation-Test Data Split (Leakage Prevention)

To prevent data leakage and evaluate generalization rigorously, a 3-way stratified split was applied:

* **Training Set (70%):** 3,500 records (used exclusively to learn model weights and biases)
* **Validation Set (15%):** 750 records (used exclusively for hyperparameter tuning)
* **Testing Set (15%):** 750 records (kept completely unseen until final model evaluation)

*Stratified sampling (`stratify=y`) was enforced across all splits to preserve the target distribution (~90.4% / 9.6%).*

---

### 8. Feature Scaling

Standardized features using `StandardScaler`:

* Scaler was fitted **ONLY on `X_train`** to prevent data leakage.
* Transformed `X_train`, `X_val`, and `X_test` using the fitted training scaler parameters.

---

### 9. Hyperparameter Tuning & Neural Network Experiments

Tuned the Artificial Neural Network systematically by changing **one architectural variable at a time** across 5 experiments:

| Exp # | Hidden Layers | Neurons / Architecture | Learning Rate ($\eta$) | Batch Size | Epochs | Train Acc | Val Acc | Train Loss | Val Loss | Key Observation & Takeaway | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **1** | 1 | 16 | 0.10 | 32 | 50 | 97.77% | 98.00% | 0.0599 | 0.0765 | Baseline model with stable convergence. | Baseline |
| **2** | **1** | **16** | **0.01** | **32** | **50** | **98.57%** | **98.20%** | **0.0399** | **0.0550** | **Optimal LR improved convergence, reduced loss, and peaked Val Acc.** | **🏆 Winner** |
| **3** | 1 | 32 | 0.01 | 32 | 50 | 99.37% | 98.10% | 0.0231 | 0.0444 | Higher capacity fitted training data better, but showed slight overfitting signs. | Over-capacity |
| **4** | 2 | 64, 32 | 0.01 | 32 | 50 | 96.14% | 97.60% | 0.0993 | 0.0667 | Extra depth increased parameter complexity without improving generalization. | Sub-optimal |
| **5** | 2 | 64, 32 + Dropout(0.3) | 0.01 | 32 | 50 | 94.09% | 93.90% | 0.1279 | 0.1291 | Dropout regularized too aggressively, leading to severe underfitting. | Underfitting |

---

### 10. Final Model Architecture

The optimal model selected from hyperparameter tuning (**Experiment 2**) consists of:

* **Input Layer:** 11 features
* **Hidden Layer 1:** 16 Neurons (ReLU activation)
* **Output Layer:** 1 Neuron (Sigmoid activation)
* **Optimizer:** Adam ($\text{learning rate} = 0.01$)
* **Loss Function:** Binary Cross-Entropy
* **Batch Size:** 32 | **Epochs:** 50

---

### 11. Model Evaluation on Unseen Test Set

After final model selection, the model was evaluated **once** on the untouched 15% testing set.

#### Evaluation Metrics (Test Set):

| Metric | Value | Technical / Business Context |
| :--- | :---: | :--- |
| **Test Accuracy** | **97.20%** | Correctly classifies 97.2% of unseen loan candidates. |
| **Precision (Class 1)** | **86.96%** | When predicting loan acceptance, the model is right 87% of the time. |
| **Recall (Class 1)** | **83.33%** | Captures 83.3% of all potential loan acceptors in the test set. |
| **F1-Score (Class 1)** | **85.11%** | Harmonic mean reflecting strong performance despite target imbalance. |
| **ROC-AUC** | **0.9877** | Exceptional discrimination capability across classification thresholds. |

#### Confusion Matrix (Test Set)

```text
[[446   6]
 [  8  40]]