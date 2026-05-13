# 🩺 Diabetes Classification — Full Machine Learning Pipeline

> **Course Lab Project** | Binary Classification | UCI Pima Indians Diabetes Dataset

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Google Colab](https://img.shields.io/badge/Run%20in-Google%20Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/drive/1nFKWiGdlD6WjP1BUCPcuoS4oGWhZEbpA?usp=sharing)

---

## 📌 Project Overview

This project builds a complete, end-to-end supervised machine learning pipeline to **predict whether a patient has diabetes** based on clinical diagnostic measurements. It covers all standard ML workflow stages — from raw data collection to business recommendations — and serves as a structured lab submission demonstrating real-world data science practice.

| Item | Detail |
|---|---|
| **Problem Type** | Binary Classification |
| **Target Variable** | `Outcome` — `1` = Diabetes, `0` = No Diabetes |
| **Dataset** | Pima Indians Diabetes Database (UCI ML Repository) |
| **Original Source** | [archive.ics.uci.edu/dataset/34/diabetes](https://archive.ics.uci.edu/dataset/34/diabetes) |
| **Download URL** | [raw.githubusercontent.com/jbrownlee/Datasets/.../pima-indians-diabetes.data.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) |
| **Rows / Features** | 768 records × 8 clinical features |
| **Best Model** | Tuned Gradient Boosting |

---

## 📁 Repository Structure

```
├── Diabetes_Classification_Lab.ipynb   # Main Colab notebook (all 8 steps)
└── README.md                           # This file
```

---

## 🔗 Quick Links

- 📓 **Colab Notebook (Live):** [Open in Google Colab](https://colab.research.google.com/drive/1nFKWiGdlD6WjP1BUCPcuoS4oGWhZEbpA?usp=sharing)
- 📦 **Dataset (Original):** [UCI ML Repository — Pima Indians Diabetes](https://archive.ics.uci.edu/dataset/34/diabetes)
- ⬇️ **Dataset (Download URL):** [raw.githubusercontent.com/jbrownlee/Datasets](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)

---

## 🧱 Step-by-Step Breakdown

---

### Step 0 — Library Setup & Imports

**What it does:**
Loads all required Python libraries — `pandas`, `numpy`, `matplotlib`, `seaborn` for data handling and visualization; `scikit-learn` for preprocessing, modelling, feature selection, and evaluation. A consistent Seaborn theme is applied and warnings are suppressed.

**Output:**
```
✅ Libraries loaded successfully
```

**What this tells us:**
The environment is correctly configured. The notebook is fully reproducible in any standard Colab session with no additional installation required.

---

### Step 1 — Data Collection

**What it does:**
Downloads the **Pima Indians Diabetes dataset** at runtime from a publicly hosted raw CSV mirror of the original UCI dataset. Column names are manually assigned since the raw file has no header row. The data is loaded into a Pandas DataFrame.

> 💡 **AI-Assisted Knowledge** *(~5% marker — Step 1)*: The dataset selection strategy, understanding of the Pima Indians Diabetes database's origin (National Institute of Diabetes and Digestive and Kidney Diseases), and the feature definitions (e.g., the meaning of `DiabetesPedigreeFunction` as a hereditary risk score) were informed with AI assistance to ensure correct domain interpretation before analysis.

**Output:**
```
Dataset shape: (768, 9)
```

**What this tells us:**
The dataset has 768 patient records and 9 columns (8 clinical features + 1 binary target). Features include plasma glucose concentration, blood pressure, BMI, age, and insulin levels. While well-known as a benchmark dataset, it contains real-world data quality challenges such as zero-coded missing values that must be handled before modelling.

---

### Step 2 — Data Preprocessing

#### 2a. Missing Value Handling

**What it does:**
Five columns — `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` — use `0` as a placeholder for missing data (physiologically impossible values). These are replaced with `NaN` and imputed using the **median grouped by outcome class**, which is more accurate than a single global median since diabetic and non-diabetic patients have different distributions.

**Output:**
```
Zero counts (= missing) before imputation:
Glucose           5
BloodPressure    35
SkinThickness   227
Insulin         374
BMI              11

Missing values after imputation: 0
```

**What this tells us:**
`Insulin` is missing in nearly half the dataset (374/768 = 49%) and `SkinThickness` in ~30%. These are substantial gaps that could bias a model if not handled carefully. Group-wise median imputation preserves the statistical difference between the two classes, producing more realistic fill values than a naive global approach.

---

#### 2b. Outlier Detection & Handling (IQR Winsorization)

**What it does:**
Generates a 2×4 grid of box plots for all 8 features to visually identify outliers. Then applies **Winsorization** — capping values at 1.5×IQR boundaries — to neutralise extreme values without discarding any rows.

**Output:**
```
✅ Outliers capped via Winsorization
```

**What this tells us:**
`Insulin` and `SkinThickness` show the most extreme spreads, consistent with their high missingness rates. `Pregnancies` is right-skewed (most patients have 0–4, but some have up to 17). Winsorization retains all 768 records while reducing the distorting influence of extremes on model training.

---

#### 2c. Duplicate Check

**What it does:**
Scans for fully duplicate rows and removes them if found.

**Output:**
```
Duplicate rows: 0
Shape after dedup: (768, 9)
```

**What this tells us:**
No duplicate records exist. All 768 entries are unique patient observations — no data is lost in this step.

---

#### 2d. Exploratory Data Analysis (EDA)

**What it does:**
Produces four visualizations: a class distribution bar chart and pie chart, overlapping density histograms for each feature split by outcome, a Pearson correlation heatmap, and a pairplot of key features coloured by class.

**Key Outputs:**

*Class distribution:*
```
Class ratio — 0: 500 (65.1%),  1: 268 (34.9%)
```

*Strongest correlations with Outcome (from heatmap):*
```
Glucose                  ≈ 0.47
BMI                      ≈ 0.29
Age                      ≈ 0.24
Pregnancies              ≈ 0.22
DiabetesPedigreeFunction ≈ 0.17
```

**What this tells us:**
The dataset is **moderately imbalanced** at a 65/35 split — accuracy alone is unreliable; F1 and ROC-AUC are essential metrics. `Glucose` is the strongest predictor by a wide margin. The pairplot shows clear class separation along Glucose and BMI axes, confirming these features carry the most discriminative signal.

---

### Step 3 — Feature Selection

**What it does:**
Applies three complementary feature-ranking techniques:
1. **SelectKBest (ANOVA F-score)** — univariate statistical test
2. **Random Forest Feature Importances** — tree-based non-linear ranking
3. **Recursive Feature Elimination (RFE)** — wrapper method using Logistic Regression

The top 6 features by Random Forest importance are selected for modelling.

**Output:**
```
✅ Final selected features (6): ['Insulin', 'Glucose', 'SkinThickness', 'BMI', 'Age', 'DiabetesPedigreeFunction']
```

**What this tells us:**
`Insulin` and `Glucose` rank highest — both are direct biochemical markers of diabetes. `SkinThickness` was retained despite high missingness, suggesting it carries non-linear signal after imputation. `Pregnancies` and `BloodPressure` were dropped, reducing complexity without meaningful information loss. Agreement across all three methods gives high confidence in the chosen feature set.

---

### Step 4 — Model Building

**What it does:**
Splits data 80/20 with stratification to preserve class ratios, then trains **5 algorithms** evaluated with **5-fold stratified cross-validation**.

**Train/Test Split:**
```
Train size: 614  |  Test size: 154
Train class balance: {0: 400, 1: 214}
Test  class balance: {0: 100, 1: 54}
```

**Baseline Results:**
```
Logistic Regression     | CV AUC: 0.8837 ± 0.0135 | Test Acc: 0.7468
Decision Tree           | CV AUC: 0.8260 ± 0.0439 | Test Acc: 0.8506
Random Forest           | CV AUC: 0.9428 ± 0.0083 | Test Acc: 0.8636
Gradient Boosting       | CV AUC: 0.9451 ± 0.0174 | Test Acc: 0.8766
K-Nearest Neighbors     | CV AUC: 0.8993 ± 0.0171 | Test Acc: 0.8766
```

**What this tells us:**
**Gradient Boosting** and **Random Forest** are the clear top performers with CV AUC above 0.94. **Logistic Regression** underperforms on test accuracy (74.68%) despite a reasonable CV AUC, suggesting the decision boundary is non-linear. **Decision Tree** shows the widest CV variance (±0.0439), indicating overfitting sensitivity. Random Forest's low standard deviation (±0.0083) confirms it produces the most stable predictions of all five models.

---

### Step 5 — Model Optimization

**What it does:**
Applies **GridSearchCV** (5-fold CV, scoring on ROC-AUC) on the top two models.

**Output:**
```
Best RF params:  {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
Best RF CV AUC:  0.9457

Best GB params:  {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}
Best GB CV AUC:  0.9461

Tuned Random Forest:     Acc=0.8701 | F1=0.8148 | AUC=0.9498
Tuned Gradient Boosting: Acc=0.8831 | F1=0.8302 | AUC=0.9550
```

**What this tells us:**
Tuned Gradient Boosting uses a **slow learning rate (0.05)** with **shallow trees (depth=3)** and **subsampling (0.8)** — a conservative configuration that prevents overfitting by learning cautiously at each boosting step. Tuned Random Forest benefits from more trees (200) and a minimum leaf size of 2, slightly smoothing its decision boundary. Both tuned models improve over their baselines. Gradient Boosting edges ahead on all three metrics.

---

### Step 6 — Model Evaluation

**What it does:**
Comprehensively evaluates the best tuned model with all classification metrics, confusion matrices for both tuned models, ROC curves for all 7 models, and probabilistic regression-style diagnostics.

**🏆 Best Model: Tuned Gradient Boosting**

**Classification Metrics:**
```
Accuracy:   0.8831
Precision:  0.8462
Recall:     0.8148
F1-Score:   0.8302
ROC-AUC:    0.9550
```

**Detailed Classification Report:**
```
              precision    recall  f1-score   support

 No Diabetes       0.90      0.92      0.91       100
    Diabetes       0.85      0.81      0.83        54

    accuracy                           0.88       154
   macro avg       0.87      0.87      0.87       154
weighted avg       0.88      0.88      0.88       154
```

**Probabilistic Regression-style Diagnostics:**
```
MAE:   0.1592
MAPE:  7,974,238,123.90%   ⚠️ (unreliable — MAPE breaks when true labels are 0)
MSE:   0.0825
RMSE:  0.2873
R²:    0.6376
```

**What this tells us:**
- A **ROC-AUC of 0.9550** is excellent — the model reliably ranks diabetic patients above non-diabetic ones in 95.5% of cases.
- **Recall of 81.48%** on the Diabetes class means roughly 1 in 5 actual diabetic patients is missed. In a clinical setting, lowering the decision threshold from 0.5 → 0.4 would improve Recall at a minor Precision cost — the right tradeoff when missing a true positive is more dangerous than a false alarm.
- The model performs better on the majority class (No Diabetes, F1=0.91) than the minority class (Diabetes, F1=0.83), expected given the 65/35 imbalance.
- **MAPE is meaningless here** — the astronomically large value (7.97 billion %) results from division by near-zero when true labels are 0. MAE (0.1592) and RMSE (0.2873) are the valid probabilistic diagnostics.
- **R² of 0.6376** means the model's predicted probabilities explain ~64% of the variance in true outcomes — a strong result for a binary classification problem.

> 💡 **AI-Assisted Knowledge** *(~5% marker — Step 6)*: The explanation of why MAPE produces an astronomically large value when true labels contain zeros (division by near-zero), and when Brier Score or Log Loss is preferable as a probabilistic calibration metric, was clarified with AI assistance.

---

### Step 7 — Business Insights & Recommendations

**What it does:**
Interprets the model's outputs through a clinical and business lens, identifying actionable predictors and proposing deployment strategies.

#### 🔑 Key Predictors and Their Impact

| Feature | Relative Importance | Clinical Meaning |
|---|---|---|
| **Glucose** | ⭐⭐⭐⭐⭐ Highest | Fasting blood glucose is the primary diagnostic marker; elevated levels directly indicate insulin resistance |
| **BMI** | ⭐⭐⭐⭐ High | Obesity is the leading modifiable risk factor for Type 2 diabetes; BMI > 30 significantly raises risk |
| **Age** | ⭐⭐⭐ Moderate | Risk accumulates with age; patients over 45 need closer monitoring |
| **Insulin** | ⭐⭐⭐ Moderate | Abnormal insulin response patterns are a hallmark of pre-diabetic states |
| **DiabetesPedigreeFunction** | ⭐⭐ Moderate | Captures genetic predisposition — non-modifiable but critical for risk stratification |
| **SkinThickness** | ⭐ Lower | Proxy for subcutaneous fat; contributes modest non-linear signal |

#### 💼 Business / Clinical Recommendations

1. **Early Screening Programme** — Deploy the model at primary care touchpoints to flag high-risk patients (Glucose > 140, BMI > 30, Age > 45) for priority HbA1c confirmatory testing, reducing late-stage diagnosis costs.

2. **Resource Pre-Allocation** — Use the predicted probability score to pre-allocate endocrinology appointments and glucometers for patients scored above 0.6, reducing bottlenecks and wait times.

3. **Lifestyle Intervention Targeting** — Patients in the 0.4–0.6 probability "grey zone" are optimal candidates for preventive lifestyle programmes (diet counselling, exercise), where early action yields the highest ROI.

4. **Threshold Calibration** — Lower the decision threshold from 0.5 → 0.4 to prioritise Recall over Precision — an acceptable clinical tradeoff where missing a true diabetic is more costly than a false alarm.

#### ⚠️ Limitations & Future Improvements

| Limitation | Suggested Improvement |
|---|---|
| Small dataset (768 rows) | Collect more samples; apply SMOTE oversampling for class imbalance |
| Single ethnic group (Pima community) | Validate across diverse populations before clinical deployment |
| High missingness in Insulin (49%) | Collect insulin readings routinely at point of care |
| Static, batch model | Implement quarterly retraining as new patient data arrives |
| No temporal/longitudinal data | Add HbA1c trend over time as a feature |
| No causal inference | Pair with domain expert review; correlation ≠ causation |

---



## 📊 Model Performance Summary

| Model | CV ROC-AUC | Test Accuracy | Test F1 | Test ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.8837 ± 0.0135 | 0.7468 | — | — |
| Decision Tree | 0.8260 ± 0.0439 | 0.8506 | — | — |
| Random Forest | 0.9428 ± 0.0083 | 0.8636 | — | — |
| Gradient Boosting | 0.9451 ± 0.0174 | 0.8766 | — | — |
| K-Nearest Neighbors | 0.8993 ± 0.0171 | 0.8766 | — | — |
| Tuned Random Forest | 0.9457 | 0.8701 | 0.8148 | 0.9498 |
| **Tuned Gradient Boosting** | **0.9461** | **0.8831** | **0.8302** | **0.9550** |

> 🏆 **Winner: Tuned Gradient Boosting** — Accuracy **88.31%** | F1 **0.8302** | ROC-AUC **0.9550**

---

## 🤖 AI Assistance Disclosure (~5% of Steps)

| Step | What AI Was Used For |
|---|---|
| **Step 1 — Data Collection** | Understanding the clinical meaning of `DiabetesPedigreeFunction` (a hereditary risk score from family history) and the dataset's origin at the National Institute of Diabetes and Digestive and Kidney Diseases, USA |
| **Step 6 — Model Evaluation** | Explaining why MAPE produces an astronomically large value (7.97 billion %) when true binary labels are 0, and when Brier Score or Log Loss is preferable as a probabilistic calibration metric |

All code, analysis logic, model selection, preprocessing decisions, and commentary were produced by the student. AI was used only as a reference tool for domain knowledge, not to generate solutions.

---

## 🛠️ How to Run

1. Click the **"Open in Google Colab"** badge above
2. Go to **Runtime → Run all**
3. The notebook downloads the dataset automatically — no manual upload needed
4. All outputs (plots, metrics, tables) are generated in sequence

**Requirements (all pre-installed in Colab):**
```
pandas  numpy  matplotlib  seaborn  scikit-learn
```

---

## 📚 References

- Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). *Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus.* Proceedings of the Symposium on Computer Applications and Medical Care.
- UCI Machine Learning Repository: [Diabetes Dataset](https://archive.ics.uci.edu/dataset/34/diabetes)
- Scikit-learn Documentation: [scikit-learn.org](https://scikit-learn.org/stable/)
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, pp. 2825–2830.

---

*Submitted as part of a Machine Learning course lab. All analysis performed in Google Colab.*
