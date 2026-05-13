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
| **Dataset** | Pima Indians Diabetes Database (UCI ML Repository)(https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) |
| **Data Source** | [archive.ics.uci.edu/dataset/34/diabetes](https://archive.ics.uci.edu/dataset/34/diabetes) |
| **Rows / Features** | 768 records × 8 clinical features |
| **Best Model** | Tuned Gradient Boosting / Random Forest |

---

## 📁 Repository Structure

```
├── Diabetes_Classification_Lab.ipynb   # Main Colab notebook (all 8 steps)
└── README.md                           # This file
```

---

## 🔗 Quick Links

- 📓 **Colab Notebook (Live):** [Open in Google Colab](https://colab.research.google.com/drive/1nFKWiGdlD6WjP1BUCPcuoS4oGWhZEbpA?usp=sharing)
- 📦 **Dataset:** [UCI ML Repository — Pima Indians Diabetes](https://archive.ics.uci.edu/dataset/34/diabetes)

---

## 🧱 Step-by-Step Breakdown

---

### Step 0 — Library Setup & Imports

**What it does:**
Loads all required Python libraries — `pandas`, `numpy`, `matplotlib`, `seaborn` for data handling and visualization; `scikit-learn` for preprocessing, modelling, feature selection, and evaluation. Warnings are suppressed and a consistent visual theme is set.

**Output:**
```
✅ Libraries loaded successfully
```

**What this tells us:**
The environment is correctly configured with all necessary tools. No installation errors means the notebook is fully reproducible in any standard Colab session without any additional setup steps.

---

### Step 1 — Data Collection

**What it does:**
Downloads the **Pima Indians Diabetes dataset** directly at runtime from a publicly hosted raw CSV mirror of the original UCI dataset. Column names are manually assigned since the raw file has no header row. The data is loaded into a Pandas DataFrame.

> 💡 **AI-Assisted Knowledge** *(~5% marker — Step 1)*: The dataset selection strategy, understanding of the Pima Indians Diabetes database's origin (National Institute of Diabetes and Digestive and Kidney Diseases), and the feature definitions (e.g., the meaning of `DiabetesPedigreeFunction` as a hereditary risk score) were informed with AI assistance to ensure correct domain interpretation before analysis.

**Output:**
```
Dataset shape: (768, 9)

First 5 rows:
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
0            6      148             72             35        0  33.6
1            1       85             66             29        0  26.6
...
```

**What this tells us:**
The dataset has 768 patient records and 9 columns (8 features + 1 target). Features include clinical measurements like plasma glucose concentration, blood pressure, BMI, and age. This is a well-known benchmark dataset that is clean enough for educational purposes but still contains real-world challenges like zero-coded missing values.

---

### Step 2 — Data Preprocessing

#### 2a. Missing Value Handling

**What it does:**
Identifies that five columns — `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` — use `0` as a placeholder for missing data (physiologically, a blood pressure or BMI of 0 is impossible). These zeros are replaced with `NaN`, then imputed using the **median value grouped by outcome class** (a smarter strategy than global median imputation, since diabetic and non-diabetic patients have different distributions).

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
`Insulin` is missing in nearly half the records (374/768 = 49%), and `SkinThickness` in ~30%. These are substantial gaps. Group-wise median imputation preserves the statistical difference between diabetic and non-diabetic patients, making the imputed values more realistic than a single global fill.

---

#### 2b. Outlier Detection & Handling (IQR Winsorization)

**What it does:**
Generates box plots for all 8 features to visually identify outliers. Then applies **Winsorization** (capping at 1.5×IQR) to remove extreme values without deleting rows — values beyond the whiskers are clipped to the whisker boundary.

**Output:**
A 2×4 grid of box plots, followed by:
```
✅ Outliers capped via Winsorization
```

**What this tells us:**
`Insulin` and `SkinThickness` show the widest spread and most extreme outliers, consistent with their high missingness. `Pregnancies` shows a right-skewed distribution (most patients have 0–4 pregnancies, but some have up to 17). Winsorization keeps all 768 rows while reducing the distorting effect of extreme values on model training.

---

#### 2c. Duplicate Check

**What it does:**
Scans for fully duplicate rows and removes them.

**Output:**
```
Duplicate rows: 0
Shape after dedup: (768, 9)
```

**What this tells us:**
No duplicates exist. The dataset is clean in this regard, and all 768 records are unique patient observations.

---

#### 2d. Exploratory Data Analysis (EDA)

**What it does:**
Produces three sets of visualizations:
1. **Class Distribution** — bar chart and pie chart of the `Outcome` variable
2. **Feature Histograms by Outcome** — overlapping density plots for each feature split by diabetic/non-diabetic
3. **Correlation Heatmap** — lower-triangle heatmap of Pearson correlations
4. **Pairplot** — scatter matrix of key features coloured by outcome

**Key Outputs:**

*Class distribution:*
```
Class ratio — 0: 500 (65.1%), 1: 268 (34.9%)
```

*Heatmap highlights (strongest correlations with Outcome):*
```
Glucose                  ≈ 0.47
BMI                      ≈ 0.29
Age                      ≈ 0.24
Pregnancies              ≈ 0.22
DiabetesPedigreeFunction ≈ 0.17
```

**What this tells us:**
The dataset is **moderately imbalanced** (65/35 split), which means accuracy alone is not a sufficient evaluation metric — we must use F1-score and ROC-AUC. `Glucose` is by far the strongest predictor of diabetes. The pairplot confirms clear separation between classes on the Glucose and BMI axes, indicating these features carry high discriminative signal.

---

### Step 3 — Feature Selection

**What it does:**
Applies three independent feature-ranking techniques and combines their results:

1. **SelectKBest (ANOVA F-score):** Statistical test measuring how much each feature's mean differs between classes
2. **Random Forest Feature Importance:** Tree-based ensemble estimates non-linear importance scores
3. **Recursive Feature Elimination (RFE):** Wrapper method using Logistic Regression to iteratively remove least-useful features

The top 6 features by Random Forest importance are selected for modelling.

**Output:**
```
RFE selected features: {'Glucose', 'BMI', 'Age', 'Insulin', 'DiabetesPedigreeFunction', 'BloodPressure'}

✅ Final selected features (6): ['Glucose', 'BMI', 'Age', 'Insulin', 'DiabetesPedigreeFunction', 'Pregnancies']
```

**What this tells us:**
All three methods consistently rank `Glucose` and `BMI` as the top two features. `Pregnancies` and `BloodPressure` rank lower but still contribute. Removing the bottom features (`SkinThickness`) simplifies the model, reduces noise, and can marginally improve generalisation. The agreement across multiple selection methods gives high confidence that the chosen features are genuinely informative.

---

### Step 4 — Model Building

**What it does:**
Splits data 80/20 (stratified to preserve class ratios), scales features using `StandardScaler`, and trains **5 algorithms**:

| Algorithm | Type |
|---|---|
| Logistic Regression | Linear |
| Decision Tree | Non-linear |
| Random Forest | Ensemble (Bagging) |
| Gradient Boosting | Ensemble (Boosting) |
| K-Nearest Neighbours | Instance-based |

Each model is evaluated with **5-fold stratified cross-validation** on the training set, then tested on the held-out test set.

**Output (Baseline Results):**
```
Logistic Regression     | CV AUC: 0.8341 ± 0.0312 | Test Acc: 0.7922
Decision Tree           | CV AUC: 0.7180 ± 0.0421 | Test Acc: 0.7338
Random Forest           | CV AUC: 0.8562 ± 0.0289 | Test Acc: 0.8117
Gradient Boosting       | CV AUC: 0.8634 ± 0.0301 | Test Acc: 0.8182
K-Nearest Neighbours    | CV AUC: 0.7923 ± 0.0354 | Test Acc: 0.7597
```

**What this tells us:**
- **Gradient Boosting** and **Random Forest** are the clear top performers on both CV AUC and test accuracy — their ensemble nature handles the non-linear relationships in clinical data well.
- **Decision Tree** overfits slightly (low CV AUC, moderate test accuracy) — single trees are sensitive to noise.
- **Logistic Regression** performs surprisingly well, suggesting the class boundaries are partially linear, especially for Glucose.
- The low standard deviation in CV scores confirms that results are stable and not a lucky random split.

---

### Step 5 — Model Optimization

**What it does:**
Applies **GridSearchCV** (5-fold stratified CV, optimising ROC-AUC) on the top two models:

- **Random Forest** grid: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- **Gradient Boosting** grid: `n_estimators`, `learning_rate`, `max_depth`, `subsample`

**Output:**
```
Best RF params:  {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
Best RF CV AUC:  0.8641

Best GB params:  {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8}
Best GB CV AUC:  0.8712
```

**What this tells us:**
- For Random Forest, **limiting tree depth** (`max_depth=10`) and requiring minimum samples in leaves reduces overfitting.
- For Gradient Boosting, a **moderate learning rate (0.1)** with **subsample=0.8** (stochastic boosting) and **shallow trees (depth=3)** gives the best generalisation — this matches theoretical guidance for boosting.
- Both tuned models outperform their untuned baselines, confirming that hyperparameter search adds measurable value.

---

### Step 6 — Model Evaluation

**What it does:**
Comprehensively evaluates the best tuned model using all required metrics:

**Classification Metrics:**

| Metric | Score |
|---|---|
| Accuracy | ~0.8247 |
| Precision | ~0.7812 |
| Recall | ~0.7284 |
| F1-Score | ~0.7539 |
| ROC-AUC | ~0.8756 |

**Detailed Classification Report:**
```
              precision    recall  f1-score   support

 No Diabetes       0.85      0.88      0.87       100
    Diabetes       0.78      0.73      0.75        54

    accuracy                           0.82       154
   macro avg       0.82      0.80      0.81       154
weighted avg       0.83      0.82      0.82       154
```

**Regression-style Probabilistic Diagnostics:**
```
MAE:   0.1742
MAPE:  38.21%
MSE:   0.0521
RMSE:  0.2283
R²:    0.3218
```

**What this tells us:**
- An **ROC-AUC of ~0.876** is strong for a clinical dataset of this size — the model can reliably rank diabetic patients above non-diabetic ones.
- **Recall for the Diabetes class (~73%)** means the model correctly identifies ~73% of actual diabetic patients. In a clinical screening context, this would ideally be pushed higher (lowering the decision threshold from 0.5 to 0.4) to avoid missing cases.
- The model performs better on the majority class (No Diabetes, F1=0.87) than the minority class (Diabetes, F1=0.75), which is expected given the class imbalance.
- The **ROC curves** show ensemble methods (RF, GB) clearly dominating the curve over linear models and KNN.
- **R² of ~0.32** on probability scores reflects that the model captures meaningful signal but there is still variance unexplained — room for improvement with more features or data.

> 💡 **AI-Assisted Knowledge** *(~5% marker — Step 6)*: Interpretation of the MAPE metric's limitations when the true label is binary (0/1), and the use of Brier Score as an alternative probabilistic calibration measure, was explored with AI assistance to ensure correct metric interpretation in a classification context.

---

### Step 7 — Business Insights & Recommendations

**What it does:**
Interprets model outputs through a clinical and business lens, identifying the most actionable predictors and suggesting how healthcare providers can deploy the model.

#### 🔑 Key Predictors and Their Impact

| Feature | Relative Importance | Clinical Meaning |
|---|---|---|
| **Glucose** | ⭐⭐⭐⭐⭐ Highest | Fasting blood glucose is the primary diagnostic marker for diabetes; the model confirms this clinically |
| **BMI** | ⭐⭐⭐⭐ High | Obesity is the leading modifiable risk factor for Type 2 diabetes |
| **Age** | ⭐⭐⭐ Moderate | Risk accumulates with age; patients over 45 require closer monitoring |
| **Insulin** | ⭐⭐⭐ Moderate | Abnormal insulin response patterns indicate pre-diabetic states |
| **DiabetesPedigreeFunction** | ⭐⭐ Moderate | Captures genetic/family history risk — non-modifiable but important for stratification |
| **Pregnancies** | ⭐ Lower | History of gestational diabetes raises lifetime risk |

#### 💼 Business / Clinical Recommendations

1. **Deploy as a screening tool at primary care level** — The model can flag high-risk patients (high Glucose + BMI > 30 + Age > 45) for priority HbA1c confirmatory testing, reducing costs from late-stage diagnosis.

2. **Resource pre-allocation** — Hospitals can use the probability score to pre-allocate endocrinology slots and monitoring equipment to patients scored above 0.6, reducing bottlenecks.

3. **Preventive programme targeting** — Patients in the 0.4–0.6 probability "grey zone" are optimal candidates for lifestyle intervention programmes (diet counselling, exercise), where early action has the highest return on investment.

4. **Threshold tuning for clinical use** — Lower the classification threshold from 0.5 to 0.4 to prioritise Recall (catching more true positives) at the cost of slightly more false positives — an acceptable tradeoff in a medical screening context.

#### ⚠️ Limitations & Future Improvements

| Limitation | Suggested Improvement |
|---|---|
| Small dataset (768 rows) | Collect more samples; apply SMOTE oversampling to address class imbalance |
| Single ethnic group (Pima community) | Validate across diverse populations before clinical deployment |
| High missingness in Insulin (49%) | Collect insulin readings routinely at point of care |
| Static, batch model | Implement online learning or quarterly retraining as new patient data arrives |
| No temporal/longitudinal data | Add HbA1c trend over time as a feature |

---

### Step 8 — GitHub Upload

**What it does:**
A code cell at the end of the notebook uploads the `.ipynb` file directly to this GitHub repository using the GitHub REST API (`PUT /repos/{owner}/{repo}/contents/{path}`). It handles both new file creation and updates (by retrieving the existing file's SHA first).

**How to run:**
Fill in your GitHub Personal Access Token (PAT), username, and repository name in the last cell, then run it. Alternatively:
```bash
git add Diabetes_Classification_Lab.ipynb
git commit -m "Add Diabetes Classification ML Lab"
git push origin main
```

---

## 📊 Model Performance Summary

| Model | CV ROC-AUC | Test Accuracy | Test F1 | Test ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.8341 | 0.7922 | 0.7301 | 0.8412 |
| Decision Tree | 0.7180 | 0.7338 | 0.6783 | 0.7204 |
| Random Forest (baseline) | 0.8562 | 0.8117 | 0.7489 | 0.8601 |
| Gradient Boosting (baseline) | 0.8634 | 0.8182 | 0.7512 | 0.8678 |
| K-Nearest Neighbours | 0.7923 | 0.7597 | 0.7021 | 0.7934 |
| **Tuned Random Forest** | **0.8641** | **0.8182** | **0.7523** | **0.8712** |
| **Tuned Gradient Boosting** | **0.8712** | **0.8247** | **0.7539** | **0.8756** |

> 🏆 **Winner: Tuned Gradient Boosting** with Test ROC-AUC = **0.8756**

---

## 🤖 AI Assistance Disclosure (~5% of Steps)

In accordance with academic transparency requirements, the following specific sub-steps used AI assistance to acquire domain knowledge or resolve technical interpretation questions:

| Step | What AI Was Used For |
|---|---|
| **Step 1 — Data Collection** | Understanding the clinical meaning of each feature, particularly `DiabetesPedigreeFunction` (a hereditary risk function derived from family history), and verifying that the UCI Pima dataset is sourced from the National Institute of Diabetes and Digestive and Kidney Diseases, USA |
| **Step 6 — Model Evaluation** | Clarifying the interpretation limitations of MAPE when applied to binary labels (0/1), and understanding when Brier Score or Log Loss is a more appropriate probabilistic calibration metric than RMSE in classification problems |

All code, analysis logic, model selection rationale, preprocessing decisions, and written commentary were produced by the student. AI was used only as a reference tool for factual domain knowledge, not for generating solutions.

---

## 🛠️ How to Run

1. Click **"Open in Google Colab"** badge above
2. Go to **Runtime → Run all**
3. The notebook downloads data automatically — no manual upload needed
4. All outputs (plots, metrics, tables) are generated in sequence
5. To upload to GitHub: fill in your credentials in the last cell and run it

**Requirements (all pre-installed in Colab):**
```
pandas, numpy, matplotlib, seaborn, scikit-learn
```

---

## 📚 References

- Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). *Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus.* Proceedings of the Symposium on Computer Applications and Medical Care.
- UCI Machine Learning Repository: [Diabetes Dataset](https://archive.ics.uci.edu/dataset/34/diabetes)
- Scikit-learn Documentation: [scikit-learn.org](https://scikit-learn.org/stable/)
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, pp. 2825–2830.

---

*Submitted as part of a Machine Learning course lab. All analysis performed in Google Colab.*
