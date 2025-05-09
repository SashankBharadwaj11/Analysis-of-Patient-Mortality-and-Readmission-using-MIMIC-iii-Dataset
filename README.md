# Analysis-of-Patient-Mortality-and-Readmission-using-MIMIC-iii-Dataset

# 🏥 Predicting Mortality and Readmission in Hospitalized Patients Using Machine Learning

This project builds a robust, interpretable, and comparative pipeline for predicting two critical healthcare outcomes:
- **In-hospital mortality**
- **30-day hospital readmission**

We utilize a real-world, de-identified dataset (MIMIC-III style) and apply various machine learning models to understand which patients are most at risk — enabling proactive care planning and better clinical outcomes.

---

## 📁 Project Structure

```bash
.
├── data/                  # Raw CSVs (ADMISSIONS.csv, PATIENTS.csv, etc.)
├── notebooks/             # Jupyter notebooks or Colab workspaces
├── models/                # Saved model artifacts (optional)
├── visuals/               # Plots: SHAP, ROC, PR curves, confusion matrices
├── README.md              # Project documentation
└── requirements.txt       # Dependencies
```
Objectives
1: Predict in-hospital mortality using structured clinical features

2: Predict 30-day readmission for survivors

3: Compare models: Logistic Regression, XGBoost, CatBoost, Decision Tree

4: Build a Voting Classifier ensemble

5: Apply SHAP for explainability and transparency

6: Visualize and interpret performance trade-offs (precision, recall, AUC, MCC)

Features Used
| Feature Type | Features                                                               |
| ------------ | ---------------------------------------------------------------------- |
| Numerical    | Length of stay, number of diagnoses, number of procedures              |
| Binary       | Sepsis flag, diabetes, ventilator use, insurance type, discharge group |
| Categorical  | Admission type, admission location, ethnicity                          |

Models Implemented
✅ Logistic Regression (L1/L2 regularization)

✅ XGBoost Classifier

✅ CatBoost Classifier (handles categorical features natively)

✅ Decision Tree Classifier

✅ Voting Ensemble (soft voting with weighting)


Evaluation Metrics
For both mortality and readmission:

ROC AUC

Precision, Recall, F1-score

Matthews Correlation Coefficient (MCC)

Confusion Matrix

SHAP Explanation Plots

Model Results (Summary)
Mortality Prediction
| Model                 | AUC      | MCC      | Sensitivity | Specificity |
| --------------------- | -------- | -------- | ----------- | ----------- |
| Logistic              | 0.83     | 0.35     | 0.73        | 0.79        |
| XGBoost               | 0.87     | 0.41     | 0.26        | 0.99        |
| CatBoost              | 0.88     | 0.42     | 0.28        | 0.99        |
| Decision Tree         | 0.85     | 0.40     | 0.79        | 0.77        |
| **Voting Classifier** | **0.86** | **0.40** | 0.30        | 0.98        |
Readmission Prediction (Survivors only)
| Model                 | AUC      | MCC      | Sensitivity | Specificity |
| --------------------- | -------- | -------- | ----------- | ----------- |
| Logistic              | 0.70     | 0.24     | 0.72        | 0.56        |
| XGBoost               | 0.71     | 0.12     | 0.05        | 0.99        |
| CatBoost              | 0.72     | 0.14     | 0.08        | 0.98        |
| Decision Tree         | 0.69     | 0.24     | 0.72        | 0.55        |
| **Voting Classifier** | **0.69** | **0.17** | 0.20        | 0.92        |
Key Insights
Mortality risk is best predicted with a blend of Logistic and CatBoost models. Logistic regression provides high recall (catching most true deaths), while boosting models are better at precision.

Readmission prediction is challenging. Most models struggle with sensitivity — many readmissions are missed unless recall is explicitly optimized.

Voting ensembles improve overall AUC and stability but may reduce sensitivity without threshold tuning.

SHAP analysis showed that key contributors include:

High number of diagnoses

Use of ventilator

Sepsis and diabetes flags

Admission type and insurance status
Future Work
Apply threshold optimization to improve recall for readmissions

Use temporal validation to simulate real-world prediction

Integrate clinical notes (embeddings) for better signal

Deploy via Flask or FastAPI for real-time scoring
Requirements
pip install pandas numpy matplotlib seaborn shap xgboost catboost scikit-learn




