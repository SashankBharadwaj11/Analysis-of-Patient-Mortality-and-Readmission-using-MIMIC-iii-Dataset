# Analysis-of-Patient-Mortality-and-Readmission-using-MIMIC-iii-Dataset

[Abstract](#Abstract) 

[Contents](#Contents)

[1 Introduction](#1-introduction)  
[1.1 Problem Statement](#11-problem-statement)  
[1.2 Objectives](#12-objectives)  
[1.3 Societal Applications](#13-societal-applications)  

[2 Literature Survey](#2-literature-survey)  
[2.1 Introduction](#21-introduction)  
[2.2 Analyzing the Existing System](#22-analyzing-the-existing-system)  

[3 Analysis](#3-analysis)  
[3.1 System Requirements](#31-system-requirements)  
[3.1.1 Hardware Requirements](#311-hardware-requirements)  
[3.1.2 Software Requirements](#312-software-requirements)  
[3.2 Analysis](#32-analysis)  

[4 Dataset Description](#4-Dataset-Description)  
[4.1 Overview](4.1-overview)  
[4.2 Data Cleaning and Preprocessing](4.2-Data-Cleaning-and-Preprocessing)

[5 Visualizations](#5-visualizations)  

[6 Feature Engineering](#6-Feature-Engineering)

[7 Model Development](#7-model-development)  

[8 Results](#8-results)  

[9](#9)  

[10 Conclusion and Future Scope](#10-conclusion-and-future-scope)  

[References](#references)

# Abstract

We propose a multimodal predictive modeling framework aimed at enhancing the prediction of in-hospital mortality and 30-day hospital readmission by integrating both structured and unstructured clinical data from the MIMIC-III dataset. Our approach combines quantitative features such as diagnoses, procedures, and demographic information with narrative insights derived from clinical notes using ClinicalBERT embeddings. This fusion allows us to capture a more holistic view of patient health by leveraging both numeric trends and contextual language patterns in clinical documentation.

To strengthen the temporal understanding of patient trajectories, we incorporate time-stamped clinical events and analyze how the progression of medical conditions impacts outcomes. The model is benchmarked against established baselines including GRU-D and BERT-only architectures to assess its predictive performance. Additionally, we utilize interpretable machine learning techniques such as SHAP to gain insights into model behavior and feature influence.

Ultimately, our goal is to build a robust, interpretable, and scalable system that supports clinicians in identifying high-risk patients early, thereby improving care delivery, reducing readmission rates, and saving lives.

# Contents

## 1 Introduction

In today’s data-driven healthcare era, technology is transforming how we understand and respond to patient needs. Hospitals around the world are moving toward smarter, more proactive care models, where predicting a patient’s health trajectory is just as important as treating the present illness. Imagine a system that could warn doctors, “This patient is at high risk of being readmitted within 30 days,” or alert nurses with insights like “This individual’s vital signs suggest a higher risk of mortality.” That’s the power of predictive modeling in healthcare.

Just like virtual assistants have revolutionized the way we handle daily tasks by understanding speech and making intelligent decisions, similar intelligence can be harnessed to support clinical decision-making. Hospital readmissions and in-hospital mortality are critical indicators of healthcare quality and resource utilization. Predicting these outcomes ahead of time could save lives, reduce healthcare costs, and allow for more personalized and preventive care.

With access to massive repositories of clinical data such as MIMIC-III and advancements in machine learning and natural language processing, especially using models like ClinicalBERT, we can now mine valuable insights from both structured data (like lab results, vitals, diagnoses) and unstructured data (like doctor’s notes). These insights enable building intelligent systems that can forecast patient outcomes, guide clinical interventions, and ultimately improve the quality of care. This project builds upon that vision—leveraging openly available clinical datasets and cutting-edge AI to build a virtual assistant for hospitals, capable of predicting readmissions and mortality risk, and supporting smarter, data-driven decisions at the bedside.

## 1.1 Problem Statement

Every year, thousands of patients are readmitted to hospitals within a short time after discharge, and many face increased risk of mortality due to undetected complications or lack of timely intervention. Despite advances in healthcare, predicting such outcomes remains a major challenge for hospitals and care providers. Doctors and nurses are often overwhelmed with large volumes of patient data, making it difficult to manually assess risks or anticipate deteriorations in real-time.

The main agenda of this project is to solve this problem. The solution is to build an intelligent assistant for hospitals that can automatically predict the likelihood of patient readmission and mortality based on both clinical records and physician notes. By using advanced machine learning models and natural language processing techniques, this system can support healthcare professionals in making informed decisions and improve patient outcomes.

## 1.2 Objectives

- The main objective of building a hospital readmission and mortality prediction system is to assist healthcare professionals in identifying high-risk patients early.

Another objective is to reduce preventable hospital readmissions and improve patient outcomes by enabling timely interventions.

Fundamental objectives that maximize the overall value of predictive healthcare systems are accuracy, reliability, and clinical relevance.

Design a user-friendly and interpretable system that can be integrated into existing hospital workflows with minimal disruption.

Improve efficiency in healthcare decision-making by automating the analysis of large-scale patient data using machine learning.

Enhance the use of both structured clinical data and unstructured physician notes through advanced natural language processing.

Provide actionable insights to doctors and caregivers, helping them prioritize care, manage resources better, and ultimately save lives.

## 1.3 Societal Applications

This application has significant implications for society, especially in improving healthcare delivery and saving lives through proactive medical intervention.

This system can help hospitals prioritize high-risk patients, reducing preventable deaths and enhancing overall patient care.

It can support overburdened healthcare systems by streamlining decision-making and reducing manual workload for medical staff.

This project is feasible for future upgrades, including integration with electronic health records and real-time alert systems for critical care.

It enhances healthcare equity by using data to identify at-risk individuals who might otherwise be overlooked.

It provides timely insights that enable better discharge planning, reducing the emotional and financial burden on patients and families.

By automating predictions of adverse outcomes, it allows physicians to focus more on treatment and human interaction rather than data crunching.

It ensures continuous data monitoring and analysis, helping institutions stay prepared and responsive to emergencies.

Even with emerging challenges in healthcare, this intelligent system supports consistent, evidence-based decision-making and better resource allocation.

# 2 Literation Survey

# 2.1 Introduction

This section presents a review of existing research and systems developed for predicting hospital readmissions and patient mortality. Numerous studies and healthcare analytics solutions have been proposed that leverage data-driven models and machine learning to address these critical issues. These systems typically focus on analyzing patient data to identify those at high risk of readmission or death, enabling timely interventions and better resource allocation.

# 2.2 Analyzing the Existing System

Many existing systems apply statistical methods and machine learning algorithms to electronic health records (EHRs) for risk prediction. Popular models include Logistic Regression, Decision Trees, Random Forests, and Support Vector Machines (SVMs). More recent approaches use deep learning methods like LSTM and CNNs to capture temporal patterns in clinical data.

Hospitals and research institutions utilize tools such as Python, R, and platforms like TensorFlow or Scikit-learn to build predictive models. Additionally, frameworks like Apache Spark are used to handle large-scale healthcare datasets.

One widely known model is the LACE index (Length of stay, Acuity of admission, Comorbidities, and Emergency visits), which helps estimate the risk of readmission. Other institutions use proprietary scoring systems integrated into their EHR systems.

These predictive systems have shown success in identifying at-risk patients, but challenges such as data imbalance, missing values, and interpretability of black-box models remain. Ensuring model fairness and privacy compliance is also essential in clinical applications.

# 3 Analysis

# 3.1 System Requirements

# 3.1.1 Hardware Requirements

# 3.1.2 Software Requirements

# 3.2 Analysis

# 4 Dataset Description

The dataset used in this project is derived from MIMIC-III v1.4, a publicly available critical care database. It consists of multiple tables that provide comprehensive information about patients, their hospital admissions, diagnoses, treatments, and medications. The data spans over 40,000 hospital admissions, offering valuable insights into patient care, outcomes, and hospital performance.

# 4.1 Overview

The dataset consists of 5 main tables, each providing a specific type of information about patients and their hospital stay. These datasets allow for a deep analysis of patient demographics, admission details, medical history (diagnoses and procedures), and treatments (medications).

The data enables a comprehensive approach to understanding patient outcomes, including mortality, readmission, and the impact of various diagnoses, treatments, and patient characteristics. This structured information is essential for building predictive models in healthcare analytics, such as predicting patient outcomes or identifying risk factors for readmission.

The primary datasets included are:

## 1. ADMISSIONS.csv

Description: Contains records related to patient admissions, discharge times, and outcomes.

Information Included:

Admission and discharge timestamps.

Admission types (e.g., emergency, elective).

Discharge locations and outcomes (e.g., home, hospital transfer, death).

## 2. PATIENTS.csv

Description: Provides demographic details of patients.

Information Included:

Patient ID, date of birth (DOB), and gender.

Date of death (DOD) for patients who passed away.

## 3. DIAGNOSES_ICD.csv

Description: Contains ICD-9 diagnosis codes assigned to patients during their hospitalization.

Information Included:

ICD-9 codes for diagnoses.

Admission ID linking to the ADMISSIONS.csv dataset.

## 4. PROCEDURES_ICD.csv

Description: Lists the ICD-9 procedure codes for procedures performed during the hospital stay.

Information Included:

ICD-9 codes for medical procedures.

Admission ID linking to the ADMISSIONS.csv dataset.

## 5. PRESCRIPTIONS.csv

Description: Contains medication prescriptions given to patients during their hospital stay.

Information Included:

Medication names, start, and end dates for prescriptions.

Admission ID and patient ID linking to ADMISSIONS.csv and PATIENTS.csv datasets.

# 4.2 Data Cleaning and Preprocessing

## Handling Missing Data

**DEATHTIME:** Missing for ~53,122 records because many patients did not die during their admission. This was expected and not imputed.

**LANGUAGE:** Missing for a large portion of patients. Imputed using a constant value "Unknown".

**MARITAL_STATUS: **Missing in 10,128 records. Filled with "Unknown" or categorized accordingly.

**RELIGION:** Also had missing values and was imputed similarly using a constant "Unknown".

**EDREGTIME and EDOUTTIME:** Missing in several records and considered irrelevant for the analysis (related to emergency department registration times). These columns were dropped entirely.

## Imputation Strategy

Constant Imputation: Used for categorical fields such as LANGUAGE, MARITAL_STATUS, and RELIGION. Missing values were replaced with a placeholder like "Unknown" to retain the records without introducing bias.

## Column Dropping

Columns with excessive or non-informative missing values (e.g., EDREGTIME, EDOUTTIME) were dropped to reduce noise and maintain model relevance.

## Overall Workflow

Identified missing values.

Determined relevance based on domain knowledge.

Applied appropriate imputation or dropped columns.

Ensured cleaned data was consistent, complete, and ready for modeling.

# 5 Visualizations

# 1: Correlation Heatmap
![image](https://github.com/user-attachments/assets/8c67f821-7143-496e-b618-9b5e381c1db3)

Length of Stay (LOS_DAYS): LOS_DAYS demonstrates a moderate positive correlation with both NUM_PROCEDURES and NUM_DIAGNOSES, suggesting that patients undergoing more procedures or diagnosed with more conditions tend to have longer hospital stays.

Mortality Outcome (HOSPITAL_EXPIRE_FLAG): This variable shows a weak negative correlation with LOS_DAYS (approximately -0.17), indicating that shorter hospital stays are slightly associated with higher mortality rates.

Multicollinearity Assessment: Most feature correlations are below 0.6, suggesting minimal multicollinearity—a favorable condition for machine learning models, as it reduces the risk of misleading relationships and model instability.

![image](https://github.com/user-attachments/assets/c0ae5402-655a-4218-a2a7-af40e00a9713)

![image](https://github.com/user-attachments/assets/93c89ec8-8f4f-4a89-a710-0a9fcc2536ac)
Out of a total of 58,976 patient cases, 12,456 individuals—approximately 21%—were readmitted to the hospital.

The remaining 46,520 patients (79%) did not experience a subsequent admission.

This notable readmission rate underscores the need for proactive discharge planning and timely interventions.

In effect, nearly one in five patients returned post-discharge, emphasizing the critical role of monitoring and mitigating risks at the point of discharge.

![image](https://github.com/user-attachments/assets/10c09913-73fe-4320-9706-f20e2360921c)

**Mechanical Ventilation (HAS_VENT):** Used in 55.1% of patients who died, mechanical ventilation shows a strong link to fatal outcomes, suggesting it is a key indicator of critical illness.

**Sepsis (HAS_SEPSIS):** Involved in 23.6% of deaths, sepsis stands out as a major contributing factor, emphasizing the life-threatening nature of severe infections.

**Diabetes (HAS_DIABETES):** Found in 21.3% of deceased cases, diabetes also plays a significant role in mortality risk.

Overall, the data indicates that more than half of the patients who died required ventilation, and nearly a quarter had sepsis—underscoring how serious medical conditions heavily impact hospital mortality.


![image](https://github.com/user-attachments/assets/9a424596-b1a0-4a6f-bb82-d07ffb5169a3)
![image](https://github.com/user-attachments/assets/85e4d548-8337-40f9-86bf-99c61c8f3fcf)

![image](https://github.com/user-attachments/assets/7add7dfc-d72d-4254-8649-f24c0f7cf042)

"Urgent and emergency admissions are associated with longer hospital stays and more diagnoses, highlighting their higher clinical complexity compared to elective cases."


![image](https://github.com/user-attachments/assets/030683c2-57d5-42ce-890a-23e514494800)

![image](https://github.com/user-attachments/assets/66860f9f-bc49-4acf-ba03-6a1d9d48d3a1)

publicly insured patients consistently have higher mortality rates across all procedure groups, suggesting possible disparities in outcomes related to insurance coverage and patient severity.

![image](https://github.com/user-attachments/assets/63ee46de-62b4-4ffe-98b2-6a2d307f8e4d)

# 6 Feature Engineering

## 1. Continuous Features

LOS_DAYS: Length of stay in the hospital (in days).

NUM_DIAGNOSES: Number of ICD-9 diagnosis codes assigned.

NUM_PROCEDURES: Number of procedures performed during admission.

## 2. Binary Flags

HAS_SEPSIS: Indicates whether the patient was diagnosed with sepsis.

HAS_DIABETES: Indicates presence of diabetes in diagnosis codes.

HAS_VENT: Indicates if mechanical ventilation was used during stay.

## 3. Insurance Type (Derived from INSURANCE)

INSURANCE_PUBLIC: Includes government programs (e.g., Medicare, Medicaid).

INSURANCE_PRIVATE: Includes private or employer-sponsored plans.

INSURANCE_OTHER: Self-pay, no insurance, or other forms.

## 4. Discharge Disposition (Grouped from DISCHARGE_LOCATION)

DISCHARGE_GROUP_HOME: Patient discharged to home/self-care.

DISCHARGE_GROUP_DEATH: Discharge resulted in in-hospital death.

## 5. Categorical Features

ADMISSION_TYPE: Nature of admission (e.g., emergency, elective).

ADMISSION_LOCATION: Department/location of initial care.

ETHNICITY: Self-reported ethnicity of the patient.

## 6. Targets

HOSPITAL_EXPIRE_FLAG: Binary target; 0 = survived, 1 = died during hospital stay.

FUTURE_ADMISSION: Binary target; 1 = readmitted within 30 days, 0 = not readmitted.

## 7. Preprocessing Pipeline

DateTime Conversion : ADMITTIME, DISCHTIME converted to datetime objects for time-based feature engineering (e.g., length of stay).

Standardization: Applied to continuous features (LOS_DAYS, NUM_DIAGNOSES, NUM_PROCEDURES).

One-hot encoding: Applied to categorical features (ADMISSION_TYPE, ADMISSION_LOCATION, ETHNICITY), with the first category dropped to avoid multicollinearity.

Scaling
Applied StandardScaler or MinMaxScaler to continuous features (LOS_DAYS, NUM_DIAGNOSES, NUM_PROCEDURES) to normalize the scale.

Passthrough
Applied to Binary features (HAS_SEPSIS, HAS_DIABETES, HAS_VENT, insurance and discharge flags) passed without transformation.

Automation
Entire preprocessing flow implementation is managed using ColumnTransformer and Pipeline from scikit-learn for modularity, consistency and efficient transformation.


# 7 Model Development


# 8 Results 


# 9 Conclusion and Future Scope

# References 


📄 Project Overview

This project focuses on predicting in-hospital mortality and 30-day readmission using the MIMIC-III clinical dataset. We explore multiple machine learning models (Logistic Regression, XGBoost, CatBoost, Decision Tree, and an Ensemble Voting Classifier) and conduct thorough preprocessing, feature engineering, model tuning, evaluation, and interpretation.

We utilize a real-world, de-identified dataset (MIMIC-III style) and apply various machine learning models to understand which patients are most at risk — enabling proactive care planning and better clinical outcomes.

---

📁 Dataset Description

Data is derived from MIMIC-III v1.4:

ADMISSIONS.csv: Admission/discharge timestamps and outcome

PATIENTS.csv: Demographics (DOB, gender, DOD)

DIAGNOSES_ICD.csv: ICD-9 diagnosis codes

PROCEDURES_ICD.csv: ICD-9 procedure codes

PRESCRIPTIONS.csv: Medication data

Filtered structured features are used in models. Text data (e.g., NOTEEVENTS.csv) is not yet included.

📁 Dataset Description

Data is derived from MIMIC-III v1.4:

ADMISSIONS.csv: Admission/discharge timestamps and outcome

PATIENTS.csv: Demographics (DOB, gender, DOD)

DIAGNOSES_ICD.csv: ICD-9 diagnosis codes

PROCEDURES_ICD.csv: ICD-9 procedure codes

PRESCRIPTIONS.csv: Medication data

Filtered structured features are used in models. Text data (e.g., NOTEEVENTS.csv) is not yet included.

📊 Feature Engineering

1. Continuous Features

LOS_DAYS: Length of stay (in days)

NUM_DIAGNOSES: Number of ICD-9 diagnosis codes

NUM_PROCEDURES: Number of procedures performed

2. Binary Flags

Derived using code prefixes or presence:

HAS_SEPSIS, HAS_DIABETES, HAS_VENT

Insurance: INSURANCE_PUBLIC, INSURANCE_PRIVATE, INSURANCE_OTHER

Discharge: DISCHARGE_GROUP_HOME, DISCHARGE_GROUP_DEATH, etc.

3. Categorical Features

ADMISSION_TYPE, ADMISSION_LOCATION, ETHNICITY

4. Targets

HOSPITAL_EXPIRE_FLAG: Binary (0=survived, 1=died)

future_admission: Binary (0=not readmitted within 30 days, 1=readmitted)

🔧 Preprocessing Pipeline

Standardization: Applied to numeric features

One-hot encoding: Applied to categorical features (drop first category)

Passthrough: Binary features

Managed using ColumnTransformer and Pipeline in scikit-learn


🧠 Models Evaluated

1. Logistic Regression

penalty: l1/l2

C: Regularization strength

solver: liblinear

class_weight='balanced'

Evaluation: ROC AUC, Precision/Recall, MCC

2. XGBoost (XGBClassifier)

Tuned with RandomizedSearchCV

Grid: n_estimators, max_depth, learning_rate, subsample

Best AUC (Mortality): 0.872

SHAP used for interpretation

3. CatBoostClassifier

CatBoost handles categorical features natively

Grid: iterations, learning_rate, depth, l2_leaf_reg

Best AUC (Mortality): 0.876

Strong performance, especially on mortality prediction

4. DecisionTreeClassifier

Grid: max_depth, min_samples_split, min_samples_leaf, criterion

Simpler and interpretable

Best AUC (Mortality): 0.848

5. Ensemble Voting Classifier

Combines LogisticRegression, XGBoost, and Decision Tree

voting='soft' with weights: (LR=1, XGB=2, DT=1)

Best AUC (Mortality): 0.862


🔝 Performance Summary

🔥 Mortality Models (Target: HOSPITAL_EXPIRE_FLAG)

Model

AUC

MCC

Sensitivity

Specificity

Logistic Reg

0.833

0.351

0.733

0.786

XGBoost

0.872

0.409

0.260

0.990

CatBoost

0.876

0.421

0.278

0.989

DecisionTree

0.848

0.390

0.790

0.770

Ensemble

0.862

0.401

0.305

0.981

🛋️ Readmission Models (Target: future_admission)

Model

AUC

MCC

Sensitivity

Specificity

Logistic Reg

0.697

0.236

0.719

0.560

XGBoost

0.714

0.121

0.054

0.988

CatBoost

0.716

0.143

0.082

0.980

DecisionTree

0.692

0.190

0.720

0.550

Ensemble

0.691

0.165

0.197

0.923

📊 Visualizations

ROC & PR Curves for each model

Confusion matrices (with ConfusionMatrixDisplay)

SHAP summary and bar plots for XGBoost, CatBoost, and Decision Trees

💼 Model Interpretability (SHAP)

Used shap.Explainer and TreeExplainer for SHAP values

summary_plot and bar plots used for feature ranking

Ensured inputs to SHAP matched post-transform feature arrays

🚀 Deployment Ready

Preprocessing and modeling wrapped in Pipeline

Suitable for deployment via Streamlit or Flask

Future plans include real-time prediction with ClinicalBERT note embeddings

⚠️ Known Issues

Readmission models show class imbalance (sensitivity remains low)

Future work needed on threshold tuning or SMOTE

SHAP for ensemble not directly supported; interpret individual components

🔬 Technologies Used

Python, pandas, numpy

scikit-learn, XGBoost, CatBoost

SHAP, matplotlib, seaborn

💡 Future Work

Integrate clinical note embeddings (e.g., ClinicalBERT)

Use temporal features like admission frequency or time since last admission

Experiment with stacking classifiers or AutoML

Build a Streamlit UI for end-to-end interaction

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




