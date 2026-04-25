# 🎓 Student Academic Recovery Prediction

A machine learning project that predicts whether a student is likely to recover academically using early-semester academic performance indicators.

---

## 📌 Project Overview

Early identification of academically at-risk students is essential for timely intervention.
This project applies supervised machine learning techniques to predict student academic recovery using early-semester data such as enrolled units, evaluated units, approved units, age, and scholarship status.

Multiple machine learning models were implemented and compared to identify the most effective approach for prediction.

👉 The goal is to provide an **early predictive signal before final academic outcomes are known**, enabling proactive academic support.

---

## 📊 Dataset

* **Name:** Student Dropout and Academic Success Dataset
* **Source:** Kaggle
* **Link:** https://www.kaggle.com/datasets/mahwiz/students-dropout-and-academic-success-dataset

The dataset contains anonymized academic and demographic data of higher education students, including:

* Age at enrollment
* Curricular units enrolled (1st semester)
* Curricular units evaluated (1st semester)
* Curricular units approved (1st semester)
* Scholarship status
* Final academic outcome (Target)

---

## 🧠 Problem Statement

Traditional academic evaluation systems identify struggling students too late.
This project aims to predict academic recovery using early academic indicators to enable **timely and proactive intervention**.

---

## 🎯 Objective

To build a machine learning model that predicts whether a student is likely to recover academically using early-semester academic performance data.

---

## ⚙️ Machine Learning Models Used

1. Logistic Regression (Baseline Model)
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors (KNN)
4. Random Forest Classifier
5. XGBoost Classifier (Best Performing Model)

---

## 🔍 Model Comparison

Multiple supervised learning models were implemented and evaluated:

* **Logistic Regression** – Used as a baseline model
* **SVM** – Effective for handling complex decision boundaries
* **KNN** – Instance-based learning for comparison
* **Random Forest** – Captures non-linear relationships using ensemble learning
* **XGBoost** – Gradient boosting technique that improves performance by correcting previous errors

📌 **Best Model:** XGBoost
📌 **Reason:** Highest F1-score and better handling of complex patterns in data

---

## 📈 Model Evaluation Metrics

The models were evaluated using:

1. **Accuracy** – Overall correctness
2. **Precision** – Correct positive predictions
3. **Recall** – Ability to detect actual positives
4. **F1-Score** – Balance between precision and recall (**Primary Metric**)
5. **Confusion Matrix** – Class-wise performance analysis

---

## 📊 Model Performance (F1 Score)

| Model         | F1 Score |
| ------------- | -------- |
| XGBoost       | ~0.90    |
| SVM           | ~0.89    |
| Logistic      | ~0.89    |
| Random Forest | ~0.89    |
| KNN           | ~0.88    |

---

## 🧾 Features Used

The following features were selected based on early availability and relevance:

1. Age at Enrollment
2. Curricular Units Enrolled (1st Semester)
3. Curricular Units Evaluated (1st Semester)
4. Curricular Units Approved (1st Semester)
5. Scholarship Holder

### 🔎 Feature Selection Rationale

* Available early in academic timeline
* Strong correlation with academic performance
* Supported by literature and domain understanding

---

## 📌 Key Insight

Students who **approve more curricular units in the first semester** are significantly more likely to recover academically.

---

## 🚀 How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train models

```
python src/train_model.py
```

### 3. Make predictions

```
python src/Predict.py
```

### 4. View feature importance

```
python src/feature_importance.py
```

---

## 📈 Visualizations

* Model comparison graph (F1 Score)
* Confusion matrices for all models
* Feature importance plots

---

## 🔮 Future Improvements

* Include additional demographic and behavioral features
* Apply advanced hyperparameter tuning
* Deploy as a web application
* Integrate real-time academic monitoring system

---

## 📌 Conclusion

This project demonstrates how machine learning can be effectively used for **early prediction of academic recovery**, helping institutions take proactive steps to support students and improve outcomes.

---
