Student Academic Recovery Prediction

A machine learning project that predicts whether a student is likely to recover academically using early-semester academic performance indicators.

📌 Project Overview

Early identification of academically at-risk students is essential for timely intervention.
This project applies supervised machine learning techniques to predict student academic recovery based on early-semester data such as enrolled units, evaluated units, approved units, age, and scholarship status. Additionally, multiple machine learning models were compared to identify the most effective approach for prediction.

The goal is to provide an early predictive signal before final academic outcomes are known.

📊 Dataset

Name: Student Dropout and Academic Success Dataset

Source: Kaggle

Link: https://www.kaggle.com/datasets/mahwiz/students-dropout-and-academic-success-dataset

The dataset contains anonymized academic and demographic data of higher education students, including:

Age at enrollment

Curricular units enrolled (1st semester)

Curricular units evaluated (1st semester)

Curricular units approved (1st semester)

Scholarship status

Final academic outcome (Target)

🧠 Problem Statement

Traditional academic evaluation systems identify struggling students too late.
This project aims to predict academic recovery using early academic indicators to enable proactive intervention.

🎯 Objective

To build a machine learning model that predicts whether a student is likely to recover academically using early-semester academic performance data.

⚙️ Machine Learning Models Used

1. Logistic Regression (Baseline Model)
2. Linear Regression (Converted for classification comparison)
3. Support Vector Machine (SVM)
4. Random Forest Classifier (Final Model)

🔍 Model Comparison

Multiple models were implemented to compare performance and ensure robust prediction.

- Logistic Regression was used as a baseline model.
- Linear Regression was used for comparative analysis after converting outputs into classification labels.
- Support Vector Machine (SVM) was used to handle complex decision boundaries.
- Random Forest was selected as the final model due to its ability to capture non-linear relationships.

The models were compared using evaluation metrics such as Accuracy, Precision, Recall, F1-score, and Confusion Matrix.


📈 Model Evaluation Metrics

The models were evaluated using:

1. Accuracy – Overall correctness of predictions  
2. Precision – Correct positive predictions  
3. Recall – Ability to identify actual positive cases  
4. F1-Score – Balance between precision and recall  
5. Confusion Matrix – Detailed class-wise prediction analysis  
6. Feature Importance – Identifies most impactful features (for Random Forest)
   

**Features Used**

1.Age at Enrollment

2.Curricular Units Enrolled (1st Semester)

3.Curricular Units Evaluated (1st Semester)

4.Curricular Units Approved (1st Semester)

5.Scholarship Holder

These features were selected based on:

1.Early availability

2.Relevance to academic performance

3.Literature review comparison

