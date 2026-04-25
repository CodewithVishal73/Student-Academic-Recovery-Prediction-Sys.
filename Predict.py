import pickle
import pandas as pd

print("\n=== Student Academic Recovery Prediction ===\n")

# ================= MODEL SELECTION =================
print("Select Model:")
print("1. Logistic Regression")
print("2. SVM")
print("3. KNN")
print("4. Random Forest")
print("5. XGBoost")

try:
    choice = int(input("Enter choice (1-5): "))
except:
    print("Invalid input")
    exit()

# ================= LOAD MODEL =================
scaler = None

if choice == 1:
    with open("../model/logistic_recovery_model.pkl", "rb") as f:
        model, scaler = pickle.load(f)
    model_name = "Logistic Regression"

elif choice == 2:
    with open("../model/svm_recovery_model.pkl", "rb") as f:
        model, scaler = pickle.load(f)
    model_name = "SVM"

elif choice == 3:
    with open("../model/knn_recovery_model.pkl", "rb") as f:
        model, scaler = pickle.load(f)
    model_name = "KNN"

elif choice == 4:
    with open("../model/random_forest_recovery_model.pkl", "rb") as f:
        model = pickle.load(f)
    model_name = "Random Forest"

elif choice == 5:
    with open("../model/xgboost_recovery_model.pkl", "rb") as f:
        model = pickle.load(f)
    model_name = "XGBoost"

else:
    print("Invalid choice")
    exit()

# ================= USER INPUT =================
try:
    age = int(input("Enter Age at enrollment: "))
    enrolled = int(input("Enter number of curricular units enrolled (1st sem): "))
    evaluations = int(input("Enter number of curricular units evaluated (1st sem): "))
    approved = int(input("Enter number of curricular units approved (1st sem): "))
    scholarship = int(input("Scholarship holder? (1 = Yes, 0 = No): "))
except:
    print("Invalid input. Please enter numeric values.")
    exit()

# ================= CREATE DATAFRAME =================
student_data = pd.DataFrame([{
    'Age at enrollment': age,
    'Curricular units 1st sem (enrolled)': enrolled,
    'Curricular units 1st sem (evaluations)': evaluations,
    'Curricular units 1st sem (approved)': approved,
    'Scholarship holder': scholarship
}])

# ================= APPLY SCALING =================
if scaler is not None:
    student_data = scaler.transform(student_data)

# ================= PREDICTION =================
prediction = model.predict(student_data)

print(f"\n--- Prediction using {model_name} ---")

if prediction[0] == 1:
    print("✅ Student is likely to RECOVER academically")
else:
    print("❌ Student is NOT likely to recover")

# ================= PROBABILITY =================
if hasattr(model, "predict_proba"):
    probability = model.predict_proba(student_data)
    print(f"📊 Recovery Probability: {probability[0][1]*100:.2f}%")
else:
    print("ℹ️ Probability not available for this model")