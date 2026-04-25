import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from data_preprocessing import preprocess_data

warnings.filterwarnings("ignore")

print("Training started...")

data = preprocess_data("../dataset/student_data.csv")
print("Data loaded. Shape:", data.shape)

X = data.drop('Target', axis=1)
y = data['Target']

plt.figure(figsize=(6,4))
sns.countplot(x='Target', data=data)
plt.title("Class Distribution")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}
accuracies = {}
predictions = {}

lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

results["Logistic"] = f1_score(y_test, lr_pred)
accuracies["Logistic"] = accuracy_score(y_test, lr_pred)
predictions["Logistic"] = lr_pred

svm = SVC(kernel='rbf', C=10, gamma=0.05, probability=True)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)

results["SVM"] = f1_score(y_test, svm_pred)
accuracies["SVM"] = accuracy_score(y_test, svm_pred)
predictions["SVM"] = svm_pred

knn = KNeighborsClassifier(n_neighbors=9, weights='distance')
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)

results["KNN"] = f1_score(y_test, knn_pred)
accuracies["KNN"] = accuracy_score(y_test, knn_pred)
predictions["KNN"] = knn_pred

rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

results["Random Forest"] = f1_score(y_test, rf_pred)
accuracies["Random Forest"] = accuracy_score(y_test, rf_pred)
predictions["Random Forest"] = rf_pred

xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

results["XGBoost"] = f1_score(y_test, xgb_pred)
accuracies["XGBoost"] = accuracy_score(y_test, xgb_pred)
predictions["XGBoost"] = xgb_pred

print("\n===== MODEL PERFORMANCE =====")

for model in results:
    print(f"\n--- {model} ---")
    print(f"Accuracy: {accuracies[model]:.4f}")
    print(f"F1 Score: {results[model]:.4f}")
    print(classification_report(y_test, predictions[model]))

for model, pred in predictions.items():
    plt.figure(figsize=(4,3))
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d')
    plt.title(f"{model} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

best_model_name = max(results, key=results.get)
print("\nBest Model (based on F1 Score):", best_model_name)

with open("../model/logistic_recovery_model.pkl", "wb") as f:
    pickle.dump((lr, scaler), f)
    
with open("../model/svm_recovery_model.pkl", "wb") as f:
    pickle.dump((svm, scaler), f)
    
with open("../model/knn_recovery_model.pkl", "wb") as f:
    pickle.dump((knn, scaler), f)

with open("../model/random_forest_recovery_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("../model/xgboost_recovery_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

print("\nAll models saved successfully.")


sns.set_style("whitegrid")

sorted_models = sorted(results, key=results.get, reverse=True)
f1_scores = [results[m] for m in sorted_models]

plt.figure(figsize=(14,7))

colors = sns.color_palette("viridis", len(sorted_models))
colors[0] = (0.0, 0.6, 0.2)  

bars = plt.bar(sorted_models, f1_scores, color=colors, edgecolor='black', linewidth=1.5)

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + 0.005,
        f"{yval:.3f}",
        ha='center',
        fontsize=12,
        fontweight='bold'
    )

plt.ylim(0.6, 1.0)
plt.yticks(np.arange(0.6, 1.01, 0.05))  
plt.title("Model Comparison (F1 Score)", fontsize=20, fontweight='bold')
plt.xlabel("Models", fontsize=14)
plt.ylabel("F1 Score", fontsize=14)

plt.xticks(rotation=20, fontsize=12)
plt.yticks(fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
print("\n===== FINAL F1 SCORES =====")
for m in sorted_models:
    print(f"{m}: {results[m]:.4f}")
