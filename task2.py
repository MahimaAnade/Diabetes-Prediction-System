# task2.py

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load processed data
X_train, X_test, y_train, y_test = joblib.load("processed_data.pkl")

# Define classifiers
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

print("\n=== Model Accuracy Comparison ===\n")
best_model = None
best_score = 0
best_model_name = ""

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy = {acc:.4f}")

    if acc > best_score:
        best_score = acc
        best_model = model
        best_model_name = name

print(f"\nBest Model: {best_model_name} with Accuracy: {best_score:.4f}\n")
print("Classification Report:\n")
print(classification_report(y_test, best_model.predict(X_test)))

# Save best model
joblib.dump(best_model, "diabetes_best_model.pkl")
print("✅ Best model saved as 'diabetes_best_model.pkl'")
