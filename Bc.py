
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)  
y = pd.Series(data.target)  # Target (0 = Benign, 1 = Malignant)

print("Feature Data Shape:", X.shape)
print("Target Distribution:\n", y.value_counts())


sns.countplot(x=y,color="red")
plt.title("Target Class Distribution (0=Benign, 1=Malignant)")
plt.show()


plt.figure(figsize=(4,6))
sns.heatmap(X.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    if y_prob is not None:
        print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

print("Random Forest Evaluation:")
evaluate_model(rf_model, X_test, y_test)
print("\nSVM Evaluation:")
evaluate_model(svm_model, X_test, y_test)


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_
print("Tuned Random Forest Evaluation:")
evaluate_model(best_rf, X_test, y_test)


import joblib
joblib.dump(best_rf, "breast_cancer_rf_model.pkl")
print("Model saved as 'breast_cancer_rf_model.pkl'")
#two models :1.random forest 2.Support Vector Evaluation(SVM)


