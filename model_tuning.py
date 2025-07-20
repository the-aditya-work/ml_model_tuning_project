import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}

# Evaluate models
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

# Display initial results
print("\n Model Performance (Before Tuning):\n")
print(pd.DataFrame(results).T)

# GridSearchCV for Random Forest

print("\n Running GridSearchCV for Random Forest...\n")

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train, y_train)

print(" Best Parameters for Random Forest:", grid_rf.best_params_)

# RandomizedSearchCV for SVC

print("\n Running RandomizedSearchCV for SVC...\n")

param_dist_svc = {
    'C': uniform(0.1, 10),
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

random_svc = RandomizedSearchCV(SVC(), param_distributions=param_dist_svc, n_iter=10, cv=5,
                                scoring='f1', n_jobs=-1, random_state=42)
random_svc.fit(X_train, y_train)

print(" Best Parameters for SVC:", random_svc.best_params_)

# Evaluate Tuned Models

print("\n Evaluation of Tuned Models:\n")

# Random Forest (Tuned)
best_rf = grid_rf.best_estimator_
rf_pred = best_rf.predict(X_test)
print("Random Forest (Tuned):")
print(classification_report(y_test, rf_pred))

# SVC (Tuned)
best_svc = random_svc.best_estimator_
svc_pred = best_svc.predict(X_test)
print("SVC (Tuned):")
print(classification_report(y_test, svc_pred))
