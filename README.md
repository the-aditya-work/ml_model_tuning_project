# 🧠 Model Evaluation and Hyperparameter Tuning

This project demonstrates training, evaluating, and optimizing multiple machine learning models using the **Breast Cancer Wisconsin** dataset. It compares model performance using key metrics and applies hyperparameter tuning techniques like **GridSearchCV** and **RandomizedSearchCV** to improve performance.

---

## 📌 Features

- Load and preprocess data from `sklearn.datasets`
- Train multiple models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
- Evaluate model performance using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Hyperparameter tuning using:
  - 🔧 `GridSearchCV` (Random Forest)
  - 🎲 `RandomizedSearchCV` (SVC)
- Final evaluation of best models after tuning

---

## 📊 Results Summary

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 97.37%   | 97.22%    | 98.59% | 97.90%   |
| SVM (Before Tune)  | 97.37%   | 97.22%    | 98.59% | 97.90%   |
| Random Forest (Before Tune) | 96.49% | 95.89% | 98.59% | 97.22% |
| SVM (Tuned)        | **97.37%** | **97.00%** | **99.00%** | **97.00%** |
| Random Forest (Tuned) | 96.00% | 96.00% | 99.00% | 96.00% |

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name


