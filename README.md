# Credit Card Fraud Detection

Binary classification on a highly imbalanced dataset of real credit card transactions. The focus is on model comparison, threshold optimization, and translating model outputs into business decisions.

---

## Problem

Fraud detection is a class imbalance problem: only **0.17% of transactions are fraudulent** (492 out of 284,807). A model that predicts "not fraud" for everything achieves 99.8% accuracy — but catches zero fraudsters. The useful metric here is **Precision-Recall AUC**, not accuracy.

The goal is not just to maximize detection, but to find the right operating point given a business constraint: how many legitimate users are you willing to block to catch more fraud?

---

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraud cases. Features V1–V28 are PCA-transformed anonymized variables. Only `Time` and `Amount` are in their original form.

---

## Approach

**Validation strategy:** TimeSeriesSplit (10 folds) — transactions are sorted by time to avoid data leakage, reflecting real deployment conditions where you train on past data and predict future fraud.

**Class imbalance:** RandomOverSampler applied inside the pipeline on training folds only — never on the validation or test set.

**Models trained and compared:**
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- Soft Voting Ensemble (LR + RF + GB)
- Stacking Classifier

**Tuning:** GridSearchCV optimizing for Average Precision (PR AUC) on each fold.

**Feature analysis:** t-tests to identify statistically significant features, permutation importance across all three base models, SHAP values for the Gradient Boosting model.

---

## Results

| Model | PR AUC (Test) |
|---|---|
| Logistic Regression | 0.745 |
| Gradient Boosting | 0.783 |
| Soft Voting Ensemble | 0.797 |
| **Random Forest** | **0.804** |

Random Forest achieves the best test PR AUC of **0.804**.

---

## Threshold Optimization & Business Impact

PR AUC is a summary metric — in practice, you choose a threshold based on business priorities. Two operating points were analyzed:

**High recall operating point** (catch 70% of fraud):
- Precision: 0.971
- Frauds caught: 66 out of 94
- Legitimate users incorrectly flagged: 2 out of 71,108 (0.003%)

**High precision operating point** (flag only when 90%+ confident):
- Precision: 0.909
- Recall: 0.745

The key insight: at a 70% recall threshold, the model flags only 2 legitimate users incorrectly — a very low false positive rate that minimizes operational cost (manual review, customer friction, chargebacks) while still catching the majority of fraud.

---

## Error Analysis

False negatives and false positives were analyzed across all three models to understand where they agree and disagree:

- **11 transactions** were missed by all three models simultaneously — likely edge cases with unusual feature profiles
- **43 transactions** were incorrectly flagged as fraud by all three models — these share similar feature distributions to true fraud, particularly in V13 and V3

---

## Stack

`Python` · `scikit-learn` · `imbalanced-learn` · `pandas` · `numpy` · `scipy` · `eli5` · `SHAP` · `matplotlib` · `seaborn`
