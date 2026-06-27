# Credit Risk Analysis — Predicting Loan Default

An end-to-end data science project analyzing what drives loan default and building models to predict it, using the "Give Me Some Credit" dataset (150,000 borrowers). The focus is rigorous analysis — exploratory data analysis, honest model comparison on metrics appropriate for imbalanced data, and interpretation of what actually drives credit risk.

---

## The question

Given a borrower's financial profile, how likely are they to experience serious financial distress (default) in the next two years — and what factors drive that risk?

---

## Dataset

- **Source:** "Give Me Some Credit" (Kaggle)
- **Size:** 150,000 borrowers, 10 features + binary target
- **Target:** `SeriousDlqin2yrs` (1 = serious delinquency, 0 = none)
- **Key characteristic:** highly imbalanced — only **6.7%** of borrowers default

---

## Key findings

**1. Default risk falls steadily with age.**
Borrowers under 30 default at 11.6%, more than 5x the rate of those over 70 (2.3%).

**2. Past late payments are a powerful predictor.**
Default rate climbs from 4% (never late) to ~50% (late 5+ times) — a 12x increase.

**3. Multicollinearity among delinquency features.**
The three late-payment columns are nearly perfectly correlated (0.98-0.99) — they capture the same underlying behavior (repeat delinquency). Noted for model choice.

**4. Credit utilization is the top driver — and correlation alone misses it.**
The Random Forest ranks credit utilization as the #1 driver of default (importance 0.27), yet it had near-zero *linear* correlation with the target. The relationship is non-linear (risk concentrates at very high utilization), which simple correlation underestimates — a reminder that feature importance from a non-linear model reveals drivers that correlation misses.

---

## Models & results

Both models use `class_weight="balanced"` to handle the 6.7% default rate. Evaluated on a stratified 80/20 split using ROC-AUC, precision, and recall — **not accuracy**, which is misleading on imbalanced data (predicting "no default" for everyone scores 93% accuracy while being useless).

| Model | ROC-AUC | Precision (Default) | Recall (Default) | F1 (Default) |
|---|---|---|---|---|
| Logistic Regression (scaled) | 0.8021 | 0.18 | 0.67 | 0.29 |
| Random Forest | **0.8427** | **0.57** | 0.16 | 0.25 |

**The trade-off:** Random Forest ranks risk better overall (higher ROC-AUC) and is more precise, but catches fewer defaulters. Logistic regression catches far more defaulters (67% recall) at the cost of more false alarms. The right model depends on the lender's cost structure — whether a missed defaulter or a wrongly rejected applicant is more expensive.

---

## What drives default (feature importance)

Top drivers from the Random Forest:
1. Revolving credit utilization (0.27)
2. Debt ratio (0.14)
3. Monthly income (0.12)
4. Age (0.12)
5. Late-payment history (~0.09 each)

---

## Approach (notebook walkthrough)

1. **Data understanding** — shape, types, missing values, target balance
2. **EDA** — default rate by age, by late-payment count, correlation heatmap
3. **Cleaning** — median imputation for income (skewed, ~20% missing), 0 for missing dependents
4. **Modeling** — stratified split, logistic regression (scaled, in a pipeline) and random forest, both class-balanced
5. **Comparison** — ROC-AUC / precision / recall / F1 side by side
6. **Interpretation** — feature importance and business conclusion

---

## Tech stack

Python, pandas, numpy, matplotlib, seaborn, scikit-learn

---

## Business conclusion

The analysis identifies the key drivers of default: high credit utilization, high debt ratio, lower income, younger age, and a history of late payments. A Random Forest ranks default risk with ROC-AUC 0.84. For a lender, the model can prioritize applications for review — but the decision threshold should be tuned to the business cost of missed defaulters vs. wrongly rejected applicants. Richer features (employment history, loan purpose) would likely improve precision.

---

## Repo structure

```
credit-risk-analysis/
├── notebooks/
│   └── credit_risk_analysis.ipynb   # the full analysis
├── README.md
└── requirements.txt
```

> Dataset not committed — download from Kaggle ("Give Me Some Credit") and place `cs-training.csv` alongside the notebook.

---

## Limitations & next steps

- Default threshold not yet tuned to a cost function — a real deployment would optimize it to the lender's economics.
- Could add gradient boosting (XGBoost/LightGBM) and SHAP for deeper interpretability.
- Feature engineering (utilization bands, income-to-debt interactions) could lift precision.

---

> Educational / portfolio project demonstrating end-to-end data science: EDA, cleaning, modeling, evaluation, and interpretation.
