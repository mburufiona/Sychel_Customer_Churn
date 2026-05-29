
# Sychel Customer Churn Application

Predictive classification project to identify customers at high risk of churn for a telecommunications stakeholder. Built with a recall-first objective to minimize missed churners, using interpretable baselines and iterative modeling with appropriate preprocessing for an imbalanced dataset.

## Overview

- Goal: Predict whether a customer will churn based on their usage patterns, service plans, and account information.

- Why it matters: Retaining customers is less costly than acquiring new ones; proactively identifying churn risk enables timely interventions.

- Approach: Full ML workflow — data understanding, feature engineering, stratified train/test split, class-imbalance handling, baseline and tuned models, recall/PR-AUC-driven evaluation, and business recommendations.

## Business and Data Understanding

### Stakeholder
- Customer retention team in a telecommunications company. They need to catch as many potential churners as possible (high recall) while keeping outreach costs reasonable.

### Business Problem
- Customer churn is a major challenge for telecommnication companies as acquiring new customers is more costly than retaining existing ones. The customer retention team's goal is to proactively identify customers at high risk of churning and try to help before they leave.
- Missed churners (false negatives) are expensive: if not flagged, they may leave before action can be taken. Therefore, recall on the churn class (class 1) is the priority metric.

### Dataset
- Source: Churn dataset (CSV) with 3,333 customers.
- Key features: State, area code, international plan, voice mail plan, usage metrics (minutes, calls), customer service calls, etc.
- Target: churn (boolean).

### Data Characteristics
- Imbalanced target: ~14.5% churners vs ~85.5% non-churners.
- Several correlated variables by design (e.g., charges are linear transforms of minutes).

 
## Exploratory Data Analysis (EDA)

We explored the target distribution and key behavioral patterns to guide modeling and feature engineering:

- Class distribution (countplot of churn)
  - The churn class is the minority (~14.5%), confirming an imbalanced problem. Accuracy alone would be misleading; recall and PR-AUC are more appropriate for evaluation.

- Usage distribution (histogram of total day minutes)
  - Most customers exhibit moderate usage with a long tail of heavier users. This motivated log transforms for usage minutes to reduce skew and stabilize models.

- Customer service calls vs churn (boxplot)
  - Churners tend to have higher customer service calls, consistent with dissatisfaction. This informed creating service_call_rate and keeping customer service calls as key predictors.

- Correlation heatmap (numeric features)
  - Charges are highly correlated with minutes (by design), so charge columns were dropped to avoid overweighting the same signal.
  - Usage metrics across periods are moderately correlated; engineered averages and totals help summarize intensity cleanly.

These EDA findings directly informed our data preparation (dropping charge columns, engineering averages and service_call_rate, applying log transforms) and our metric choice (recall, PR-AUC) for an imbalanced classification problem.

 

## Modeling

### Data Preparation and Feature Engineering
- Dropped redundant “charge” features:
  - total day/eve/night/intl charge removed to avoid overweighting the same signal (charges = minutes × rate).
- Encodings:
  - Binary categorical features: international plan, voice mail plan mapped to 0/1.
  - area code one-hot encoded (drop_first=True).
  - state dropped to avoid high-cardinality and reduce overfit risk; area code retained as a regional proxy.
- Feature engineering:
  - Average call duration by period (day/eve/night/intl) with safe division-by-zero handling.
  - total_calls (sum of calls across periods).
  - service_call_rate = customer service calls / (total_calls + 1).
  - log transforms for usage minutes to reduce skew.
- Train/test split:
  - Stratified split (80/20) with fixed random_state=42 to preserve class distribution and reproducibility.
- Scaling:
  - StandardScaler fit on training data and applied to test data for Logistic Regression only.
  - Tree-based models trained/evaluated on unscaled features.

### Models Considered
- Logistic Regression (baseline):
  - class_weight='balanced' and max_iter=1000; trained on scaled features.
- Decision Tree (baseline and tuned):
  - class_weight='balanced', trained on unscaled features.
  - Hyperparameter tuning via GridSearchCV with scoring='recall' to prioritize the business objective.

### Preventing Data Leakage
- Transformers (e.g., StandardScaler) fit on training only, then applied to test.
- Grid search and model training performed on training data; test data only used for final evaluation.

## Evaluation

Given the class imbalance and business need to minimize false negatives:
- Primary metrics: Recall (class 1) and PR-AUC (Average Precision).  
- Secondary metrics: ROC-AUC, accuracy, precision, F1-score.
- Visual diagnostics: Confusion matrices, ROC curves, Precision-Recall curves.

### Results (Test Set)
- Logistic Regression:
  - Recall ≈ 0.711, PR-AUC ≈ 0.429, ROC-AUC ≈ 0.819
- Decision Tree (baseline):
  - Recall ≈ 0.309, PR-AUC ≈ 0.479, ROC-AUC ≈ 0.741
- Tuned Decision Tree (final model):
  - Recall ≈ 0.752, PR-AUC ≈ 0.598, ROC-AUC ≈ 0.803
- Accuracy ≈ 0.79

The tuned Decision Tree achieved the best combination of high recall and PR-AUC, aligning with the business objective while maintaining strong overall performance.


## Final Model

- Selected Model: Tuned Decision Tree (class_weight='balanced'), tuned via recall-focused GridSearchCV.
- Why:
  - Highest recall and superior PR-AUC among candidates, meaning better identification of churners with fewer unnecessary alerts than the baseline tree or logistic regression.
  - Interpretability: Feature importances indicate operational drivers (e.g., customer service calls and usage intensity).

## Insights

- Strong churn indicators:
  - High customer service calls and higher service_call_rate suggest dissatisfaction and are consistently associated with churn.
  - Usage intensity features (averages and totals) help capture behavioral patterns relevant to churn risk.

## Recommendations

- Decision policy: Use the tuned Decision Tree with a recall-oriented threshold to flag high-risk customers.
- Operational playbook:
  - Proactively contact flagged customers with targeted retention offers or plan reviews.
  - Pay extra attention to customers with high service_call_rate and specific usage patterns.
- KPI monitoring:
  - Track recall and precision of churn flags, and measure program ROI (savings vs outreach costs).
  - Adjust thresholds to balance cost and coverage.
- Maintenance:
  - Retrain and recalibrate periodically (e.g., quarterly) to address data drift.
  - Extend features (e.g., ticket severity, recent plan changes, tenure trends) to further improve recall and PR-AUC.

## Limitations

- Segment performance:
  - Customers with very low usage may be harder to classify; consider segment-specific thresholds.
- Data drift:
  - Behavioral and seasonal changes may alter relationships; monitor and retrain regularly.
- Feature scope:
  - No direct measures of satisfaction (NPS, survey ratings) were included; adding them could improve performance.
- Trade-offs:
  - Higher recall can increase false positives and outreach volume; precision should be monitored for operational efficiency.

## How to Run

- Notebook: Open `index.ipynb` and run cells sequentially.
- Data path: Ensure `data/bigml_59c28831336c6604c800002a.csv` is present relative to the notebook.
- Dependencies:
  - Python 3.x
  - pandas, numpy, matplotlib, seaborn
  - scikit-learn



## Conclusion

This project delivers a recall-first churn prediction model suitable for proactive retention. The tuned Decision Tree provides the best recall and PR-AUC among candidates while remaining interpretable for business stakeholders. With , ongoing monitoring, and periodic retraining, the model can be deployed to support timely and cost-effective customer retention strategies.

# Tableau link
Customer Churn Dashboard["https://public.tableau.com/app/profile/fiona.mburu/viz/CustomerChurnAnalysis_17800659420390/CustomerChurnDashboard#1"]
