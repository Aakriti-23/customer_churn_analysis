# Telco Customer Churn Analysis

End-to-end churn prediction pipeline - exploratory analysis, four ML models, SHAP explainability, business threshold optimization, and a Power BI dashboard for stakeholder reporting.

---

## What's Inside

| File | Description |
|---|---|
| `Customer_Churn_Analysis.ipynb` | Full analysis: EDA → ML → explainability → export |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | IBM Telco dataset (7,043 customers, 21 features) |
| `customer_churn_dashboard.pbix` | Power BI dashboard connected to enriched predictions |
| `assets/` | Dashboard screenshots |

---

## Dataset

The IBM Telco Customer Churn dataset covers 7,043 telecom subscribers across demographics, services, account details, and a binary churn label.

**Key features:**
- **Demographics** - gender, senior citizen status, partner, dependents
- **Services** - phone, internet (DSL / Fiber optic / None), streaming, online security, tech support
- **Account** - contract type, payment method, paperless billing, tenure, monthly charges, total charges
- **Target** - `Churn` (Yes / No)

---

## Notebook Walkthrough

### 1 · Data Cleaning
- Dropped `customerID` (non-informative)
- Coerced `TotalCharges` to numeric; removed 11 null rows (zero-tenure new customers)
- Remapped `SeniorCitizen` from 0/1 to No/Yes

### 2 · Exploratory Data Analysis
Interactive charts built with **Plotly** and **Seaborn**:
- Overall churn rate and gender distribution
- Churn by contract type, payment method, internet service, senior citizen status, partner/dependent status
- Heatmap of churn rate across contract × payment method segments

### 3 · Model Training
Four classifiers trained inside `sklearn` pipelines (StandardScaler + OneHotEncoder):

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Ensemble, feature importance |
| XGBoost | Best performer - selected for production |
| LightGBM | Evaluated at default (0.5) and recall-optimized (0.3) thresholds |

All models evaluated on ROC-AUC, accuracy, precision, recall, F1, confusion matrix, and 5-fold cross-validated ROC-AUC.

### 4 · Business Threshold Optimization
Instead of the default 0.5 threshold, the notebook sweeps thresholds from 0.05 → 0.90 and computes net business value at each point:

```
Net Value = (True Positives × $500 revenue saved) − (All Flagged × $50 retention cost)
```

The optimal threshold is selected to maximize net value - balancing recall against unnecessary outreach spend.

### 5 · Explainability (SHAP)
SHAP `TreeExplainer` applied to XGBoost:
- Beeswarm summary plot (direction + magnitude per feature)
- Bar plot (mean absolute SHAP values)
- Top drivers: `tenure`, `MonthlyCharges`, `Contract`, `TotalCharges`, `InternetService`

### 6 · Export for Power BI
The enriched dataset is written to PostgreSQL via SQLAlchemy with added prediction columns:

| Column | Description |
|---|---|
| `Churn_Probability` | Model's predicted probability (0–1) |
| `Predicted_Churn` | Binary flag at optimal threshold |
| `High_Risk_Customer` | True if probability ≥ 0.50 |
| `Expected_Revenue_Saved` | Predicted churners × $500 |
| `Retention_Cost` | Predicted churners × $50 |
| `Net_Value` | Revenue saved − outreach cost |

---

## Power BI Dashboard

The dashboard has four pages connected by a navigation bar. All pages share slicers for Contract, Tenure Group, Internet Service, High Risk Customer, and Senior Citizen.

---

### Page 1 - Executive Overview
<img src="assets/Dashboard%20executive%20overview.png" width="700"/>
<img src="assets/Dashboard%20executive%20overview-Filter.png" width="700"/>

The landing page. Six KPI cards give an at-a-glance health check of the customer base under the current filter selection: **Total Customers**, **Actual Churn %**, **Predicted Churn Rate %**, **Potential Revenue Saved**, **Net Value**, and **High Risk Customers**. Designed for leadership - one number per question, no charts needed.

---

### Page 2 - Churn Analysis
<img src="assets/Dashboard%20churn%20analysis.png" width="700"/>
<img src="assets/Dashboard%20churn%20analysis-Filter.png" width="700"/>

Breaks down predicted churn across four dimensions in a 2×2 grid. The donut chart shows fiber optic customers account for **47%** of predicted churn despite not being the majority segment. The contract bar chart confirms month-to-month holders churn at roughly **2×** the rate of one-year and **4×** the rate of two-year customers. The tenure chart shows churn drops sharply after year two. The payment method chart flags electronic check users as the highest-risk group.

---

### Page 3 - Risk Insights
<img src="assets/Dashboard%20risk%20insights.png" width="700"/>
<img src="assets/Dashboard%20risk%20insights-Filter.png" width="700"/>

Combines the model's ML output with business context. The summary table breaks tenure, monthly charges, churn probability, and net value by contract type - month-to-month customers represent **$186K** in monthly charges and **$1.14M** in net value at risk. The scatter plot maps tenure against monthly charge groups, with churners (blue) clustered in the high-charge, low-tenure quadrant. The risk band bar chart on the right segments the full customer base into Low, Medium, and High Risk - roughly **1,600 customers** fall in the high-risk band.

---

### Page 4 - Business Impact
<img src="assets/Dashboard%20business%20impact.png" width="700"/>
<img src="assets/Dashboard%20business%20impact-Filter.png" width="700"/>

Translates predictions into financial decisions. The bar chart compares revenue at risk between non-high-risk and high-risk customers, making the case for targeted spend. The scatter plot on the right plots churn probability against monthly charges, colored by actual churn - showing that high-probability churners (dark blue) concentrate in the $80–$120 monthly charge range. The left-side slicers (Contract, Tenure, Internet Service, High Risk, Senior Citizen) let stakeholders drill into any segment and see the financial impact update in real time.

---

## Key Findings

1. **Contract type is the strongest churn signal** - month-to-month customers churn at dramatically higher rates than annual or two-year contract holders.
2. **Newer customers are at the highest risk** - churn probability drops significantly with tenure.
3. **Monthly charges matter** - higher bills correlate with higher churn, especially in the first year.
4. **Fiber optic users churn more** than DSL or no-internet customers, suggesting a service quality or pricing perception issue.
5. **Customers without tech support** are meaningfully more likely to leave - it acts as a retention anchor.

---

## Recommendations

- **Prioritize** month-to-month customers with high monthly charges for proactive retention outreach.
- **Use the optimized threshold** (not 0.5) in production - it was tuned to maximize net revenue, not just accuracy.
- **Engage new customers early**, before negative signals compound.
- **Consider bundling tech support** as a retention incentive, particularly for fiber optic users.
- **Retrain every 1–2 months** as customer behavior and product mix evolve.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data wrangling | `pandas`, `numpy` |
| Visualization | `plotly`, `seaborn`, `matplotlib` |
| ML & pipelines | `scikit-learn`, `xgboost`, `lightgbm` |
| Explainability | `shap` |
| Hyperparameter search | `RandomizedSearchCV` |
| DB export | `sqlalchemy`, `psycopg2` (PostgreSQL) |
| Model persistence | `joblib` |
| Dashboard | Power BI |

---

## Getting Started

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost lightgbm shap sqlalchemy psycopg2-binary joblib tqdm
```

```bash
jupyter notebook Customer_Churn_Analysis.ipynb
```

Before running the export cell, update the database credentials:

```python
username = "your_username"
password = "your_password"
host     = "localhost"
port     = "5432"
database = "customer_churn_dataset"
```

The trained model is saved automatically as `final_xgboost_churn_model.pkl`.

---

## Dataset Source

IBM Sample Data - publicly available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
