# 📉 Telco Customer Churn Prediction
> **A Machine Learning system to identify at-risk customers and prevent revenue loss.**

### 📖 Project Overview
Customer Churn (customers cancelling their subscription) is a major revenue killer in the Telecom industry. This project builds a machine learning solution to:
1.  **Predict** if a customer will leave next month.
2.  **Explain** the root causes (e.g., pricing, contract type).
3.  **Deploy** a dashboard for Marketing teams to take proactive action.

---

### 🧠 The Business Problem
The Marketing Director needed to know **who** was leaving and **why**.
* **Goal:** Maximize **Recall** (catch as many churners as possible).
* **Constraint:** Do not spam too many safe customers (maintain acceptable Precision).
* **Outcome:** Improved Churn detection rate from **45% (baseline)** to **73%** by optimizing the decision threshold.

---

### 📊 Key Results
| Metric | Baseline (Threshold 0.5) | Optimized (Threshold 0.3) | Business Impact |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 79% | N/A | Overall correctness. |
| **Recall (Churners Caught)** | **45%** | **73%** | We now identify **+106** more at-risk customers per batch. |
| **Precision** | High | Moderate | Accepted trade-off to minimize missed churners. |

**Top 3 Predictors of Churn:**
1.  **Total Charges:** (High lifetime value customers behave differently).
2.  **Tenure:** New customers are the most volatile.
3.  **Monthly Charges:** Higher bills correlate strongly with churn.

---

### 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit (Web App)
* **Model Persistence:** Joblib

---

## Setup Instructions
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. **Download Data:**
   - Download the Telco Customer Churn dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
   - Create a folder named `data/`.
   - Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` inside it.
4. Run the app: `streamlit run app.py`

---

### 📂 Project Structure
```text
├── data/                   # Dataset (Uploaded to GitHub for reference)
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/                 # Saved models & feature columns
│   ├── churn_rf_model.joblib
│   └── churn_columns.joblib
├── notebooks/              # Experimentation & Analysis
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── app.py                  # Streamlit Dashboard Entry Point
├── requirements.txt        # Python dependencies
├── .gitignore              # Files to ignore (data, venv)
└── README.md               # Project Documentation
