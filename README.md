# LendScope — Loan & Card Portfolio Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-189FDD?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LendScope** is an interactive, production-ready analytics dashboard for loan and card portfolio monitoring. It combines rich visual exploration with a built-in XGBoost default-prediction scorer, enabling credit risk analysts, portfolio managers, and data scientists to inspect performance, diagnose risk drivers, and simulate underwriting decisions — all from a single Streamlit interface.

**🔗 Live Demo:** [lendscope-dashboard.streamlit.app](https://lendscope-dashboard.streamlit.app/#lend-scope-loan-and-card-portfolio-analytics)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Data Schema](#data-schema)
- [Machine Learning Model](#machine-learning-model)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Locally](#running-locally)
  - [Deployment](#deployment)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

LendScope is a unified dashboard designed to address a common gap in lending analytics: the separation between **descriptive portfolio monitoring** and **predictive default modeling**. Most teams either explore historical data in one tool and score new applications in another, or rely on static reports that cannot drill down into risk drivers.

This project bridges that gap by pairing:

- A **multi-dimensional filtering engine** that cascades customer, loan, and card-transaction filters across all views.
- A **suite of Plotly visualizations** covering portfolio trends, risk concentration, repayment behavior, and fraud patterns.
- An **XGBoost classifier** trained directly on the loan book, with SHAP-style log-odds contributions surfaced for every scored application.
- **Business decision bands** (`APPROVE_BELOW = 0.10`, `DECLINE_ABOVE = 0.25`) that translate model outputs into actionable underwriting actions.

The dashboard is designed to be **data-source agnostic**: it reads from local CSVs for development and supports PostgreSQL via a single environment variable for production deployments.

---

## Key Features

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>📊 Portfolio Analytics</h3>
      <ul>
        <li><strong>Monthly Funded vs. Received</strong> — dual-axis trend with cumulative cash-flow tracking.</li>
        <li><strong>Loan Status Mix</strong> — donut chart segmenting Current, Fully Paid, and Charged Off.</li>
        <li><strong>Risk Band Distribution</strong> — stacked bars across FICO bands (Poor → Exceptional).</li>
        <li><strong>Geographic Heatmap</strong> — state-level exposure and default concentration.</li>
        <li><strong>Term & Purpose Breakdown</strong> — funded volume and bad-rate by loan purpose and term.</li>
      </ul>
    </td>
    <td width="50%" valign="top">
      <h3>🧠 Default Scoring Engine</h3>
      <ul>
        <li><strong>XGBoost classifier</strong> trained on 16 numerical and categorical features.</li>
        <li><strong>Interactive scorer</strong> — adjust application parameters and receive a real-time default probability.</li>
        <li><strong>SHAP log-odds contributions</strong> — top-K feature impacts displayed as a horizontal waterfall chart.</li>
        <li><strong>Decision bands</strong> — automatic Approve / Review / Decline classification.</li>
        <li><strong>Model diagnostics</strong> — ROC curve, Precision–Recall curve, confusion matrix, and gain-based feature importance.</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>💳 Card Transaction Monitoring</h3>
      <ul>
        <li><strong>Fraud rate by merchant category</strong> — identify high-risk transaction segments.</li>
        <li><strong>Spend velocity trends</strong> — monthly transaction volume and average ticket size.</li>
        <li><strong>Channel analysis</strong> — online vs. in-store vs. ATM fraud incidence.</li>
        <li><strong>Card type distribution</strong> — credit vs. debit vs. prepaid exposure.</li>
      </ul>
    </td>
    <td width="50%" valign="top">
      <h3>🔎 Exploration & UX</h3>
      <ul>
        <li><strong>Cascading sidebar filters</strong> — state, credit score range, home ownership, loan grade, status, purpose, card type, channel, and date range.</li>
        <li><strong>Live record counters</strong> — customer, loan, and transaction counts update as filters change.</li>
        <li><strong>Empty-state handling</strong> — graceful warnings when no data matches the current selection.</li>
        <li><strong>One-click reset</strong> — restore all filters to their default state.</li>
      </ul>
    </td>
  </tr>
</table>

---


**Key architectural decisions:**

| Decision | Rationale |
|---|---|
| **Cascading filters** | Customer-level filters (state, credit score, ownership) propagate to loans and transactions via `customer_id`, ensuring consistency across views. |
| **Plotly over native Streamlit charts** | Enables hover templates, unified hover mode, custom risk colour scales, and exportable SVG/PNG figures. |
| **XGBoost native `pred_contribs`** | Avoids a separate SHAP dependency; contributions are returned directly in log-odds space and rendered as a waterfall chart. |
| **CSV-first, Postgres-optional** | Zero-friction local development with sample data; production-ready database support via a single environment variable. |
| **`@st.cache_data` on loaders** | Prevents re-reading CSVs and re-parsing dates on every interaction, keeping the UI responsive. |

---

## Tech Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Frontend** | Streamlit | ≥ 1.36 | Web app framework, sidebar controls, layout, tabs |
| **Visualization** | Plotly | ≥ 5.20 | Interactive charts (scatter, donut, heatmap, waterfall) |
| **Data** | pandas | ≥ 2.1 | Data manipulation, filtering, aggregation |
| **Data** | NumPy | ≥ 1.26 | Numerical operations, array handling |
| **ML** | XGBoost | ≥ 2.0 | Gradient-boosted default classifier |
| **ML** | scikit-learn | ≥ 1.4 | Train/test split, ROC-AUC, PR-AUC, confusion matrix |
| **ML** | Joblib | ≥ 1.3 | Model metadata serialization |
| **Database** | SQLAlchemy | ≥ 2.0 | PostgreSQL engine and ORM-free `read_sql` |
| **Database** | psycopg2-binary | ≥ 2.9 | PostgreSQL driver |

---

## Project Structure  
```text
lendscope-dashboard/
│
├── .streamlit/ # Streamlit configuration (theme, server)
│ └── config.toml
│
├── data/ # Sample / demo datasets (CSV)
│ ├── customers.csv
│ ├── loans.csv
│ └── card_transactions.csv
│
├── models/ # Trained model artifacts (generated)
│ ├── lendscope_xgb.json
│ └── lendscope_xgb_meta.joblib
│
├── app.py # Main Streamlit application entry point
├── charts.py # Plotly figure builders for portfolio tabs
├── model.py # XGBoost training, scoring, decision bands
├── model_charts.py # Plotly figures for the scoring tab
├── data_loader.py # Data ingestion, caching, and filtering
├── make_sample_data.py # Script to generate synthetic demo CSVs
├── requirements.txt # Python dependencies
├── .gitignore
├── LICENSE # MIT License
└── README.md
```

---


---

## Data Schema

The dashboard expects three relational tables (CSV files or PostgreSQL tables). Column names are configurable in `data_loader.py`.

<details>
<summary><strong>customers.csv</strong></summary>

| Column | Type | Description |
|---|---|---|
| `customer_id` | string | Primary key, joins to loans and transactions |
| `signup_date` | date | Customer onboarding date |
| `credit_score` | integer | FICO score (300–850) |
| `address_state` | string | Two-letter US state code |
| `home_ownership` | string | `RENT`, `OWN`, `MORTGAGE`, `OTHER` |
| `annual_income` | float | Self-reported annual income |
| `employment_length` | float | Years of employment |

</details>

<details>
<summary><strong>loans.csv</strong></summary>

| Column | Type | Description |
|---|---|---|
| `loan_id` | string | Primary key |
| `customer_id` | string | Foreign key to customers |
| `issue_date` | date | Loan origination date |
| `loan_status` | string | `Current`, `Fully Paid`, `Charged Off`, etc. |
| `funded_amount` | float | Principal disbursed |
| `total_payment` | float | Total amount received to date |
| `int_rate` | float | Interest rate (%) |
| `term_months` | integer | Loan term in months |
| `dti` | float | Debt-to-income ratio |
| `installment` | float | Monthly instalment amount |
| `grade` | string | Loan grade (`A`–`G`) |
| `purpose` | string | Loan purpose (e.g., `debt_consolidation`) |
| `verification_status` | string | `Verified`, `Source Verified`, `Not Verified` |
| `application_type` | string | `Individual` or `Joint App` |
| `delinq_2yrs` | integer | Delinquencies in past 2 years |
| `open_accounts` | integer | Number of open credit lines |
| `revol_util` | float | Revolving line utilization (%) |

</details>

<details>
<summary><strong>card_transactions.csv</strong></summary>

| Column | Type | Description |
|---|---|---|
| `transaction_id` | string | Primary key |
| `customer_id` | string | Foreign key to customers |
| `transaction_date` | date | Transaction timestamp |
| `amount` | float | Transaction amount |
| `card_type` | string | `Credit`, `Debit`, `Prepaid` |
| `channel` | string | `Online`, `In-Store`, `ATM` |
| `merchant_category` | string | Merchant category code description |
| `is_fraud` | boolean | Ground-truth fraud label |
| `credit_score` | integer | Customer FICO score at transaction time |

</details>

---

## Machine Learning Model

### XGBoost Default Classifier

The model is trained on the `loans` table to predict `is_bad` (1 if `loan_status == "Charged Off"`, else 0).

**Feature set (16 features):**

| Type | Features |
|---|---|
| **Numerical** (11) | `int_rate`, `term_months`, `dti`, `annual_income`, `credit_score`, `employment_length`, `delinq_2yrs`, `open_accounts`, `revol_util`, `installment`, `funded_amount` |
| **Categorical** (6) | `grade`, `purpose`, `home_ownership`, `address_state`, `verification_status`, `application_type` |

> **Note:** The numerical count above includes 11 features; the categorical list contains 6, making 17 total. The model's `FEATURES` list concatenates both, and the dashboard exposes all of them in the interactive scorer.

**Training pipeline:**

1. Drop rows with missing target values.
2. Impute numerical features with training-set medians.
3. Align categorical levels to training codes (unseen categories → `"Unknown"`).
4. Stratified 80/20 train-test split (`random_state=42`).
5. Compute `scale_pos_weight` to handle class imbalance.
6. Train XGBoost with early stopping on the validation set.

**Evaluation metrics displayed in the dashboard:**

- ROC-AUC curve
- Precision–Recall curve with baseline
- Confusion matrix at a configurable threshold
- Top-15 feature importance by gain share

**Decision bands:**

| Band | Probability of Default | Action |
|---|---|---|
| ✅ **Approve** | < 0.10 | Auto-approve the application |
| ⚠️ **Review** | 0.10 – 0.25 | Manual underwriting review |
| 🚫 **Decline** | > 0.25 | Auto-decline the application |

**SHAP-style explainability:** For every scored application, the dashboard renders a horizontal waterfall chart of the top 12 log-odds contributions. Positive values push the prediction toward default; negative values push toward repayment.

---

## Getting Started

### Prerequisites

- **Python** 3.10 or later
- **pip** or **conda**
- Optional: **PostgreSQL** 13+ (only if using a database instead of CSVs)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/RajayJain/lendscope-dashboard.git
cd lendscope-dashboard

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic demo data (skip if you have real CSVs)
python make_sample_data.py

# 5. Running Locally
streamlit run app.py
```

- The app will open at http://localhost:8501. Use the sidebar to filter the portfolio and navigate between the Analytics and Scoring tabs.

---

## Deployment
**Streamlit Community Cloud:**

- Push the repository to GitHub.

- Go to share.streamlit.io and click New app.

- Select the repository, branch (main), and main file (app.py).

**Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
**Bash:**
```bash
docker build -t lendscope .
docker run -p 8501:8501 lendscope
```
---

## Configuration

| Environment Variable | Default | Description | 
 | ----- | ----- | ----- | 
| `LENDSCOPE_DB_URL` | (unset) | PostgreSQL connection string. When set, the dashboard reads from the database instead of CSV files. Example: `postgresql://user:pass@host:5432/lendscope` | 

Streamlit secrets (`.streamlit/secrets.toml`):

```
LENDSCOPE_DB_URL = "postgresql://user:pass@host:5432/lendscope"

```

When `LENDSCOPE_DB_URL` is present, `data_loader.py` creates a SQLAlchemy engine (with `pool_pre_ping=True`) and issues `SELECT * FROM {table}` for each of the three tables.

## Usage Guide

### 1. Filter the Portfolio

Open the sidebar and select any combination of:

* **State** — multi-select from all customer states

* **Credit score** — slider from 300 to 850

* **Home ownership** — Rent, Own, Mortgage, Other

* **Loan grade** — A through G

* **Loan status** — Current, Fully Paid, Charged Off, etc.

* **Loan purpose** — debt consolidation, credit card, home improvement, etc.

* **Card type** — Credit, Debit, Prepaid

* **Channel** — Online, In-Store, ATM

* **Date range** — issue date / transaction date

*All filters cascade: selecting a state restricts loans and transactions to customers in that state.*

### 2. Explore the Analytics Tab

The main dashboard is organised into sections:

* **KPI strip** — total funded, total received, net cash flow, bad rate, average credit score

* **Trend charts** — monthly funded vs. received, monthly bad rate

* **Composition charts** — loan status donut, grade distribution, purpose breakdown

* **Risk charts** — FICO band stacked bars, state-level heatmap

* **Card charts** — fraud rate by merchant category, spend velocity, channel analysis

### 3. Score an Application

Switch to the **Scoring** tab:

* Adjust the 16 application parameters using the sliders and dropdowns (pre-populated with sensible defaults).

* The model returns a probability of default and a decision band (Approve / Review / Decline).

* Inspect the SHAP waterfall to see which features drove the prediction.

* Review model diagnostics (ROC, PR, confusion matrix, feature importance) in the collapsible section below.

## Contributing

Contributions are welcome. To maintain code quality and consistency, please follow these guidelines:

* Fork the repository and create a feature branch from main.

* Follow the existing module structure — new charts belong in `charts.py` or `model_charts.py`; new data logic belongs in `data_loader.py`.

* Use type hints on all public functions.

* Add docstrings to new functions and classes (Google style preferred).

* Test locally with `streamlit run app.py` before opening a pull request.

* Keep PRs focused — one feature or fix per pull request.

* Update this README if your change affects configuration, data schema, or user-facing behaviour.

**Reporting issues:** Please use the GitHub issue tracker and include:

* Steps to reproduce

* Expected vs. actual behaviour

* Screenshots (if applicable)

* Environment details (Python version, OS, browser)

## License

This project is licensed under the MIT License. See the LICENSE file for the full text.

```
MIT License

Copyright (c) 2026 Rajay Jain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

## Acknowledgements

* Built with Streamlit and Plotly.

* Gradient-boosted modeling via XGBoost.

* Evaluation utilities from scikit-learn.

* Inspired by the need for transparent, explainable credit-risk tooling in consumer lending.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" alt="divider"/>

### ⭐ If this project helped you, drop a star — it means a lot!

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=16&duration=2500&pause=800&color=27AE60&center=true&vCenter=true&width=600&lines=LendScope+%E2%80%94+Bank+%26+Card+Analytics+Dashboard;Built+with+%F0%9F%92%99+%E2%80%94+Streamlit+and+Plotly;Gradient+Boosted+Modelling+Via+XGBoost;Evaluation+Utilities+From+Scikit+Learn;Happy+Dashboarding!+%F0%9F%93%8A" alt="Footer"/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&text=Thanks%20for%20visiting!&fontSize=28&fontColor=ffffff" />


<sub>🏦 <b>LendScope</b> — Bank Loan & Card Portfolio Analytics Dashboard · FY2026 · Made with ❤️ and a lot of <code>GROUP BY</code></sub>

</div>
