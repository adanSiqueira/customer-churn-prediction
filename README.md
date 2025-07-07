# 🛍️ Customer Churn Prediction in eCommerce with Google Cloud
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-BigQuery-blue?logo=googlecloud&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-black?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualizations-orange?logo=matplotlib)
![Seaborn](https://img.shields.io/badge/Seaborn-EDA-blue?logo=seaborn)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-yellowgreen?logo=scikit-learn)

Predicting when a customer is likely to stop buying is one of the most critical insights for any subscription-based or transactional business. This project uses real-world eCommerce data to develop machine learning models capable of identifying churn risk — helping companies take action before losing valuable clients.

---

## 🚀 Project Overview

**Goal:**  
Build a machine learning pipeline to predict customer churn and uncover retention insights based on behavioral data from the [TheLook eCommerce dataset](https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce).

**Business Impact:**  
By identifying customers likely to churn, marketing teams can implement re-engagement strategies and loyalty campaigns, increasing **Customer Lifetime Value (CLV)** and **Revenue Retention**.

---

## 📦 Dataset: TheLook eCommerce (BigQuery Public Data)

The dataset consists of 7 structured tables related to users, orders, products, inventory, and transactions. Data was extracted using custom SQL queries, merged in Python (Pandas), and cleaned for modeling.

Key features include:
- Customer demographics (age, gender, location)
- Purchase behavior (order frequency, spend, recency)
- Product types and categories
- Delivery and return timestamps

---

## 🔧 Tools & Techniques

| Area               | Tools / Methods Used                                      |
|--------------------|-----------------------------------------------------------|
| Data Extraction    | SQL on BigQuery                                           |
| Data Wrangling     | Pandas, NumPy                                             |
| Churn Definition   | Recency logic & Kaplan-Meier survival modeling            |
| Exploratory Analysis | Seaborn, Matplotlib, descriptive statistics              |
| Survival Analysis  | Kaplan-Meier Estimator (lifelines)                       |
| Modeling (optional) | XGBoost                                                  |
| Evaluation         | ROC-AUC, Confusion Matrix, Precision-Recall              |

---

## 📊 Key Analyses & Findings

- **Kaplan-Meier survival curve** shows that ~50% of customers never return after their first purchase.
- Customers who make **a second purchase are far more likely to remain active** for extended periods (500+ days).
- A fixed churn threshold (e.g. 90 days) may **underestimate customer lifetime** for loyal buyers — suggesting the need for time-aware churn models.

---

## 📁 Folder Structure

```
├──
│ ├── 01_data preparation.ipynb
│ ├── 02_model_development.ipynb
│ └── 03_model_interpretation.ipynb
├── processed_data/
│ └── clients_info.csv
├── README.md
└── requirements.txt
```

---

## ✅ Skills Demonstrated

- SQL data extraction from public cloud datasets  
- End-to-end churn analysis using Python  
- Kaplan-Meier and survival modeling  
- Feature engineering for customer behavior  
- Business-driven data storytelling and interpretation

---

## Local Deployment with Streamlit

This project includes a web application built using [Streamlit](https://streamlit.io/), allowing you to interact with the churn prediction model directly from your browser.

### ▶️ How to Run Locally

1. **Install Dependencies**  
   Make sure you have Python installed (version 3.8 or higher). Then, install the required packages with:

   ```bash
   pip install -r requirements.txt

2. **Run the App**  
   In the root directory of the project, run:

   ```bash
   streamlit run app.py

3. **Access the app**  
  After executing the command above, Streamlit will automatically start a local server and display a URL in your terminal, such as:

   ```bash
   http://localhost:8501
   

### ⚙️ App Features (Technical Overview)

- 🧠 **Model Deployment**: Integrates a production-ready XGBoost classification model for churn prediction
- 🧾 **Manual Data Input**: Accepts user-defined inputs including `age`, `gender`, `number_of_orders`, and `total_spent`
- 🧮 **Dynamic Feature Engineering**: Automatically computes `average_ticket` as a derived feature (`total_spent / number_of_orders`)
- 📈 **Churn Inference**: Outputs binary churn prediction (`0` = active, `1` = churn) in real time
- 🧠 **Model Explainability**: Integrates **SHAP (SHapley Additive Explanations)** to generate global and local interpretability visualizations
- 📊 **Visual Insights**: Includes force plots and summary plots to showcase feature impact on predictions
- 🚀 **End-to-End Pipeline**: Demonstrates the full ML lifecycle — from data preprocessing to model inference and explainability — in a single interactive interface

