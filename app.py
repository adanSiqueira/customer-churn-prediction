import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np

# Load trained model
@st.cache_resource
def load_model():
    with open("model/churn_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# SHAP explainer loader (optional)
@st.cache_resource
def load_explainer(_model):
    return shap.Explainer(_model)

# Set up page
st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("🔍 Churn Prediction App")

# Sidebar for input
st.sidebar.header("Client Information")
age = st.sidebar.slider("Age", 18, 100, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
number_of_orders = st.sidebar.slider("Number of Orders", 1, 4, 1)
total_spent = st.sidebar.number_input("Total Spent (R$)", min_value=0.0, value=100.0)
order_1_year = st.sidebar.slider("Year of 1st Purchase", 2018, 2025, 2021)
order_1_month = st.sidebar.slider("Month of 1st Purchase", 1, 12, 6)
order_1_day = st.sidebar.slider("Day of 1st Purchase", 1, 31, 15)

# Compute average ticket
ticket = total_spent / number_of_orders if number_of_orders > 0 else 0

# Map gender to int
gender_encoded = 1 if gender == "Male" else 0

data = {
    'number_of_orders': number_of_orders,
    'total_spent': total_spent,
    'average_ticket': ticket,
    'gender': gender_encoded,
    'age': age,
    'order_1_year': order_1_year,
    'order_1_month': order_1_month,
    'order_1_day_of_month': order_1_day
}
input_df = pd.DataFrame([data])

st.subheader("🔍 Input Summary")
st.write(input_df)

# Predict button
if st.sidebar.button("Predict Churn"):
    model = load_model()

    # Create input df
    data = pd.DataFrame({
        "number_of_orders": [number_of_orders],
        "total_spent": [total_spent],
        "average_ticket": [ticket],
        "gender": [gender_encoded],
        "age": [age],
        "order_1_year": [order_1_year],
        "order_1_month": [order_1_month],
        "order_1_day_of_month": [order_1_day]
    })

    # Make prediction
    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.markdown("### ⚠️ This client is **likely to churn**.")
    else:
        st.markdown("### ✅ This client is **likely to stay**.")

    st.markdown(f"**Churn probability:** `{prob:.2%}`")

    # SHAP explanation
    with st.expander("SHAP Explanation"):
        explainer = load_explainer(model)
        shap_values = explainer(data)

        shap.plots.waterfall(shap_values[0], max_display=10)
        fig = plt.gcf()  # pega a figura gerada automaticamente pelo SHAP (matplotlib backend)
        st.pyplot(fig, bbox_inches='tight')


    # Feature importance (bar)
    with st.expander(" Global Feature Importance"):
        st.markdown("Based on XGBoost gain importance.")
        import xgboost as xgb
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        xgb.plot_importance(model, ax=ax, max_num_features=10, importance_type='gain')
        ax.set_title("Top 10 Important Features")

        st.pyplot(fig, bbox_inches='tight')