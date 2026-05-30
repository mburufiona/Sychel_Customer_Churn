import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json


st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="centered"
)


model = joblib.load("Customerchurn_model.joblib")


with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)



st.title("Customer Churn Predictor")
st.markdown("Identify customers at high risk of churning before they leave.")
st.markdown("---")


st.sidebar.header("Customer Details")
st.sidebar.markdown("Fill in the customer information below.")

# Dropdowns
intl_plan = st.sidebar.selectbox(
    "International Plan",
    ["No", "Yes"]
)


vm_plan = st.sidebar.selectbox(
    "Voice Mail Plan",
    ["No", "Yes"]
)

area_code = st.sidebar.selectbox(
    "Area Code",
    [408, 415, 510]
)

# Sliders
account_length = st.sidebar.slider(
    "Account Length (months)",
    min_value=1,
    max_value=250,
    value=100
)

num_vm = st.sidebar.slider(
    "Number of Voicemail Messages",
    min_value=0,
    max_value=50,
    value=0
)


day_mins = st.sidebar.slider(
    "Total Day Minutes",
    min_value=0.0,
    max_value=400.0,
    value=180.0
)

day_calls = st.sidebar.slider(
    "Total Day Calls",
    min_value=0,
    max_value=200,
    value=100
)

eve_mins = st.sidebar.slider(
    "Total Evening Minutes",
    min_value=0.0,
    max_value=400.0,
    value=200.0
)

eve_calls = st.sidebar.slider(
    "Total Evening Calls",
    min_value=0,
    max_value=200,
    value=100
)

night_mins = st.sidebar.slider(
    "Total Night Minutes",
    min_value=0.0,
    max_value=400.0,
    value=200.0
)

night_calls = st.sidebar.slider(
    "Total Night Calls",
    min_value=0,
    max_value=200,
    value=100
)

intl_mins = st.sidebar.slider(
    "Total International Minutes",
    min_value=0.0,
    max_value=20.0,
    value=10.0
)

intl_calls = st.sidebar.slider(
    "Total International Calls",
    min_value=0,
    max_value=20,
    value=4
)

svc_calls = st.sidebar.slider(
    "Customer Service Calls",
    min_value=0,
    max_value=10,
    value=1
)


avg_day_call   = day_mins   / max(day_calls, 1)
avg_eve_call   = eve_mins   / max(eve_calls, 1)
avg_night_call = night_mins / max(night_calls, 1)
avg_intl_call  = intl_mins  / max(intl_calls, 1)


total_calls    = day_calls + eve_calls + night_calls + intl_calls
# total calls across all periods

service_call_rate = svc_calls / (total_calls + 1)


log_day   = np.log1p(day_mins)
log_eve   = np.log1p(eve_mins)
log_night = np.log1p(night_mins)
log_intl  = np.log1p(intl_mins)

input_data = {
    "account length":        account_length,  
    "international plan":    1 if intl_plan == "Yes" else 0,  
    "voice mail plan":       1 if vm_plan == "Yes" else 0,    
    "number vmail messages": num_vm,
    "total day minutes":     day_mins,
    "total day calls":       day_calls,
    "total eve minutes":     eve_mins,
    "total eve calls":       eve_calls,
    "total night minutes":   night_mins,
    "total night calls":     night_calls,
    "total intl minutes":    intl_mins,
    "total intl calls":      intl_calls,
    "customer service calls": svc_calls,
    "avg_day_call":          avg_day_call,
    "avg_eve_call":          avg_eve_call,
    "avg_night_call":        avg_night_call,
    "avg_intl_call":         avg_intl_call,
    "total_calls":           total_calls,
    "service_call_rate":     service_call_rate,
    "log_total day minutes": log_day,
    "log_total eve minutes": log_eve,
    "log_total night minutes": log_night,
    "log_total intl minutes":  log_intl,
    "area code_415": 1 if area_code == 415 else 0,
    "area code_510": 1 if area_code == 510 else 0,
}

input_df = pd.DataFrame([input_data])

for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]


st.subheader("Churn Risk Assessment")

if st.button("Predict Churn Risk "):


    prediction  = model.predict(input_df)[0]


    probability = model.predict_proba(input_df)[0][1]


    
    if prediction == True:
        st.error(f" HIGH CHURN RISK — {round(probability * 100, 1)}% probability")
        st.markdown("**Recommended action:** Flag for immediate retention outreach")
    else:
        st.success(f" LOW CHURN RISK — {round(probability * 100, 1)}% probability")
        st.markdown("**Recommended action:** Continue standard engagement")

    # Probability bar
    st.markdown("**Churn probability:**")
    st.progress(float(probability))


    st.markdown("---")


    col1, col2, col3 = st.columns(3)
    # creates 3 equal columns side by side
    col1.metric("Service Calls",  svc_calls)
    col2.metric("Day Minutes",    round(day_mins, 1))
    col3.metric("Service Rate",   round(service_call_rate, 3))
 

st.markdown("---")
st.caption("Model: Tuned Decision Tree  |  Dataset: Sychel Telecom  |  Built with Streamlit")