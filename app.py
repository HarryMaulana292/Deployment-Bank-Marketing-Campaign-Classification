import streamlit as st
import pickle
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from category_encoders import BinaryEncoder, OrdinalEncoder

st.set_page_config(
    page_title = "Bank Marketing Campaign Classification",
    page_icon = ":bank:"
)

st.title("Deposit or not Deposit Classification")

if "model" not in st.session_state:
    model = pickle.load(open("model.sav", "rb"))
    st.session_state["model"] = model
    
age_input = st.number_input("Insert customer's age :person_in_tuxedo:", min_value = 0, step = 1)
with st.expander("Insert customer's job :briefcase:"):
    job_input = st.radio(label = "Choose one", options = ["admin.",
                                                        "housemaid",
                                                        "technician",
                                                        "management",
                                                        "student",
                                                        "services",
                                                        "blue-collar",
                                                        "entrepreneur",
                                                        "retired",
                                                        "unemployed",
                                                        "self-employed",
                                                        "other"])
balance_unput = st.number_input("Insert customer balance :heavy_dollar_sign:", min_value = 0)
housing_input = st.radio("Insert customer's housing credit :house:", ["yes", "no"])
loan_input = st.radio("Insert customer's loan credit :money_with_wings:", ["yes", "no"])
contact_input = st.radio("Insert customer's last contact :telephone_receiver:", ["cellular", "telephone", "other"])
with st.expander("Insert month of last contacted with customer :crescent_moon:"):
    month_input = st.radio(label = "Choose one", options = ["jan",
                                                        "feb",
                                                        "mar",
                                                        "apr",
                                                        "may",
                                                        "jun",
                                                        "july",
                                                        "aug",
                                                        "sep",
                                                        "oct",
                                                        "nov",
                                                        "dec"])
campaign_input = st.number_input("Insert how many contact performed with customer during this campaign :1234:",
                                 min_value = 0, step = 1)
pdays_input = st.number_input("Insert how many days after last contact with customer :calendar:",
                                 min_value = -1, step = 1)
poutcome_input = st.radio("Insert the last campaign result :money_mouth_face:", ["success", "failure",
                                                               "other", "unknown"])
if st.button('Model Predict'):
    data = np.array([age_input, job_input, balance_unput, housing_input, loan_input, 
                     contact_input, month_input, campaign_input, pdays_input, poutcome_input]).reshape(1, -1)
    columns = ['age', 'job', 'balance', 'housing', 'loan', 'contact', 'month', 'campaign', 'pdays', 'poutcome']
    data_df = pd.DataFrame(data = data, columns = columns)
    result = st.session_state["model"].predict(data_df)
    if result[0] == 1:
        st.write(f"Prediction Result : Deposit")
    else :
        st.write(f"Prediction Result : Not Deposit")
        
else:
    st.write("Please input the feature above to start modelling prediciton")
