import streamlit as st
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(page_title="Loan Status Prediction", page_icon="💰")

# Add title
st.title("Loan Status Prediction")

# Load and preprocess data
@st.cache_data
def load_data():
    loan_dataset = pd.read_csv('loan_dataset.csv')
    loan_dataset = loan_dataset.dropna()
    
    # Label encoding
    loan_dataset.replace({"Loan_Status":{'N':0,'Y':1},
                         'Married':{'No':0,'Yes':1},
                         'Gender':{'Male':1,'Female':0},
                         'Self_Employed':{'No':0,'Yes':1},
                         'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},
                         'Education':{'Graduate':1,'Not Graduate':0}}, inplace=True)
    
    # Replace '3+' with 4 in Dependents
    loan_dataset = loan_dataset.replace(to_replace='3+', value=4)
    
    # Prepare features and target
    X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'], axis=1)
    y = loan_dataset['Loan_Status']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    
    return classifier, X.columns

# Load the model
model, features = load_data()

# Create input form
st.subheader("Enter Loan Application Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3", "4"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term (in months)", min_value=0)
    credit_history = st.selectbox("Credit History", [0, 1], help="0: No credit history, 1: Has credit history")
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# Create prediction button
if st.button("Predict Loan Status"):
    # Prepare input data
    input_data = {
        'Gender': 1 if gender == "Male" else 0,
        'Married': 1 if married == "Yes" else 0,
        'Dependents': int(dependents),
        'Education': 1 if education == "Graduate" else 0,
        'Self_Employed': 1 if self_employed == "Yes" else 0,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': 0 if property_area == "Rural" else (1 if property_area == "Semiurban" else 2)
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Show result
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("Loan is likely to be approved! 🎉")
    else:
        st.error("Loan is likely to be rejected.")
        
    # Show disclaimer
    st.info("Note: This is a prediction based on historical data and should not be considered as a final decision.")
