import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.write("Customer details fill karke Churn status predict karein.")

st.subheader("Enter Customer Details:")
gender = st.selectbox("Gender", ["M", "F"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
city = st.selectbox("City", ["KOLKATA", "HYDERABAD", "CHENNAI"])
country = st.selectbox("Country", ["India"])
purchase_amount = st.number_input("Purchase Amount", min_value=0.0, value=25000.0)
# 3. Model nd Scaler Load karna
@st.cache_resource
def load_files():
    model = tf.keras.models.load_model('model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, feature_columns = load_files()

# 4. Predict Button nd Processing
if st.button("Predict Churn", key="predict_btn"):
    input_data = pd.DataFrame([{
        'Gender': gender,
        'Age': age,
        'City': city,
        'Purchase_Amount': purchase_amount,
        'Country': country
    }])
    
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    
    input_scaled = scaler.transform(input_encoded)
    prediction_prob = model.predict(input_scaled)[0][0]
    
    st.markdown("---")
    if prediction_prob > 0.5:
        st.error(f"⚠️ **High Risk of Churn!** (Probability: {prediction_prob:.2%})")
        st.write("Customer leaves hone ke chances zyada hain.")
    else:
        st.success(f"✅ **Customer Will Stay!** (Churn Probability: {prediction_prob:.2%})")
        st.write("Customer retain rahega.")
