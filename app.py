# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config at the top
st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")

# Load model and preprocessors
model = joblib.load("best_crop_yield_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Get possible values for dropdowns
state_encoder = label_encoders['State']
crop_encoder = label_encoders['Crop']
all_states = list(state_encoder.classes_)
all_crops = list(crop_encoder.classes_)
seasons = ['Whole Year', 'Rabi', 'Kharif', 'Zaid']

# Season mapping
season_map = {'Whole Year': 1, 'Rabi': 2, 'Kharif': 3, 'Zaid': 4}

st.markdown("# 🌾 Crop Yield Prediction App")
st.markdown("This app predicts crop yield (tonnes/hectare) based on environmental, agricultural and location-based features.")

st.sidebar.markdown("## 📝 Input Parameters")

# Input Fields
state = st.sidebar.selectbox("Select State", all_states)
crop = st.sidebar.selectbox("Select Crop", all_crops)
season = st.sidebar.selectbox("Select Season", seasons)
crop_year = st.sidebar.number_input("Crop Year", min_value=2000, max_value=2025, value=2023)
area = st.sidebar.number_input("Cultivated Area (hectares)", min_value=0.1, value=1.0)
production = st.sidebar.number_input("Total Production (tonnes)", min_value=0.0, value=2.0)
rainfall = st.sidebar.number_input("Annual Rainfall (mm)", min_value=0.0, value=100.0)
fertilizer = st.sidebar.number_input("Fertilizer Used (kg/hectare)", min_value=0.0, value=50.0)
pesticide = st.sidebar.number_input("Pesticide Used (kg/hectare)", min_value=0.0, value=20.0)

# Encode input data
input_data = {
    'Crop': crop_encoder.transform([crop])[0],
    'Crop_Year': crop_year,
    'Season': season_map[season],
    'State': state_encoder.transform([state])[0],
    'Area': area,
    'Production': production,
    'Annual_Rainfall': rainfall,
    'Fertilizer': fertilizer,
    'Pesticide': pesticide
}

df_input = pd.DataFrame([input_data])
df_scaled = scaler.transform(df_input)

# Predict
if st.button("Predict Yield"):
    prediction = model.predict(df_scaled)[0]
    st.success(f"🌱 Predicted Crop Yield: **{prediction:.2f} tonnes/hectare**")

    # Add download button
    download_df = pd.DataFrame({
        'State': [state],
        'Crop': [crop],
        'Season': [season],
        'Crop Year': [crop_year],
        'Predicted Yield (tonnes/hectare)': [prediction]
    })
    
    csv = download_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prediction",
        data=csv,
        file_name='crop_yield_prediction.csv',
        mime='text/csv'
    )
