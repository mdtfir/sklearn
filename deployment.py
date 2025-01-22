import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('model.pkl')

# App title
st.title("Diamond Price Predictor")

# Input form for user to enter features
st.sidebar.header("Diamond Features")
shape = st.sidebar.selectbox("Shape", ['Cushion Modified', 'Oval', 'Pear', 'Other'])
cut = st.sidebar.selectbox("Cut", ['Ideal', 'Very Good', 'Good', 'Fair'])
color = st.sidebar.selectbox("Color", ['D', 'E', 'F', 'G', 'H'])
clarity = st.sidebar.selectbox("Clarity", ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2'])
carat_weight = st.sidebar.slider("Carat Weight", 0.2, 5.0, step=0.01)
length_width_ratio = st.sidebar.slider("Length/Width Ratio", 0.5, 3.0, step=0.01)
depth_pct = st.sidebar.slider("Depth %", 50.0, 80.0, step=0.1)
table_pct = st.sidebar.slider("Table %", 50.0, 80.0, step=0.1)
polish = st.sidebar.selectbox("Polish", ['Excellent', 'Very Good', 'Good'])
symmetry = st.sidebar.selectbox("Symmetry", ['Excellent', 'Very Good', 'Good'])
girdle = st.sidebar.selectbox("Girdle", ['Medium', 'Thick', 'Thin', 'Medium to Thick'])
culet = st.sidebar.selectbox("Culet", ['None', 'Small', 'Medium', 'Large'])
length = st.sidebar.slider("Length (mm)", 3.0, 15.0, step=0.1)
width = st.sidebar.slider("Width (mm)", 3.0, 15.0, step=0.1)
height = st.sidebar.slider("Height (mm)", 2.0, 15.0, step=0.1)
diamond_type = st.sidebar.selectbox("Type", ['Lab-Grown', 'Natural'])
fluorescence = st.sidebar.selectbox("Fluorescence", ['None', 'Faint', 'Medium', 'Strong'])

# Collect the inputs into a DataFrame
input_data = {
    'Shape': shape,
    'Cut': cut,
    'Color': color,
    'Clarity': clarity,
    'Carat Weight': carat_weight,
    'Length/Width Ratio': length_width_ratio,
    'Depth %': depth_pct,
    'Table %': table_pct,
    'Polish': polish,
    'Symmetry': symmetry,
    'Girdle': girdle,
    'Culet': culet,
    'Length': length,
    'Width': width,
    'Height': height,
    'Type': diamond_type,
    'Fluorescence': fluorescence
}

input_df = pd.DataFrame([input_data])

# Perform one-hot encoding to match model requirements
encoded_df = pd.get_dummies(input_df)

# Ensure that the encoded columns align with the model's training data
# Fill missing columns with 0
model_columns = joblib.load('columns_after_encoding.pkl')  # Columns used in training
for col in model_columns:
    if col not in encoded_df:
        encoded_df[col] = 0
encoded_df = encoded_df[model_columns]

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(encoded_df)
    st.write(f"### Predicted Diamond Price: ${prediction[0]:,.2f}")

