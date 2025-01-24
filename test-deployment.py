import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the trained model
model = joblib.load('model.pkl')

# App title
st.title("Diamond Price Predictor")

# Hardcoded values for dropdown options
shape_options = [
    'Cushion Modified', 'Oval', 'Pear', 'Cushion', 'Emerald', 'Heart',
    'Marquise', 'Princess', 'Radiant', 'Round', 'Square Radiant'
]
cut_options = ['Astor', 'Excellent', 'Ideal', 'Uncut', 'Very Good']
color_options = ['D', 'E', 'F', 'G', 'H']
clarity_options = ['FL', 'IF', 'VS1', 'VS2', 'VVS1', 'VVS2']
polish_options = ['Excellent', 'Good', 'Very Good']
symmetry_options = ['Excellent', 'Good', 'Very Good']
girdle_options = [
    'Medium', 'Thick', 'Thin', 'Extremely Thick', 'Extremely Thin to Extremely Thick',
    'Extremely Thin to Medium', 'Extremely Thin to Slightly Thick', 'Medium to Thick',
    'Slightly Thick', 'Slightly Thick to Thick', 'Very Thick'
]
culet_options = ['None', 'Small', 'Medium', 'Large', 'Very Large', 'Very Small']
type_options = ['Lab-Grown', 'Natural']
fluorescence_options = ['None', 'Faint', 'Medium', 'Strong']

# Sidebar dropdowns with hardcoded options
shape = st.sidebar.selectbox("Shape", shape_options)
cut = st.sidebar.selectbox("Cut", cut_options)
color = st.sidebar.selectbox("Color", color_options)
clarity = st.sidebar.selectbox("Clarity", clarity_options)
polish = st.sidebar.selectbox("Polish", polish_options)
symmetry = st.sidebar.selectbox("Symmetry", symmetry_options)
girdle = st.sidebar.selectbox("Girdle", girdle_options)
culet = st.sidebar.selectbox("Culet", culet_options)
diamond_type = st.sidebar.selectbox("Type", type_options)
fluorescence = st.sidebar.selectbox("Fluorescence", fluorescence_options)

# Numeric inputs
carat_weight = st.sidebar.number_input("Carat Weight (ct)", min_value=0.2, max_value=5.0, step=0.01)
length_width_ratio = st.sidebar.number_input("Length/Width Ratio", min_value=0.5, max_value=3.0, step=0.01)
depth_pct = st.sidebar.number_input("Depth %", min_value=50.0, max_value=80.0, step=0.1)
table_pct = st.sidebar.number_input("Table %", min_value=50.0, max_value=80.0, step=0.1)
length = st.sidebar.number_input("Length (mm)", min_value=3.0, max_value=15.0, step=0.1)
width = st.sidebar.number_input("Width (mm)", min_value=3.0, max_value=15.0, step=0.1)
height = st.sidebar.number_input("Height (mm)", min_value=2.0, max_value=15.0, step=0.1)

# Collect the inputs into a DataFrame
input_data = {
    'Carat Weight': carat_weight,
    'Length/Width Ratio': length_width_ratio,
    'Depth %': depth_pct,
    'Table %': table_pct,
    'Length': length,
    'Width': width,
    'Height': height,
    'Shape_Cushion': 1 if shape == "Cushion" else 0,
    'Shape_Cushion Modified': 1 if shape == "Cushion Modified" else 0,
    'Shape_Emerald': 1 if shape == "Emerald" else 0,
    'Shape_Heart': 1 if shape == "Heart" else 0,
    'Shape_Marquise': 1 if shape == "Marquise" else 0,
    'Shape_Oval': 1 if shape == "Oval" else 0,
    'Shape_Pear': 1 if shape == "Pear" else 0,
    'Shape_Princess': 1 if shape == "Princess" else 0,
    'Shape_Radiant': 1 if shape == "Radiant" else 0,
    'Shape_Round': 1 if shape == "Round" else 0,
    'Shape_Square Radiant': 1 if shape == "Square Radiant" else 0,
    'Cut_Astor': 1 if cut == "Astor" else 0,
    'Cut_Excellent': 1 if cut == "Excellent" else 0,
    'Cut_Ideal': 1 if cut == "Ideal" else 0,
    'Cut_Uncut': 1 if cut == "Uncut" else 0,
    'Cut_Very Good': 1 if cut == "Very Good" else 0,
    'Color_D': 1 if color == "D" else 0,
    'Color_E': 1 if color == "E" else 0,
    'Color_F': 1 if color == "F" else 0,
    'Color_G': 1 if color == "G" else 0,
    'Color_H': 1 if color == "H" else 0,
    'Clarity_FL': 1 if clarity == "FL" else 0,
    'Clarity_IF': 1 if clarity == "IF" else 0,
    'Clarity_VS1': 1 if clarity == "VS1" else 0,
    'Clarity_VS2': 1 if clarity == "VS2" else 0,
    'Clarity_VVS1': 1 if clarity == "VVS1" else 0,
    'Clarity_VVS2': 1 if clarity == "VVS2" else 0,
    'Polish_Excellent': 1 if polish == "Excellent" else 0,
    'Polish_Good': 1 if polish == "Good" else 0,
    'Polish_Very Good': 1 if polish == "Very Good" else 0,
    'Symmetry_Excellent': 1 if symmetry == "Excellent" else 0,
    'Symmetry_Good': 1 if symmetry == "Good" else 0,
    'Symmetry_Very Good': 1 if symmetry == "Very Good" else 0,
    'Girdle_Extremely Thick': 1 if girdle == "Extremely Thick" else 0,
    'Girdle_Extremely Thin to Extremely Thick': 1 if girdle == "Extremely Thin to Extremely Thick" else 0,
    'Girdle_Extremely Thin to Medium': 1 if girdle == "Extremely Thin to Medium" else 0,
    'Girdle_Extremely Thin to Slightly Thick': 1 if girdle == "Extremely Thin to Slightly Thick" else 0,
    'Girdle_Medium': 1 if girdle == "Medium" else 0,
    'Girdle_Medium to Extremely Thick': 1 if girdle == "Medium to Extremely Thick" else 0,
    'Girdle_Medium to Slightly Thick': 1 if girdle == "Medium to Slightly Thick" else 0,
    'Girdle_Medium to Thick': 1 if girdle == "Medium to Thick" else 0,
    'Girdle_Medium to Very Thick': 1 if girdle == "Medium to Very Thick" else 0,
    'Girdle_Slightly Thick': 1 if girdle == "Slightly Thick" else 0,
    'Girdle_Slightly Thick to Extremely Thick': 1 if girdle == "Slightly Thick to Extremely Thick" else 0,
    'Girdle_Slightly Thick to Slightly Thick': 1 if girdle == "Slightly Thick to Slightly Thick" else 0,
    'Girdle_Slightly Thick to Thick': 1 if girdle == "Slightly Thick to Thick" else 0,
    'Girdle_Slightly Thick to Very Thick': 1 if girdle == "Slightly Thick to Very Thick" else 0,
    'Girdle_Thick': 1 if girdle == "Thick" else 0,
    'Girdle_Thick to Extremely Thick': 1 if girdle == "Thick to Extremely Thick" else 0,
    'Girdle_Thick to Very Thick': 1 if girdle == "Thick to Very Thick" else 0,
    'Girdle_Thin': 1 if girdle == "Thin" else 0,
    'Girdle_Thin to Extremely Thick': 1 if girdle == "Thin to Extremely Thick" else 0,
    'Girdle_Thin to Medium': 1 if girdle == "Thin to Medium" else 0,
    'Girdle_Thin to Slightly Thick': 1 if girdle == "Thin to Slightly Thick" else 0,
    'Girdle_Thin to Thick': 1 if girdle == "Thin to Thick" else 0,
    'Girdle_Thin to Very Thick': 1 if girdle == "Thin to Very Thick" else 0,
    'Girdle_Very Thick': 1 if girdle == "Very Thick" else 0,
    'Girdle_Very Thick to Extremely Thick': 1 if girdle == "Very Thick to Extremely Thick" else 0,
    'Girdle_Very Thin to Extremely Thick': 1 if girdle == "Very Thin to Extremely Thick" else 0,
    'Girdle_Very Thin to Slightly Thick': 1 if girdle == "Very Thin to Slightly Thick" else 0,
    'Girdle_Very Thin to Thick': 1 if girdle == "Very Thin to Thick" else 0,
    'Girdle_Very Thin to Very Thick': 1 if girdle == "Very Thin to Very Thick" else 0,
    'Culet_Medium': 1 if culet == "Medium" else 0,
    'Culet_Pointed': 1 if culet == "Pointed" else 0,
    'Culet_Small': 1 if culet == "Small" else 0,
    'Culet_Very Large': 1 if culet == "Very Large" else 0,
    'Culet_Very Small': 1 if culet == "Very Small" else 0,
    'Type_GIA': 1 if diamond_type == "GIA" else 0,
    'Type_GIA Lab-Grown': 1 if diamond_type == "GIA Lab-Grown" else 0,
    'Type_IGI Lab-Grown': 1 if diamond_type == "IGI Lab-Grown" else 0,
    'Fluorescence_Faint': 1 if fluorescence == "Faint" else 0,
    'Fluorescence_Medium': 1 if fluorescence == "Medium" else 0,
    'Fluorescence_Strong': 1 if fluorescence == "Strong" else 0,
    'Fluorescence_Unknown': 1 if fluorescence == "Unknown" else 0,
}
input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.session_state.prediction = prediction[0]  # Save prediction in session state

# Display Prediction
if 'prediction' in st.session_state:
    st.write(f"### Predicted Diamond Price: ${st.session_state.prediction:,.2f}")

# Feature Importance Plot
if st.button("Show Feature Importance"):
    feature_importance = model.feature_importances_
    feature_names = input_df.columns
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_df, x='Importance', y='Feature', ax=ax)
    st.pyplot(fig)

# Interactive Scatter Plot
if 'prediction' in st.session_state:
    st.subheader("Interactive Scatter Plot")
    selected_feature = st.selectbox("Select Feature", input_df.columns)
    fig = px.scatter(input_df, x=selected_feature, y=[st.session_state.prediction], labels={'y': 'Predicted Price'})
    st.plotly_chart(fig)
