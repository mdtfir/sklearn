import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64

# Load the trained model
model = joblib.load('model.pkl')

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('jumpman.jpg')


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
type_options = ['Type_GIA','Type_GIA Lab-Grown','Type_IGI Lab-Grown']
fluorescence_options = ['None', 'Faint', 'Medium', 'Strong']

# Sidebar layout with columns
with st.sidebar:
    st.header("Diamond Input Features")

    # Create two columns for inputs
    col1, col2 = st.columns(2)

    # Categorical features in the first column
    with col1:
        st.subheader("Categorical Features")
        shape = st.selectbox("Shape", shape_options)
        cut = st.selectbox("Cut", cut_options)
        color = st.selectbox("Color", color_options)
        clarity = st.selectbox("Clarity", clarity_options)
        polish = st.selectbox("Polish", polish_options)
        symmetry = st.selectbox("Symmetry", symmetry_options)
        girdle = st.selectbox("Girdle", girdle_options)
        culet = st.selectbox("Culet", culet_options)
        diamond_type = st.selectbox("Type", type_options)
        fluorescence = st.selectbox("Fluorescence", fluorescence_options)

    # Numerical features in the second column
    with col2:
        st.subheader("Numerical Features")
        carat_weight = st.number_input("Carat Weight (ct)", min_value=0.2, max_value=5.0, step=0.01)
        length_width_ratio = st.number_input("Length/Width Ratio", min_value=0.5, max_value=3.0, step=0.01)
        depth_pct = st.number_input("Depth %", min_value=50.0, max_value=80.0, step=0.1)
        table_pct = st.number_input("Table %", min_value=50.0, max_value=80.0, step=0.1)
        length = st.number_input("Length (mm)", min_value=3.0, max_value=15.0, step=0.1)
        width = st.number_input("Width (mm)", min_value=3.0, max_value=15.0, step=0.1)
        height = st.number_input("Height (mm)", min_value=2.0, max_value=15.0, step=0.1)

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

col1, spacer2, col2,  = st.columns([4, 1, 4])

with col1:
    # Map one-hot encoded columns back to their original input features
    feature_mapping = {
    # Carat Weight and Dimensions
    'Carat Weight': 'Carat Weight',
    'Length/Width Ratio': 'Length/Width Ratio',
    'Depth %': 'Depth %',
    'Table %': 'Table %',
    'Length': 'Length',
    'Width': 'Width',
    'Height': 'Height',

    # Shape
    'Shape_Cushion': 'Shape',
    'Shape_Cushion Modified': 'Shape',
    'Shape_Emerald': 'Shape',
    'Shape_Heart': 'Shape',
    'Shape_Marquise': 'Shape',
    'Shape_Oval': 'Shape',
    'Shape_Pear': 'Shape',
    'Shape_Princess': 'Shape',
    'Shape_Radiant': 'Shape',
    'Shape_Round': 'Shape',
    'Shape_Square Radiant': 'Shape',

    # Cut
    'Cut_Astor': 'Cut',
    'Cut_Excellent': 'Cut',
    'Cut_Ideal': 'Cut',
    'Cut_Uncut': 'Cut',
    'Cut_Very Good': 'Cut',

    # Color
    'Color_D': 'Color',
    'Color_E': 'Color',
    'Color_F': 'Color',
    'Color_G': 'Color',
    'Color_H': 'Color',

    # Clarity
    'Clarity_FL': 'Clarity',
    'Clarity_IF': 'Clarity',
    'Clarity_VS1': 'Clarity',
    'Clarity_VS2': 'Clarity',
    'Clarity_VVS1': 'Clarity',
    'Clarity_VVS2': 'Clarity',

    # Polish
    'Polish_Excellent': 'Polish',
    'Polish_Good': 'Polish',
    'Polish_Very Good': 'Polish',

    # Symmetry
    'Symmetry_Excellent': 'Symmetry',
    'Symmetry_Good': 'Symmetry',
    'Symmetry_Very Good': 'Symmetry',

    # Girdle
    'Girdle_Extremely Thick': 'Girdle',
    'Girdle_Extremely Thin to Extremely Thick': 'Girdle',
    'Girdle_Extremely Thin to Medium': 'Girdle',
    'Girdle_Extremely Thin to Slightly Thick': 'Girdle',
    'Girdle_Medium': 'Girdle',
    'Girdle_Medium to Extremely Thick': 'Girdle',
    'Girdle_Medium to Slightly Thick': 'Girdle',
    'Girdle_Medium to Thick': 'Girdle',
    'Girdle_Medium to Very Thick': 'Girdle',
    'Girdle_Slightly Thick': 'Girdle',
    'Girdle_Slightly Thick to Extremely Thick': 'Girdle',
    'Girdle_Slightly Thick to Slightly Thick': 'Girdle',
    'Girdle_Slightly Thick to Thick': 'Girdle',
    'Girdle_Slightly Thick to Very Thick': 'Girdle',
    'Girdle_Thick': 'Girdle',
    'Girdle_Thick to Extremely Thick': 'Girdle',
    'Girdle_Thick to Very Thick': 'Girdle',
    'Girdle_Thin': 'Girdle',
    'Girdle_Thin to Extremely Thick': 'Girdle',
    'Girdle_Thin to Medium': 'Girdle',
    'Girdle_Thin to Slightly Thick': 'Girdle',
    'Girdle_Thin to Thick': 'Girdle',
    'Girdle_Thin to Very Thick': 'Girdle',
    'Girdle_Very Thick': 'Girdle',
    'Girdle_Very Thick to Extremely Thick': 'Girdle',
    'Girdle_Very Thin to Extremely Thick': 'Girdle',
    'Girdle_Very Thin to Slightly Thick': 'Girdle',
    'Girdle_Very Thin to Thick': 'Girdle',
    'Girdle_Very Thin to Very Thick': 'Girdle',

    # Culet
    'Culet_Medium': 'Culet',
    'Culet_Pointed': 'Culet',
    'Culet_Small': 'Culet',
    'Culet_Very Large': 'Culet',
    'Culet_Very Small': 'Culet',

    # Type
    'Type_GIA': 'Type',
    'Type_GIA Lab-Grown': 'Type',
    'Type_IGI Lab-Grown': 'Type',

    # Fluorescence
    'Fluorescence_Faint': 'Fluorescence',
    'Fluorescence_Medium': 'Fluorescence',
    'Fluorescence_Strong': 'Fluorescence',
    'Fluorescence_Unknown': 'Fluorescence',
    }

    # Aggregate feature importances by original input features
    feature_importance = model.feature_importances_
    feature_names = input_df.columns
    feature_df = pd.DataFrame({
        'Original Feature': [feature_mapping.get(name, name) for name in feature_names],
        'Importance': feature_importance
    }).groupby('Original Feature').sum().reset_index()

    # Sort by importance
    feature_df = feature_df.sort_values(by='Importance', ascending=False)

    # Plot the simplified feature importance
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_df, x='Importance', y='Original Feature', ax=ax)
    st.pyplot(fig)


with col2:

    # Define dummy data with default values
    dummy_data = {
        'Carat Weight': 0.7,
        'Length/Width Ratio': 1.0,
        'Depth %': 61.0,
        'Table %': 57.0,
        'Length': 5.5,
        'Width': 5.5,
        'Height': 3.5,
        'Shape_Cushion': 1,
        'Shape_Cushion Modified': 0,
        'Shape_Emerald': 0,
        'Shape_Heart':  0,
        'Shape_Marquise':  0,
        'Shape_Oval':0 ,
        'Shape_Pear': 0,
        'Shape_Princess': 0,
        'Shape_Radiant': 0,
        'Shape_Round': 0,
        'Shape_Square Radiant': 0,
        'Cut_Astor':  0,
        'Cut_Excellent': 1 ,
        'Cut_Ideal':  0,
        'Cut_Uncut':  0,
        'Cut_Very Good':  0,
        'Color_D': 1 ,
        'Color_E':  0,
        'Color_F':  0,
        'Color_G':  0,
        'Color_H':  0,
        'Clarity_FL':  0,
        'Clarity_IF':  0,
        'Clarity_VS1': 0,
        'Clarity_VS2':  0,
        'Clarity_VVS1':  0,
        'Clarity_VVS2': 1 ,
        'Polish_Excellent': 1,
        'Polish_Good': 0,
        'Polish_Very Good':  0,
        'Symmetry_Excellent': 1 ,
        'Symmetry_Good': 0,
        'Symmetry_Very Good': 0,
        'Girdle_Extremely Thick':  0,
        'Girdle_Extremely Thin to Extremely Thick':  0,
        'Girdle_Extremely Thin to Medium':  0,
        'Girdle_Extremely Thin to Slightly Thick':  0,
        'Girdle_Medium':  0,
        'Girdle_Medium to Extremely Thick': 1 ,
        'Girdle_Medium to Slightly Thick':  0,
        'Girdle_Medium to Thick':  0,
        'Girdle_Medium to Very Thick':  0,
        'Girdle_Slightly Thick':  0,
        'Girdle_Slightly Thick to Extremely Thick':  0,
        'Girdle_Slightly Thick to Slightly Thick':  0,
        'Girdle_Slightly Thick to Thick':  0,
        'Girdle_Slightly Thick to Very Thick':  0,
        'Girdle_Thick':  0,
        'Girdle_Thick to Extremely Thick':  0,
        'Girdle_Thick to Very Thick':  0,
        'Girdle_Thin':  0,
        'Girdle_Thin to Extremely Thick':  0,
        'Girdle_Thin to Medium':  0,
        'Girdle_Thin to Slightly Thick':  0,
        'Girdle_Thin to Thick':  0,
        'Girdle_Thin to Very Thick':  0,
        'Girdle_Very Thick':  0,
        'Girdle_Very Thick to Extremely Thick':  0,
        'Girdle_Very Thin to Extremely Thick':  0,
        'Girdle_Very Thin to Slightly Thick': 0,
        'Girdle_Very Thin to Thick':  0,
        'Girdle_Very Thin to Very Thick':  0,
        'Culet_Medium':  0,
        'Culet_Pointed': 1 ,
        'Culet_Small':  0,
        'Culet_Very Large': 0,
        'Culet_Very Small':  0,
        'Type_GIA': 0,
        'Type_GIA Lab-Grown':  0,
        'Type_IGI Lab-Grown': 1 ,
        'Fluorescence_Faint':  0,
        'Fluorescence_Medium':  0,
        'Fluorescence_Strong':  0,
        'Fluorescence_Unknown': 1 ,
    }

    # Convert dummy data to DataFrame
    dummy_df = pd.DataFrame([dummy_data])

    st.subheader("Feature Impact on Price")

    # Define relevant feature groups
    categorical_groups = ["Shape", "Cut", "Color", "Clarity"]
    numerical_columns = [
        'Carat Weight', 'Length/Width Ratio', 'Depth %', 'Table %',
        'Length', 'Width', 'Height'
    ]

    # Combine into a single list for dropdown
    feature_options = numerical_columns + categorical_groups

    # Select feature to analyze
    selected_feature = st.selectbox(
        "Select Feature to Analyze",
        feature_options
    )

    if selected_feature in categorical_groups:
        # Identify related fields in dummy data
        related_fields = [
            key for key in dummy_data.keys() if key.startswith(selected_feature + '_')
        ]
        x_axis = [field.replace(selected_feature + '_', '') for field in related_fields]
        predicted_prices = []

        for active_field in related_fields:
            temp_data = dummy_data.copy()
            # Set the active field to 1 and others in the group to 0
            for field in related_fields:
                temp_data[field] = 1 if field == active_field else 0
            # Predict price using the modified dummy data
            predicted_price = model.predict(pd.DataFrame([temp_data]))[0]  # Example prediction
            predicted_prices.append(predicted_price)

        # Plot results
        fig = px.bar(
            x=x_axis,
            y=predicted_prices,
            labels={"x": selected_feature, "y": "Predicted Price"},
            title=f"Effect of {selected_feature} on Price"
        )
        st.plotly_chart(fig)

    elif selected_feature in numerical_columns:
        intervals = [round(dummy_data[selected_feature] + i * 0.1, 2) for i in range(-5, 50)]
        predicted_prices = []

        for value in intervals:
            temp_data = dummy_data.copy()
            temp_data[selected_feature] = value
            predicted_price = model.predict(pd.DataFrame([temp_data]))[0]  # Example prediction
            predicted_prices.append(predicted_price)

        # Plot results
        fig = px.line(
            x=intervals,
            y=predicted_prices,
            labels={"x": selected_feature, "y": "Predicted Price"},
            title=f"Effect of {selected_feature} on Price"
        )
        st.plotly_chart(fig)

    else:
        st.error("Unsupported feature type for analysis.")

