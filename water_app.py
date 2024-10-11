import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('best_SVC_model.pkl')
scaler = joblib.load('scaler.pkl')

# WHO limits for water quality parameters
who_limits = {
    'ph': (6.5, 8.5),         # pH
    'hardness': (0, 200),     # Max limit
    'solids': (0, 1000),      # Max limit
    'chloramines': (0, 4),    # Max limit
    'sulfate': (0, 1000),     # Max limit
    'conductivity': (0, 400),  # Max limit
    'organic_carbon': (0, 10),  # Max limit
    'trihalomethanes': (0, 80),  # Max limit
    'turbidity': (0, 5)       # Max limit
}

# Streamlit app title
st.title("Water Quality Prediction Model")

# User input for each parameter
ph = st.slider('pH', 0.0, 14.0, 7.0)  # pH
hardness = st.slider('Hardness', 0.0, 300.0, 150.0)  # Hardness
solids = st.slider('Solids', 0.0, 50000.0, 15000.0)  # Solids
chloramines = st.slider('Chloramines', 0.0, 10.0, 3.0)  # Chloramines
sulfate = st.slider('Sulfate', 0.0, 500.0, 250.0)  # Sulfate
conductivity = st.slider('Conductivity', 0.0, 600.0, 300.0)  # Conductivity
organic_carbon = st.slider('Organic Carbon', 0.0, 30.0, 8.0)  # Organic Carbon
trihalomethanes = st.slider('Trihalomethanes', 0.0, 150.0, 60.0)  # Trihalomethanes
turbidity = st.slider('Turbidity', 0.0, 10.0, 3.0)  # Turbidity

# Prediction button
if st.button("Predict"):
    # Prepare the input data for the model
    input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate,
                                 conductivity, organic_carbon, trihalomethanes, turbidity]],
                               columns=['ph', 'hardness', 'solids', 'chloramines',
                                        'sulfate', 'conductivity', 'organic_carbon',
                                        'trihalomethanes', 'turbidity'])  # Lowercase names
    
    # Check if inputs are within WHO limits
    is_within_limits = True
    for param, limits in who_limits.items():
        if not (limits[0] <= input_data[param].values[0] <= limits[1]):
            is_within_limits = False
            break
    
    # Output based on WHO limits
    if is_within_limits:
        result = "Potable"
        summary = "The water is pure and safe for human consumption."
    else:
        result = "Not Potable"
        summary = "The water is not safe for consumption due to unsafe parameters."
    
    # Display the result
    st.write(f"Prediction: {result}")
    st.write(summary)