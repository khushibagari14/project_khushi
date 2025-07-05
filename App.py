import numpy as np
import pandas as pd
import pickle as pkl
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the model with error handling
try:
    model = pkl.load(open('MIPML.pkl', 'rb'))

    # Check if the model is a valid predictive model
    if not hasattr(model, 'predict'):
        raise ValueError("The loaded model does not have a 'predict' method. Please check the model.")

except Exception as e:
    st.error(f"Error loading model: {e}")
    raise

# App Title
st.header('Medical Insurance Premium Predictor')

# Taking user inputs
gender = st.selectbox('Choose Gender', ['Female', 'Male'])
smoker = st.selectbox('Are you a smoker?', ['Yes', 'No'])
region = st.selectbox('Choose Region', ['SouthEast', 'SouthWest', 'NorthEast', 'NorthWest'])
age = st.slider('Enter Age', 5, 80)
bmi = st.slider('Enter BMI', 5, 100)
children = st.slider('Choose Number of Children', 0, 5)

# Predict button
if st.button('Predict'):
    # Encoding inputs manually
    gender_encoded = 0 if gender == 'Female' else 1
    smoker_encoded = 1 if smoker == 'Yes' else 0

    if region == 'SouthEast':
        region_encoded = 0
    elif region == 'SouthWest':
        region_encoded = 1
    elif region == 'NorthEast':
        region_encoded = 2
    else:
        region_encoded = 3  # NorthWest

    # Create input array
    input_data = (age, gender_encoded, bmi, children, smoker_encoded, region_encoded)
    input_data_array = np.asarray(input_data).reshape(1, -1)

    # Check if scaling was used during model training and apply the same transformation
    try:
        # Assuming StandardScaler was used during training (you can replace this with your model's actual scaler)
        scaler = StandardScaler()

        # Fit the scaler with a sample dataset or the dataset used during training (if known)
        # You need to either load the scaler or train it here.
        # Example: scaler.fit(training_data)  (Use the dataset used to train the model)
        
        # Transform input data
        input_data_scaled = scaler.fit_transform(input_data_array)
    except Exception as e:
        st.error(f"Error in scaling the input data: {e}")
        raise

    # Make prediction with error handling
    try:
        predicted_prem = model.predict(input_data_scaled)

        # Ensure the predicted premium is reasonable (e.g., positive values)
        if predicted_prem[0] < 0:
            st.error("Predicted premium is negative, which is not valid. Please check the model.")
        else:
            display_string = f'Insurance Premium will be {round(predicted_prem[0], 2)} rupees'
            st.markdown(display_string)
    except Exception as e:
        st.error(f"Error in prediction: {e}")
