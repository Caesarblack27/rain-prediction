import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
import requests
import tempfile
import os

# Function to load the pretrained model
@st.cache(allow_output_mutation=True)
def load_pretrained_model():
    try:
        # Load model from GitHub
        model_url = "https://github.com/Caesarblack27/rain-prediction/raw/main/rain_prediction_model.h5"
        response = requests.get(model_url)
        response.raise_for_status()
        
        # Save model to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_model:
            temp_model.write(response.content)
            temp_model.close()
            model = load_model(temp_model.name)
        
        # Delete temporary file after loading model
        os.remove(temp_model.name)
        
        return model
    except Exception as e:
        st.error(f"Unable to load model: {str(e)}")
        return None

# Load data from URL
url = "https://raw.githubusercontent.com/Caesarblack27/rain-prediction/main/weatherAUS.csv"
data = pd.read_csv(url)

# Fill missing values
data.fillna(data.mode().iloc[0], inplace=True)

# Encode categorical variables
label_encoder_location = LabelEncoder()
data['Location'] = label_encoder_location.fit_transform(data['Location'])

label_encoder_wind_gust_dir = LabelEncoder()
data['WindGustDir'] = label_encoder_wind_gust_dir.fit_transform(data['WindGustDir'])

label_encoder_wind_dir_9am = LabelEncoder()
data['WindDir9am'] = label_encoder_wind_dir_9am.fit_transform(data['WindDir9am'])

label_encoder_wind_dir_3pm = LabelEncoder()
data['WindDir3pm'] = label_encoder_wind_dir_3pm.fit_transform(data['WindDir3pm'])

label_encoder_rain_today = LabelEncoder()
data['RainToday'] = label_encoder_rain_today.fit_transform(data['RainToday'])

# Define all features needed for prediction
all_features = [
    'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am',
    'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
    'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
]

# Main Streamlit app
def main():
    st.title('Rain Prediction App')

    # Load the pretrained model
    model = load_pretrained_model()

    if model is None:
        st.error("Failed to load the model. Please check the logs for details.")
        return

    # Define scaler here, after model is loaded
    scaler = StandardScaler()

    # User inputs
    st.subheader('Enter the weather details:')
    
    # Initialize user input variables
    user_inputs = {}
    
    # Collect user inputs for each feature
    for feature in all_features:
        if feature == 'Location':
            user_inputs[feature] = st.selectbox(feature, label_encoder_location.classes_)
        elif feature == 'WindGustDir':
            user_inputs[feature] = st.selectbox(feature, label_encoder_wind_gust_dir.classes_)
        else:
            user_inputs[feature] = st.number_input(feature, value=float(data[feature].mode()[0]))

    if st.button('Predict'):
        try:
            # Create DataFrame from user inputs
            user_data = {
                'Location': label_encoder_location.transform([user_inputs['Location']])[0],
                'MinTemp': user_inputs['MinTemp'],
                'MaxTemp': user_inputs['MaxTemp'],
                'Rainfall': user_inputs['Rainfall'],
                'Evaporation': user_inputs['Evaporation'],
                'Sunshine': user_inputs['Sunshine'],
                'WindGustDir': label_encoder_wind_gust_dir.transform([user_inputs['WindGustDir']])[0],
                'WindGustSpeed': user_inputs['WindGustSpeed'],
                'WindDir9am': label_encoder_wind_dir_9am.transform([user_inputs['WindDir9am']])[0],
                'WindDir3pm': label_encoder_wind_dir_3pm.transform([user_inputs['WindDir3pm']])[0],
                'WindSpeed9am': user_inputs['WindSpeed9am'],
                'WindSpeed3pm': user_inputs['WindSpeed3pm'],
                'Humidity9am': user_inputs['Humidity9am'],
                'Humidity3pm': user_inputs['Humidity3pm'],
                'Pressure9am': user_inputs['Pressure9am'],
                'Pressure3pm': user_inputs['Pressure3pm'],
                'Cloud9am': user_inputs['Cloud9am'],
                'Cloud3pm': user_inputs['Cloud3pm'],
                'Temp9am': user_inputs['Temp9am'],
                'Temp3pm': user_inputs['Temp3pm']
            }

            user_data_df = pd.DataFrame(user_data, index=[0])

            # Scale user data
            user_data_scaled = scaler.transform(user_data_df)

            # Print the shape of user_data_scaled for debugging
            st.write(f"Shape of user_data_scaled: {user_data_scaled.shape}")

            # Predict
            prediction = model.predict(user_data_scaled)
            prediction_result = "Yes" if prediction[0][0] >= 0.5 else "No"
            st.write(f'Will it rain tomorrow? {prediction_result}')

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

if __name__ == '__main__':
    main()
