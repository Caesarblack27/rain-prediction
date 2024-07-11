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

# Select all necessary features for prediction
selected_features = [
    'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am',
    'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
    'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday'
]

# Check for any new categorical values in WindDir9am and WindDir3pm
def check_and_update_label_encoder(label_encoder, column, mode_value):
    if mode_value not in label_encoder.classes_:
        label_encoder.classes_ = np.append(label_encoder.classes_, mode_value)
    return label_encoder.transform([mode_value])[0]

# Main Streamlit app
def main():
    st.title('Rain Prediction App')

    # Load the pretrained model
    model = load_pretrained_model()

    if model is None:
        st.error("Failed to load the model. Please check the logs for details.")
        return

    # User inputs
    st.subheader('Enter the weather details:')
    location = st.selectbox('Location', label_encoder_location.classes_)
    min_temp = st.number_input('MinTemp', min_value=float(data['MinTemp'].min()), max_value=float(data['MinTemp'].max()), value=10.0)
    max_temp = st.number_input('MaxTemp', min_value=float(data['MaxTemp'].min()), max_value=float(data['MaxTemp'].max()), value=20.0)
    wind_gust_dir = st.selectbox('WindGustDir', label_encoder_wind_gust_dir.classes_)
    wind_gust_speed = st.number_input('WindGustSpeed', min_value=float(data['WindGustSpeed'].min()), max_value=float(data['WindGustSpeed'].max()), value=30.0)

    if st.button('Predict'):
        # Encode user input
        encoded_location = label_encoder_location.transform([location])[0]
        encoded_wind_gust_dir = label_encoder_wind_gust_dir.transform([wind_gust_dir])[0]

        # Handle unseen labels in WindDir9am and WindDir3pm
        wind_dir_9am_mode = data['WindDir9am'].mode()[0]
        wind_dir_3pm_mode = data['WindDir3pm'].mode()[0]

        encoded_wind_dir_9am = check_and_update_label_encoder(label_encoder_wind_dir_9am, data['WindDir9am'], wind_dir_9am_mode)
        encoded_wind_dir_3pm = check_and_update_label_encoder(label_encoder_wind_dir_3pm, data['WindDir3pm'], wind_dir_3pm_mode)

        # Prepare user data for prediction
        user_data = {
            'Location': encoded_location,
            'MinTemp': min_temp,
            'MaxTemp': max_temp,
            'Rainfall': data['Rainfall'].mode()[0],
            'Evaporation': data['Evaporation'].mode()[0],
            'Sunshine': data['Sunshine'].mode()[0],
            'WindGustDir': encoded_wind_gust_dir,
            'WindGustSpeed': wind_gust_speed,
            'WindDir9am': encoded_wind_dir_9am,
            'WindDir3pm': encoded_wind_dir_3pm,
            'WindSpeed9am': data['WindSpeed9am'].mode()[0],
            'WindSpeed3pm': data['WindSpeed3pm'].mode()[0],
            'Humidity9am': data['Humidity9am'].mode()[0],
            'Humidity3pm': data['Humidity3pm'].mode()[0],
            'Pressure9am': data['Pressure9am'].mode()[0],
            'Pressure3pm': data['Pressure3pm'].mode()[0],
            'Cloud9am': data['Cloud9am'].mode()[0],
            'Cloud3pm': data['Cloud3pm'].mode()[0],
            'Temp9am': data['Temp9am'].mode()[0],
            'Temp3pm': data['Temp3pm'].mode()[0],
            'RainToday': data['RainToday'].mode()[0]
        }
        user_data_df = pd.DataFrame(user_data, index=[0])

        # Scale user data
        user_data_scaled = scaler.transform(user_data_df)

        # Print the shape of user_data_scaled for debugging
        st.write(f"Shape of user_data_scaled: {user_data_scaled.shape}")

        # Predict
        try:
            prediction = model.predict(user_data_scaled)
            prediction_result = "Yes" if prediction[0][0] >= 0.5 else "No"
            st.write(f'Will it rain tomorrow? {prediction_result}')
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

if __name__ == '__main__':
    main()
