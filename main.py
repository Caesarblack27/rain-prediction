import streamlit as st
import pandas as pd
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

# Select only the required features
selected_features = ['Location', 'MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed']
X = data[selected_features]
y = data['RainTomorrow']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Default values for other features
default_values = {
    'Rainfall': data['Rainfall'].mean(),
    'Evaporation': data['Evaporation'].mean(),
    'Sunshine': data['Sunshine'].mean(),
    'WindDir9am': label_encoder_wind_gust_dir.transform([data['WindDir9am'].mode()[0]])[0],
    'WindDir3pm': label_encoder_wind_gust_dir.transform([data['WindDir3pm'].mode()[0]])[0],
    'WindSpeed9am': data['WindSpeed9am'].mean(),
    'WindSpeed3pm': data['WindSpeed3pm'].mean(),
    'Humidity9am': data['Humidity9am'].mean(),
    'Humidity3pm': data['Humidity3pm'].mean(),
    'Pressure9am': data['Pressure9am'].mean(),
    'Pressure3pm': data['Pressure3pm'].mean(),
    'Cloud9am': data['Cloud9am'].mean(),
    'Cloud3pm': data['Cloud3pm'].mean(),
    'Temp9am': data['Temp9am'].mean(),
    'Temp3pm': data['Temp3pm'].mean(),
    'RainToday': label_encoder_location.transform([data['RainToday'].mode()[0]])[0]
}

# Main Streamlit app
def main():
    st.title('Rain Prediction App')

    # Build and train the model
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

        # Prepare user data for prediction
        user_data = {
            'Location': encoded_location,
            'MinTemp': min_temp,
            'MaxTemp': max_temp,
            'WindGustDir': encoded_wind_gust_dir,
            'WindGustSpeed': wind_gust_speed,
            'Rainfall': default_values['Rainfall'],
            'Evaporation': default_values['Evaporation'],
            'Sunshine': default_values['Sunshine'],
            'WindDir9am': default_values['WindDir9am'],
            'WindDir3pm': default_values['WindDir3pm'],
            'WindSpeed9am': default_values['WindSpeed9am'],
            'WindSpeed3pm': default_values['WindSpeed3pm'],
            'Humidity9am': default_values['Humidity9am'],
            'Humidity3pm': default_values['Humidity3pm'],
            'Pressure9am': default_values['Pressure9am'],
            'Pressure3pm': default_values['Pressure3pm'],
            'Cloud9am': default_values['Cloud9am'],
            'Cloud3pm': default_values['Cloud3pm'],
            'Temp9am': default_values['Temp9am'],
            'Temp3pm': default_values['Temp3pm'],
            'RainToday': default_values['RainToday']
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
