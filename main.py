import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from io import BytesIO
import requests

# Load data from URL
url = "https://raw.githubusercontent.com/Caesarblack27/rain-prediction/main/weatherAUS.csv"
data = pd.read_csv(url)

# Fill missing values
data.fillna(data.mode().iloc[0], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Feature and label separation
X = data[['Location', 'MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed']]
y = data['RainTomorrow']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to load pretrained model
def load_pretrained_model():
    model_url = "https://github.com/Caesarblack27/rain-prediction/raw/main/rain_prediction_model.h5"
    try:
        response = requests.get(model_url)
        response.raise_for_status()
        model = load_model(BytesIO(response.content))
        return model
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        return None
    
    # Load the model from memory
    model = load_model(BytesIO(response.content))
    
    return model

# Predict rain
def predict_rain(model, input_data):
    input_data_scaled = scaler.transform(input_data)
    return model.predict(input_data_scaled)

# Main Streamlit app
def main():
    st.title('Rain Prediction App')

    # Build and train the model
    model = load_pretrained_model()

    # User inputs
    st.subheader('Enter the weather details:')
    location = st.selectbox('Location', data['Location'].unique())
    min_temp = st.number_input('MinTemp', min_value=float(data['MinTemp'].min()), max_value=float(data['MinTemp'].max()), value=10.0)
    max_temp = st.number_input('MaxTemp', min_value=float(data['MaxTemp'].min()), max_value=float(data['MaxTemp'].max()), value=20.0)
    wind_gust_dir = st.selectbox('WindGustDir', data['WindGustDir'].unique())
    wind_gust_speed = st.number_input('WindGustSpeed', min_value=float(data['WindGustSpeed'].min()), max_value=float(data['WindGustSpeed'].max()), value=30.0)

    if st.button('Predict'):
        # Encode and scale user input
        user_data = pd.DataFrame({
            'Location': [label_encoder.transform([location])[0]],
            'MinTemp': [min_temp],
            'MaxTemp': [max_temp],
            'WindGustDir': [label_encoder.transform([wind_gust_dir])[0]],
            'WindGustSpeed': [wind_gust_speed]
        })

        # Predict
        prediction = predict_rain(model, user_data)
        st.write(f'Will it rain today? {"Yes" if prediction[0][0] >= 0.5 else "No"}')

if __name__ == '__main__':
    main()
