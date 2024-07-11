import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import load_model
from io import BytesIO
import requests
import tempfile
import os

# Function to load the pretrained model
@st.cache(allow_output_mutation=True)
def load_pretrained_model():
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

# Load data from URL
url = "https://raw.githubusercontent.com/Caesarblack27/rain-prediction/main/weatherAUS.csv"
data = pd.read_csv(url)

# Fill missing values
data.fillna(data.mode().iloc[0], inplace=True)

# Encode categorical variables
label_encoder_location = LabelEncoder()
label_encoder_location.fit(data['Location'].unique())

label_encoder_wind_gust_dir = LabelEncoder()
label_encoder_wind_gust_dir.fit(data['WindGustDir'].unique())

# Feature and label separation
X = data[['Location', 'MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed']]
y = data['RainTomorrow']

# Scale features
scaler = StandardScaler()

# Encode categorical variables before scaling
X['Location'] = label_encoder_location.transform(X['Location'])
X['WindGustDir'] = label_encoder_wind_gust_dir.transform(X['WindGustDir'])

# Fit and transform features
X = scaler.fit_transform(X)

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
        # Encode user input
        encoded_location = label_encoder_location.transform([location])[0]
        encoded_wind_gust_dir = label_encoder_wind_gust_dir.transform([wind_gust_dir])[0]

        # Prepare user data for prediction
        user_data = scaler.transform([[encoded_location, min_temp, max_temp, encoded_wind_gust_dir, wind_gust_speed]])

        # Predict
        prediction = model.predict(user_data)
        prediction_result = "Yes" if prediction[0][0] >= 0.5 else "No"
        st.write(f'Will it rain tomorrow? {prediction_result}')

if __name__ == '__main__':
    main()
