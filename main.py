import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
import requests
from io import BytesIO

# Load data from URL
url = "https://raw.githubusercontent.com/Caesarblack27/rain-prediction/main/weatherAUS.csv"
data = pd.read_csv(url)

# Fill missing values
data.fillna(data.mode().iloc[0], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_cols = ['Location', 'WindGustDir']
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Feature and label separation
X = data[['Location', 'MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed']]
y = data['RainTomorrow']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split (not needed here since we're using a pre-trained model)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to load pre-trained model
def load_pretrained_model():
    # GitHub URL for the pre-trained model file
    model_url = "https://github.com/Caesarblack27/rain-prediction/raw/main/rain_prediction_model.h5"
    
    # Fetch the model file from GitHub
    response = requests.get(model_url)
    response.raise_for_status()
    
    # Load the model from memory
    model = load_model(BytesIO(response.content))
    
    return model

# Load pre-trained model
model = load_pretrained_model()

# Predict rain function
def predict_rain(model, input_data):
    input_data_scaled = scaler.transform(input_data)
    return model.predict(input_data_scaled)

# Main Streamlit app
def main():
    st.title('Rain Prediction App')

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
