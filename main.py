import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

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

# Prepare dataset for training
X = data[all_features]
y = data['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to create the model
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main Streamlit app
def main():
    st.title('Rain Prediction App')

    # Initialize user input variables
    user_inputs = {}

    # Collect user inputs for each feature
    for feature in all_features:
        if feature == 'Location':
            user_inputs[feature] = st.selectbox(feature, label_encoder_location.inverse_transform(np.arange(len(label_encoder_location.classes_))))
        elif feature == 'WindGustDir':
            user_inputs[feature] = st.selectbox(feature, label_encoder_wind_gust_dir.inverse_transform(np.arange(len(label_encoder_wind_gust_dir.classes_))))
        elif feature == 'WindDir9am' or feature == 'WindDir3pm':
            user_inputs[feature] = st.selectbox(feature, label_encoder_wind_dir_9am.inverse_transform(np.arange(len(label_encoder_wind_dir_9am.classes_))))
        else:
            user_inputs[feature] = st.number_input(feature, value=float(data[feature].mode()[0]))

    if st.button('Train Model'):
        # Create and train the model
        model = create_model(X_train_scaled.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        history = model.fit(X_train_scaled, y_train, epochs=20, validation_split=0.2, callbacks=[early_stopping])

        # Save the model
        model.save('rain_prediction_model.h5')
        st.success('Model trained and saved successfully!')

    if st.button('Predict'):
        try:
            # Check and handle unseen labels for categorical variables
            for feature in ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']:
                if user_inputs[feature] not in label_encoder_location.classes_:
                    st.warning(f"Unseen label '{user_inputs[feature]}' for '{feature}', using most common label instead.")
                    user_inputs[feature] = label_encoder_location.classes_[0]  # Use most common label

            # Convert user inputs to DataFrame
            user_data_df = pd.DataFrame([user_inputs])

            # Encode user input categorical data
            user_data_df['Location'] = label_encoder_location.transform(user_data_df['Location'])
            user_data_df['WindGustDir'] = label_encoder_wind_gust_dir.transform(user_data_df['WindGustDir'])
            user_data_df['WindDir9am'] = label_encoder_wind_dir_9am.transform(user_data_df['WindDir9am'])
            user_data_df['WindDir3pm'] = label_encoder_wind_dir_3pm.transform(user_data_df['WindDir3pm'])

            # Ensure numeric data is in correct format for scaling
            numeric_user_data = user_data_df[all_features].apply(pd.to_numeric, errors='coerce').fillna(0)

            # Scale numeric user data
            scaled_user_data = scaler.transform(numeric_user_data)

            # Load the trained model
            model = load_model('rain_prediction_model.h5')

            # Make prediction
            prediction = model.predict(scaled_user_data)
            prediction_result = "Yes" if prediction[0][0] >= 0.5 else "No"
            st.write(f'Will it rain tomorrow? {prediction_result}')

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

if __name__ == '__main__':
    main()
