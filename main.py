import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras import callbacks
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Function to encode cyclic features
def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

# Function to preprocess data
def preprocess_data(data):
    # Perform preprocessing steps
    # Encoding datetime features cyclically
    data['Date'] = pd.to_datetime(data['Date'])
    data['month'] = data.Date.dt.month
    data = encode(data, 'month', 12)
    data['day'] = data.Date.dt.day
    data = encode(data, 'day', 31)
    
    # Selecting relevant columns
    data = data[['Location', 'MinTemp', 'MaxTemp', 'WindGustSpeed', 'RainToday', 'RainTomorrow', 'month_sin', 'month_cos', 'day_sin', 'day_cos']]
    
    # Filling missing values
    data.fillna(data.mode().iloc[0], inplace=True)
    
    # Encoding categorical variables
    label_encoder = LabelEncoder()
    data['Location'] = label_encoder.fit_transform(data['Location'])
    
    # Encoding target variables
    data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0})
    data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})
    
    # Scaling numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data.drop(['RainToday', 'RainTomorrow'], axis=1))
    data_scaled = pd.DataFrame(scaled_features, columns=data.drop(['RainToday', 'RainTomorrow'], axis=1).columns)
    
    return data_scaled, data['RainToday'], data['RainTomorrow']

# Function to build and train NN model
def build_model(X_train, y_train):
    model = Sequential()
    model.add(Dense(units=32, kernel_initializer='uniform', activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    
    opt = Adam(learning_rate=0.00009)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = callbacks.EarlyStopping(min_delta=0.001, patience=20, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, batch_size=32, epochs=150, callbacks=[early_stopping], validation_split=0.2)
    
    return model

# Main Streamlit app
def main():
    st.title('Rain Prediction App')
    
    # Load data from URL
    url = "https://raw.githubusercontent.com/Caesarblack27/rain-prediction/main/weatherAUS.csv"
    data = pd.read_csv(url)
    
    st.write(data.head())
    
    # Preprocess the data
    X, y_today, y_tomorrow = preprocess_data(data)
    
    # Split data into train and test sets
    X_train, X_test, y_train_today, y_test_today = train_test_split(X, y_today, test_size=0.2, random_state=42)
    _, _, y_train_tomorrow, y_test_tomorrow = train_test_split(X, y_tomorrow, test_size=0.2, random_state=42)
    
    # Build and train the model for RainToday
    model_today = build_model(X_train, y_train_today)
    
    # Build and train the model for RainTomorrow
    model_tomorrow = build_model(X_train, y_train_tomorrow)
    
    # Display evaluation metrics
    st.subheader('Model Evaluation Metrics for RainToday')
    y_pred_today = model_today.predict(X_test)
    y_pred_today = (y_pred_today > 0.5)
    st.write(classification_report(y_test_today, y_pred_today))
    
    st.subheader('Model Evaluation Metrics for RainTomorrow')
    y_pred_tomorrow = model_tomorrow.predict(X_test)
    y_pred_tomorrow = (y_pred_tomorrow > 0.5)
    st.write(classification_report(y_test_tomorrow, y_pred_tomorrow))
    
    # User input for prediction
    st.subheader('Make a Prediction')
    location = st.selectbox('Location', data['Location'].unique())
    min_temp = st.number_input('Min Temp')
    max_temp = st.number_input('Max Temp')
    wind_gust_speed = st.number_input('Wind Gust Speed')
    date = st.date_input('Date')
    
    input_data = pd.DataFrame({
        'Location': [location],
        'MinTemp': [min_temp],
        'MaxTemp': [max_temp],
        'WindGustSpeed': [wind_gust_speed],
        'month_sin': [np.sin(2 * np.pi * date.month / 12)],
        'month_cos': [np.cos(2 * np.pi * date.month / 12)],
        'day_sin': [np.sin(2 * np.pi * date.day / 31)],
        'day_cos': [np.cos(2 * np.pi * date.day / 31)]
    })
    
    input_data['Location'] = LabelEncoder().fit(data['Location']).transform(input_data['Location'])
    scaled_input = StandardScaler().fit(X).transform(input_data)
    
    prediction_today = model_today.predict(scaled_input)
    prediction_today = 'Yes' if prediction_today > 0.5 else 'No'
    
    prediction_tomorrow = model_tomorrow.predict(scaled_input)
    prediction_tomorrow = 'Yes' if prediction_tomorrow > 0.5 else 'No'
    
    st.write(f'Will it rain today? {prediction_today}')
    st.write(f'Will it rain tomorrow? {prediction_tomorrow}')

if __name__ == '__main__':
    main()
