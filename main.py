import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
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
    # Perform preprocessing steps (similar to what you did)
    # Encoding datetime features cyclically
    data['Date'] = pd.to_datetime(data['Date'])
    data['month'] = data.Date.dt.month
    data = encode(data, 'month', 12)
    data['day'] = data.Date.dt.day
    data = encode(data, 'day', 31)
    
    # Filling missing values
    data.fillna(data.mode().iloc[0], inplace=True)
    
    # Encoding categorical variables
    label_encoder = LabelEncoder()
    object_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
    for col in object_cols:
        data[col] = label_encoder.fit_transform(data[col])
    
    # Scaling numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data.drop(['RainTomorrow', 'Date', 'day', 'month'], axis=1))
    data_scaled = pd.DataFrame(scaled_features, columns=data.drop(['RainTomorrow', 'Date', 'day', 'month'], axis=1).columns)
    
    return data_scaled, data['RainTomorrow']

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

# Function to predict rain
def predict_rain(model, input_data):
    # Scaling input data
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    return model.predict(input_data_scaled)

# Main Streamlit app
def main():
    st.title('Rain Prediction App')
    
    # Load data from URL
    url = "https://raw.githubusercontent.com/Caesarblack27/rain-prediction/main/weatherAUS.csv"
    data = pd.read_csv(url)
    
    # Preprocess the data
    X, y = preprocess_data(data)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train the model
    model = build_model(X_train, y_train)
    
    # User inputs
    st.subheader('Enter the weather details:')
    location = st.selectbox('Location', data['Location'].unique())
    min_temp = st.number_input('MinTemp', min_value=float(data['MinTemp'].min()), max_value=float(data['MinTemp'].max()), value=10.0)
    max_temp = st.number_input('MaxTemp', min_value=float(data['MaxTemp'].min()), max_value=float(data['MaxTemp'].max()), value=20.0)
    wind_gust_dir = st.selectbox('WindGustDir', data['WindGustDir'].unique())
    wind_gust_speed = st.number_input('WindGustSpeed', min_value=float(data['WindGustSpeed'].min()), max_value=float(data['WindGustSpeed'].max()), value=30.0)
    
    if st.button('Predict'):
        # Preprocess user input
        user_data = pd.DataFrame({
            'Location': [location],
            'MinTemp': [min_temp],
            'MaxTemp': [max_temp],
            'WindGustDir': [wind_gust_dir],
            'WindGustSpeed': [wind_gust_speed],
            'month_sin': [0], 'month_cos': [0], 'day_sin': [0], 'day_cos': [0]
        })
        
        # Encode and scale user input
        user_data['Location'] = LabelEncoder().fit_transform(user_data['Location'])
        user_data['WindGustDir'] = LabelEncoder().fit_transform(user_data['WindGustDir'])
        user_data_scaled = StandardScaler().fit_transform(user_data)
        
        # Predict
        prediction = predict_rain(model, user_data_scaled)
        st.write(f'Will it rain today? {"Yes" if prediction[0] >= 0.5 else "No"}')

if __name__ == '__main__':
    main()
