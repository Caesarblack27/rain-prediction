import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import callbacks
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Load data from URL
url = "https://raw.githubusercontent.com/Caesarblack27/rain-prediction/main/weatherAUS.csv"
data = pd.read_csv(url)

# Fill missing values
data.fillna(data.mode().iloc[0], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Feature and label separation
X = data[['Location', 'MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed']]
y = data['RainTomorrow']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train NN model
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

# Predict rain
def predict_rain(model, input_data):
    input_data_scaled = scaler.transform(input_data)
    return model.predict(input_data_scaled)

# Main Streamlit app
def main():
    st.title('Rain Prediction App')

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
