import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout, BatchNormalization # type: ignore
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# data preprocessing
df = pd.read_csv(r"C:\Users\julia\OneDrive\Documents\ML Projects\Coffee-Price-Prediction\coffee_output_data.csv")
df = df.iloc[::-1].reset_index(drop=True)   # Change order of the data
df.drop('index', axis=1)

# Function to change units from cents per pound to dollars per pound
def dollar_per_pound(value):
    new_value = value * 0.01
    return round(new_value, 2)

# Apply function to dataset
df['new_values'] = df['value'].apply(dollar_per_pound)
df = df.drop(columns = ['index', 'value'])

# ScaNormalize data
scaler = MinMaxScaler()
df_values_scaled = scaler.fit_transform(df['new_values'].values.reshape(-1, 1))

# Creating sequence
seq_length = 10   # Timestamps
X = []
y = []

for i in range(seq_length, len(df_values_scaled)):
    X.append(df_values_scaled[i-seq_length:i, 0])   # Extract past sequence
    y.append(df_values_scaled[i, 0])   # Next value

X = np.array(X)
y = np.array(y)

# Train-Test Split
SEED = 2
np.random.seed(SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, shuffle=False)

# Reshape test and training data into 3D format
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Building LSTM Layers
model = Sequential([
    LSTM(units=150, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),   # Prevent overfitting
    BatchNormalization(),   # Stabilizes the learning process
    LSTM(units=75, activation='relu', return_sequences=True),
    Dropout(0.2),
    BatchNormalization(),
    LSTM(units=75, activation='relu', return_sequences=True),
    Dropout(0.2),
    BatchNormalization(),
    LSTM(units=50, activation='relu', return_sequences=False),
    Dropout(0.2),
    BatchNormalization(),
    Dense(units=25, activation='relu'),   # Fully connected layer
    Dense(1),   # Output regression layer
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Train model
history = model.fit(X_train, y_train, epochs=200, batch_size=41, validation_split=0.2)

# Evaluate model
test_loss = model.evaluate(X_test, y_test)

# Make predictions
y_pred = model.predict(X_test)

# Reshape y_pred and y_test from 3D to 1D
y_pred = y_pred.flatten()
y_test = y_test.flatten()

# Performance metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
print("Mean Absolute Error; ", mae)
print("Root Mean Squared Error: ", rmse)

# Function to inverse transformed the data to its unscaled state
def min_max_inv(value, scaler):
    inv = value*(scaler.data_max_-scaler.data_min_)+scaler.data_min_
    return inv

y_test_unscaled = min_max_inv(y_test, scaler)
y_pred_unscaled = min_max_inv(y_pred, scaler)
X_test_unscaled = min_max_inv(X_test, scaler)

print("y_test_unscaled: ", y_test_unscaled)
print("y_pred_unscaled: ", y_pred_unscaled)

