import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint

model = Sequential([
    LSTM(units=150, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.5),
    BatchNormalization(),
    LSTM(units=75, activation='relu', return_sequences=True),
    Dropout(0.5),
    BatchNormalization(),
    LSTM(units=75, activation='relu', return_sequences=True),
    BatchNormalization(),
    LSTM(units=50, activation='relu', return_sequences=False),
    BatchNormalization(),
    Dense(units=25, activation='relu'),
    Dense(1),
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='C:/Users/julia/OneDrive/Documents/ML Projects/Coffee-Price-Prediction/coffee_project/best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Train the model with the checkpoint callback
history = model.fit(X_train, y_train, epochs=200, batch_size=41, validation_split=0.2, callbacks=[checkpoint_callback])