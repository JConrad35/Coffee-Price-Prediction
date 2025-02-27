import tensorflow as tf
from keras.models import load_model

# Load the best model
model = load_model('C:\Users\julia\OneDrive\Documents\ML Projects\Coffee-Price-Prediction\coffee_project\best_model.h5')

# Load the state dictionary
# model.load_weights('C:/Users/julia/OneDrive/Documents/ML Projects/Coffee-Price-Prediction/coffee_project/best_model.h5')

# Monte Crarlo Dropout
input_tensor = tf.random.normal(shape=(1, 100))
num_samples = 100
predictions = tf.stack([model(input_tensor, training=True) 
                        for _ in range(num_samples)]) # Perform multiple forward passes with dropout enabled

mean_prediction = tf.reduce_mean(predictions, axis=0)
std_prediction = tf.math.reduce_std(predictions, axis=0)

print("Mean Prediction:", mean_prediction.numpy())
print("Standard Deviation (Uncertainty):", std_prediction.numpy())

# Evaluate the model
loss, rmse = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, RMSE: {rmse}")
