import tensorflow as tf
from keras.models import load_model
import numpy as np

# Load the best model
model = load_model('C:/Users/julia/OneDrive/Documents/ML Projects/Coffee-Price-Prediction/coffee_project/best_model.h5')

# Monte Carlo Dropout
n_samples = 100

def monte_carlo_dropout_predict(model, X, n_samples):
    predictions = []
    for _ in range(n_samples):
        predictions.append(model(X, training=True))

    predictions = np.array(predictions)
    mean_predictions = predictions.mean(axis=0)
    uncertainty = predictions.var(axis=0)

    return mean_predictions, uncertainty

# Ensure X_test is defined and converted to numpy array
X_test = np.array(X_test)  # This line assumes X_test is already defined

# Evaluate the model
loss, rmse = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, RMSE: {rmse}")

# Use Monte Carlo Dropout for predictions
mean_predictions, uncertainty = monte_carlo_dropout_predict(model, X_test, n_samples)
print(f"Mean Predictions: {mean_predictions}")
print(f"Uncertainty: {uncertainty}")
