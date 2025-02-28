import tensorflow as tf
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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
        # Compute 95% confidence interval using quantiles (2.5th and 97.5th percentiles)
    lower_bound = np.quantile(predictions, 0.025, axis=0)
    upper_bound = np.quantile(predictions, 0.975, axis=0)  
    return mean_predictions, uncertainty, lower_bound, upper_bound

# Evaluate the model
loss, rmse = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, RMSE: {rmse}")

# Use Monte Carlo Dropout for predictions
mean_predictions, uncertainty, lower_bound, upper_bound = monte_carlo_dropout_predict(model, X_test, n_samples)
print(f"Mean Predictions: \n{mean_predictions}")
print(f"Uncertainty: \n{uncertainty}")
print(f"CI : \n{[lower_bound, upper_bound]}")

# Plot the mean predictions, their confidence intervals, and y_test
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Actual Prices')
plt.plot(mean_predictions, label='Predicted Prices')
plt.fill_between(range(len(y_test)), lower_bound.flatten(), upper_bound.flatten(), color='gray', alpha=0.2, label='95% Confidence Interval')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Coffee Price Prediction with Confidence Intervals')
plt.show()
