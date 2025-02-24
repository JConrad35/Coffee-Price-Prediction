import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load the trained model from the joblib file
def load_model(model_path):
    """
    Load the trained model from the specified path.
    """
    model = joblib.load(model_path)
    return model

# Load the scaler from the joblib file
def load_scaler(scaler_path):
    """
    Load the scaler from the specified path.
    """
    scaler = joblib.load(scaler_path)
    return scaler

# Make predictions using the loaded model
def make_prediction(model, scaler, input_data):
    """
    Make a prediction using the loaded model.
    """
    input_data = np.array(input_data).reshape(1, 10, 1)  # Reshape input data to 3D array
    prediction_scaled = model.predict(input_data)
    prediction = scaler.inverse_transform(prediction_scaled)  # Inverse transform to get unscaled prediction
    return prediction

# Calculate RMSE
def calculate_rmse(actual, predicted):
    """
    Calculate the Root Mean Squared Error (RMSE) between actual and predicted values.
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return rmse

# Example usage
if __name__ == "__main__":
    model_path = r'C:\Users\julia\OneDrive\Documents\ML Projects\Coffee-Price-Prediction\coffee_project\model.joblib'
    scaler_path = r'C:\Users\julia\OneDrive\Documents\ML Projects\Coffee-Price-Prediction\coffee_project\scaler.joblib'
    
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    
    # Load y_test from the .npy file
    y_test = np.load(r'C:\Users\julia\OneDrive\Documents\ML Projects\Coffee-Price-Prediction\coffee_project\y_test.npy')

    input_data = y_test[-11:-1]  # Use the last 11 values of y_test, excluding the very last value
    actual_value = [y_test[-1]]  # Use the very last value of y_test
    actual_value = scaler.inverse_transform(np.array(actual_value).reshape(-1, 1))
    prediction = make_prediction(model, scaler, input_data)
    
    # Calculate and print RMSE
    rmse = calculate_rmse(actual_value, prediction)
    print(f"Actual Value: {actual_value}")
    print(f"Prediction: {prediction}")
    print(f"RMSE: {rmse}")