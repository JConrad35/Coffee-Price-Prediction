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
    input_data = np.array(input_data).reshape(-1, 1)  # Reshape input data to 2D array
    input_data_scaled = scaler.transform(input_data).reshape(1, 10, 1)  # Scale and reshape to 3D array
    prediction_scaled = model.predict(input_data_scaled)
    prediction = scaler.inverse_transform(prediction_scaled)  # Inverse transform to get unscaled prediction
    return prediction

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
    
    # Example input data (sequence of 10 time steps)
    # from model_training import y_test_unscaled  # Import y_test_unscaled from model_training.py

    input_data = y_test_unscaled[-11:-1]  # Use the last 11 values of y_test_unscaled, excluding the very last value
    actual_value = [y_test_unscaled[-1]]  # Use the very last value of y_test_unscaled

    prediction = make_prediction(model, scaler, input_data)
    rmse = calculate_rmse(actual_value, prediction)
    print(f"RMSE: {rmse}")
    print(f"Prediction: {prediction}")
    print(y_test_unscaled[-1])
