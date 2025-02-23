import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

# Example usage
if __name__ == "__main__":
    model_path = r'C:\Users\julia\OneDrive\Documents\ML Projects\Coffee-Price-Prediction\coffee_project\model.joblib'
    scaler_path = r'C:\Users\julia\OneDrive\Documents\ML Projects\Coffee-Price-Prediction\coffee_project\scaler.joblib'
    
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    
    # Example input data (sequence of 10 time steps)
    input_data = [2.09, 2.09, 2.39, 2.31, 2.48, 2.57, 2.61, 2.78, 2.77, 3.05]
    
    prediction = make_prediction(model, scaler, input_data)
    print(f"Prediction: {prediction}")