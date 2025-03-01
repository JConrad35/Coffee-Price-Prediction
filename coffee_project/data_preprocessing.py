import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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