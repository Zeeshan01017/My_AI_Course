# AIR PASSENGERS TIME SERIES FORECASTING
# MODELS: RNN, LSTM, GRU

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout

# Load Data
df = pd.read_csv("Prectical_Development_programing_Assesment/AirPassengers.csv")
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
data = df['#Passengers'].values.reshape(-1,1)

# Scale Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(data)

# Sequence Creation Function
def create_sequences(data, look_back=12):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

look_back = 12
X, y = create_sequences(scaled, look_back)

# Train/Test Split
split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Training samples:", X_train.shape, "Testing samples:", X_test.shape)

def build_and_train(model_type, X_train, y_train, X_test, y_test, units=64, epochs=60):
    model = Sequential()
    
    if model_type == 'RNN':
        model.add(SimpleRNN(units, activation='tanh', input_shape=(X_train.shape[1], 1)))
    elif model_type == 'LSTM':
        model.add(LSTM(units, activation='tanh', input_shape=(X_train.shape[1], 1)))
    elif model_type == 'GRU':
        model.add(GRU(units, activation='tanh', input_shape=(X_train.shape[1], 1)))
    else:
        raise ValueError("Invalid model type")

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    history = model.fit(X_train, y_train, validation_data=(X_test,y_test),
                        epochs=epochs, batch_size=16, verbose=0)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test)

    # Metrics
    rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae  = mean_absolute_error(y_test_inv, y_pred_inv)

    return model, history, y_pred_inv, y_test_inv, rmse, mae


results = {}

for m in ['RNN', 'LSTM', 'GRU']:
    print(f"\nTraining {m} model...")
    model, history, y_pred, y_true, rmse, mae = build_and_train(
        m, X_train, y_train, X_test, y_test
    )
    results[m] = {'model': model, 'history': history,
                  'y_pred': y_pred, 'y_true': y_true,
                  'RMSE': rmse, 'MAE': mae}

    # Plot Loss Curve
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f"{m} - Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    # Plot Predictions
    plt.figure(figsize=(10,5))
    plt.plot(df.index[look_back+len(y_train):look_back+len(y_train)+len(y_true)],
             y_true, label="Actual")
    plt.plot(df.index[look_back+len(y_train):look_back+len(y_train)+len(y_true)],
             y_pred, label=f"{m} Predicted")
    plt.title(f"{m} Predictions vs Actual")
    plt.xlabel("Year")
    plt.ylabel("Passengers")
    plt.legend()
    plt.show()


metrics_df = pd.DataFrame({
    "RMSE": [results[m]['RMSE'] for m in results],
    "MAE" : [results[m]['MAE']  for m in results]
}, index=results.keys())

print("\nModel Performance Metrics:")
print(metrics_df)

# Barplot for easy comparison
metrics_df.plot(kind='bar', figsize=(8,5))
plt.title("Model Performance Comparison")
plt.ylabel("Error")
plt.show()
