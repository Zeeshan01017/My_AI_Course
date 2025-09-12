# TF Gated Recurrent Unit Network apply on Air passenger dataset
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from keras.metrics import Precision, Recall

df = pd.read_csv("Machine_learning_practice/AirPassengers.csv",parse_dates=["Month"], index_col="Month")
print(df.head())
 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df.values)

def create_dataset(data , time_step=1):
    x, y = [],[]
    for i in range(len(data) - time_step - 1):
        x.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step , 0])
    return np.array(x), np.array(y)
time_step = 100
x, y = create_dataset(scaled_data, time_step)
x = x.reshape(x.shape[0], x.shape[1], 1)

model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(x.shape[1],1)))
model.add(GRU(units=50))
model.add(Dense(units=1))

METRICS = metrics = ['accuracy',
                     Precision(name='precision'),
                     Recall(name='recall')]
model.compile(optimizer=Adam(learning_rate=0.001), loss = 'mean_squared_error', metrics = METRICS)

model.fit(x,y, epochs=10, batch_size = 32)

input_sequence = scaled_data[-time_step:].reshape(1,time_step,1)
predict_values = model.predict(input_sequence)
predicted_values = scaler.inverse_transform(predict_values)
print(f"The predict passengers for the next day is: {predicted_values[0][0]:.0f}")