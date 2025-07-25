import sys
import tensorflow.keras.layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

data = pd.read_csv(r'C:\Users\Dell\Desktop\Mine\DATASET\Apple stock\AAPL_stock_data.csv', skiprows=2)
data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
print("Dataset size:", len(data))
print(data.head())
print(data.columns)

dates = data['Date'].values
data_with_dates = data[['Date', 'Close']].dropna()
dates = data_with_dates['Date'].values
data = data_with_dates[['Close']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
print("scaled_data shape:", scaled_data.shape)
print("First 5 rows of scaled_data:", scaled_data[:5])
print("Number of dates after dropna:", len(dates))

time_steps = 10
if len(scaled_data) <= time_steps:
    raise ValueError(f"Dataset too small ({len(scaled_data)} rows) for time_steps={time_steps}. Need at least {time_steps+1} rows.")

def create_sequences(data, window):
    X, Y = [], []
    print("Data length:", len(data))
    print("Window size:", window)
    for i in range(window, len(data)):
        sequence = data[i-window:i, 0]
        X.append(sequence)
        Y.append(data[i, 0])
    print("Number of sequences created:", len(X))
    return np.array(X), np.array(Y)

X, Y = create_sequences(scaled_data, window=time_steps)
X = X.reshape(X.shape[0], X.shape[1], 1)
print("X shape:", X.shape)
print("Y shape:", Y.shape)

split = int(0.8 * len(X))
if split == 0:
    raise ValueError(f"Too few sequences ({len(X)}) for training. Increase dataset size or reduce time_steps.")
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

if X_train.shape[0] == 0:
    raise ValueError("X_train is empty. Reduce time_steps or increase dataset size.")

model = Sequential([
    Input(shape=(time_steps, 1)),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

model.fit(X_train, Y_train, batch_size=32, epochs=20, validation_data=(X_test, Y_test))

predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted.reshape(-1, 1))
real = scaler.inverse_transform(Y_test.reshape(-1, 1))

plt.figure(figsize=(14, 5))
dates_for_plot = dates[split + time_steps:split + time_steps + len(real)]
print("Length of dates_for_plot:", len(dates_for_plot))
print("Length of real:", len(real))
print("Sample dates:", dates_for_plot[:5])  

plt.plot(dates_for_plot, real, label='Real Price')
plt.plot(dates_for_plot, predicted, label='Predicted Price')
plt.xticks(dates_for_plot[::50], rotation=45, ha='right')  
plt.legend()
plt.title('Stock Price Prediction')
plt.tight_layout()
plt.show()

rmse = np.sqrt(mean_squared_error(real, predicted))
print(f'RMSE: {rmse}')

last_sequence = scaled_data[-time_steps:]
last_sequence = np.array(last_sequence, dtype=np.float32).reshape(1, time_steps, 1)
print("last_sequence shape:", last_sequence.shape)

future_price = model.predict(last_sequence)
future_price = scaler.inverse_transform(future_price)
print("Predicted next day's price:", future_price[0][0])

model.save(r'C:\Users\Dell\Desktop\Mine\DATASET\Apple stock\stock_lstm_model.keras')

results = pd.DataFrame({
    'Date': dates_for_plot,
    'Real Price': real.flatten(),
    'Predicted Price': predicted.flatten()
})
results.to_csv(r'C:\Users\Dell\Desktop\Mine\DATASET\Apple stock\predictions.csv', index=False)

future_predictions = []
current_sequence = scaled_data[-time_steps:].copy()
for _ in range(7):
    current_sequence_reshaped = np.array(current_sequence, dtype=np.float32).reshape(1, time_steps, 1)
    next_pred = model.predict(current_sequence_reshaped)
    future_predictions.append(next_pred[0, 0])
    current_sequence = np.append(current_sequence[1:], next_pred, axis=0)
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
print("Predicted prices for next 7 days:", future_predictions.flatten())