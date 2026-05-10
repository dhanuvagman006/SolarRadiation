import os
import sys
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils.plotter import save_all_plots

def main():
    model_name = "solar_lstm"
    data_path = os.path.join(BASE_DIR, "dakshina_kannada_solar_radiation_dataset.csv")
    model_dir = os.path.join(BASE_DIR, "Models", model_name)
    model_save_path = os.path.join(BASE_DIR, "Models", f"{model_name}.keras")
    plots_dir = os.path.join(BASE_DIR, f"{model_name}_plots")
    
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    features = ["T2M", "RH2M", "WS2M", "CLOUD_AMT", "PRECTOTCORR", "day", "month", "year"]
    target = "ALLSKY_SFC_SW_DWN"

    x = df[features].values
    y = df[target].values

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x = x_scaler.fit_transform(x)
    y = y_scaler.fit_transform(y.reshape(-1, 1))

    time_steps = 7
    future_steps = 7
    x_sequences, y_sequences = [], []
    for i in range(time_steps, len(x) - future_steps + 1):
        x_sequences.append(x[i - time_steps:i])
        y_sequences.append(y[i:i + future_steps].flatten())

    x_sequences, y_sequences = np.array(x_sequences), np.array(y_sequences)

    x_train, x_test, y_train, y_test = train_test_split(
        x_sequences, y_sequences, test_size=0.2, shuffle=False 
    )

    model = Sequential([
        LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2])),
        Dense(32, activation='relu'),
        Dense(future_steps)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        x_train, y_train, epochs=50, batch_size=32,
        validation_split=0.2, callbacks=[early_stop]
    )

    model.save(model_save_path)
    joblib.dump(x_scaler, os.path.join(model_dir, "x_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(model_dir, "y_scaler.pkl"))

    predictions = model.predict(x_test)
    predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1, future_steps)
    y_test_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, future_steps)

    save_all_plots(
        model_name=model_name,
        history=history,
        y_test_actual=y_test_actual,
        predictions=predictions,
        x_train=x_train,
        df=df,
        output_dir=plots_dir
    )

if __name__ == "__main__":
    main()