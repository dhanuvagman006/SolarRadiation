import os
import sys

import numpy as np
import tensorflow as tf
from keras.layers import Attention, Bidirectional, Conv1D, Dense, Dropout, GlobalAveragePooling1D, GRU, Input, LSTM, MaxPooling1D

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from TrainModels.common import predict_example, train_model  # noqa: E402


SAMPLE_DATA = np.array([
    [24.71, 78.2, 1.99, 39.06, 0.46, 1, 1, 2015],
    [25.10, 76.5, 2.10, 42.00, 0.12, 2, 1, 2015],
    [24.85, 80.1, 1.75, 50.00, 0.30, 3, 1, 2015],
    [25.40, 79.0, 2.50, 35.00, 0.00, 4, 1, 2015],
    [26.00, 74.2, 3.10, 20.00, 0.00, 5, 1, 2015],
    [25.50, 77.8, 2.80, 28.00, 0.10, 6, 1, 2015],
    [24.90, 81.0, 1.95, 45.00, 0.50, 7, 1, 2015],
])


def build_lstm_model(input_shape, output_steps):
    inputs = Input(shape=input_shape)
    x = LSTM(64)(inputs)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(output_steps)(x)
    return tf.keras.Model(inputs, outputs, name="solar_lstm")


def build_gru_model(input_shape, output_steps):
    inputs = Input(shape=input_shape)
    x = GRU(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = GRU(32)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(output_steps)(x)
    return tf.keras.Model(inputs, outputs, name="solar_gru")


def build_bidirectional_lstm_model(input_shape, output_steps):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.25)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.15)(x)
    outputs = Dense(output_steps)(x)
    return tf.keras.Model(inputs, outputs, name="solar_bidirectional_lstm")


def build_cnn_lstm_model(input_shape, output_steps):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="causal")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(output_steps)(x)
    return tf.keras.Model(inputs, outputs, name="solar_cnn_lstm")


def build_stacked_lstm_model(input_shape, output_steps):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(16)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(output_steps)(x)
    return tf.keras.Model(inputs, outputs, name="solar_stacked_lstm")


def build_attention_lstm_model(input_shape, output_steps):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = Attention()([x, x])
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.15)(x)
    outputs = Dense(output_steps)(x)
    return tf.keras.Model(inputs, outputs, name="solar_attention_lstm")


MODEL_BUILDERS = [
    ("solar_lstm", build_lstm_model),
    ("solar_gru", build_gru_model),
    ("solar_bidirectional_lstm", build_bidirectional_lstm_model),
    ("solar_cnn_lstm", build_cnn_lstm_model),
    ("solar_stacked_lstm", build_stacked_lstm_model),
    ("solar_attention_lstm", build_attention_lstm_model),
]

TARGET_ACCURACY_FLOORS = {
    "solar_lstm": 81.50,
    "solar_gru": 82.25,
    "solar_bidirectional_lstm": 83.00,
    "solar_cnn_lstm": 81.75,
    "solar_stacked_lstm": 83.50,
    "solar_attention_lstm": 84.25,
}


def main():
    data_path = os.path.join(BASE_DIR, "TrainModels", "processed_data.csv")
    results = []

    for model_name, build_model_fn in MODEL_BUILDERS:
        print(f"\nTraining {model_name}...")
        target_accuracy = TARGET_ACCURACY_FLOORS.get(model_name, 82.0)
        history, y_true, y_pred = train_model(
            BASE_DIR,
            model_name,
            build_model_fn,
            data_path,
            epochs=150,
            patience=15,
            target_accuracy=target_accuracy,
            fine_tune_epochs=75,
        )
        prediction_example = predict_example(BASE_DIR, model_name, SAMPLE_DATA)

        results.append({
            "model_name": model_name,
            "epochs": len(history.history["loss"]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "prediction_example": np.round(prediction_example.reshape(-1), 4).tolist(),
            "plots_dir": os.path.join(BASE_DIR, f"{model_name}_plots"),
        })

        print(f"Finished {model_name}")
        print(f"Prediction example: {np.round(prediction_example.reshape(-1), 4)}")

    print("\nTraining summary:")
    for item in results:
        print(
            f"{item['model_name']}: epochs={item['epochs']}, val_loss={item['final_val_loss']:.6f}, plots={item['plots_dir']}"
        )


if __name__ == "__main__":
    main()