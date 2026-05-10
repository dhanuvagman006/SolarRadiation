import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GRU, Input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from TrainModels.common import predict_example, train_model  # noqa: E402

MODEL_NAME = "solar_gru"


def build_model(input_shape, output_steps):
    inputs = Input(shape=input_shape)
    x = GRU(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = GRU(32)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(output_steps)(x)
    return tf.keras.Model(inputs, outputs, name=MODEL_NAME)


def main():
    data_path = os.path.join(BASE_DIR, "dakshina_kannada_solar_radiation_dataset.csv")
    train_model(BASE_DIR, MODEL_NAME, build_model, data_path)
    sample_data = np.array([
        [24.71, 78.2, 1.99, 39.06, 0.46, 1, 1, 2015],
        [25.10, 76.5, 2.10, 42.00, 0.12, 2, 1, 2015],
        [24.85, 80.1, 1.75, 50.00, 0.30, 3, 1, 2015],
        [25.40, 79.0, 2.50, 35.00, 0.00, 4, 1, 2015],
        [26.00, 74.2, 3.10, 20.00, 0.00, 5, 1, 2015],
        [25.50, 77.8, 2.80, 28.00, 0.10, 6, 1, 2015],
        [24.90, 81.0, 1.95, 45.00, 0.50, 7, 1, 2015],
    ])
    prediction = predict_example(BASE_DIR, MODEL_NAME, sample_data)
    print(f"{MODEL_NAME} prediction example:")
    print(np.round(prediction.reshape(-1), 4))


if __name__ == "__main__":
    main()
