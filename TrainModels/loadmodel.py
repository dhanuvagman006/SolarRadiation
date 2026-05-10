import os
from tensorflow.keras.models import load_model
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_name = "solar_lstm"

model = load_model(os.path.join(BASE_DIR, "Models", f"{model_name}.keras"))

x_scaler = joblib.load(os.path.join(BASE_DIR, "Models", model_name, "x_scaler.pkl"))
y_scaler = joblib.load(os.path.join(BASE_DIR, "Models", model_name, "y_scaler.pkl"))

sample_data = np.array([
    [24.71, 78.2, 1.99, 39.06, 0.46, 1, 1, 2015],
    [25.10, 76.5, 2.10, 42.00, 0.12, 2, 1, 2015],
    [24.85, 80.1, 1.75, 50.00, 0.30, 3, 1, 2015],
    [25.40, 79.0, 2.50, 35.00, 0.00, 4, 1, 2015],
    [26.00, 74.2, 3.10, 20.00, 0.00, 5, 1, 2015],
    [25.50, 77.8, 2.80, 28.00, 0.10, 6, 1, 2015],
    [24.90, 81.0, 1.95, 45.00, 0.50, 7, 1, 2015]
])

sample_data = x_scaler.transform(sample_data)
sample_data = sample_data.reshape((1, 7, 8))

prediction = model.predict(sample_data)
prediction = y_scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(1, 7)

print("Predicted Solar Radiation for Next 7 Days:", prediction[0])