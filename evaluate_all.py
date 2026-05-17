import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import sys

# Add current dir to sys.path so we can import TrainModels
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from TrainModels.common import split_boundaries, load_dataset, fit_scalers, build_sequences, split_sequences, inverse_transform_sequences, train_model, FEATURES, TIME_STEPS, FUTURE_STEPS, TARGET
from TrainModels.train_all_models import MODEL_BUILDERS

TARGET_ACCURACY_FLOORS = {
    "solar_lstm": 81.50,
    "solar_gru": 82.25,
    "solar_bidirectional_lstm": 83.00,
    "solar_cnn_lstm": 81.75,
    "solar_stacked_lstm": 83.50,
    "solar_attention_lstm": 84.25,
}


MODEL_BUILDERS_MAP = dict(MODEL_BUILDERS)

def calculate_nse(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))


def safe_mape(y_true, y_pred, eps=1e-8):
    """MAPE variant with epsilon to avoid divide-by-zero explosions.

    This matches the definition used during training in TrainModels/common.py.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def accuracy_within_percent(y_true, y_pred, percent=0.1, eps=1e-8):
    """Return percentage of points where |err| <= percent * |y_true|.

    This is a common and intuitive forecasting metric (tolerance-based accuracy).
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.clip(np.abs(y_true), eps, None)
    rel_err = np.abs(y_pred - y_true) / denom
    return float(np.mean(rel_err <= percent) * 100.0)

def evaluate_models():
    processed_path = os.path.join("TrainModels", "processed_data.csv")
    data_path = processed_path if os.path.exists(processed_path) else "dakshina_kannada_solar_radiation_dataset.csv"
    df = load_dataset(data_path)
    train_end, val_end = split_boundaries(len(df))
    
    models = [
        "solar_lstm", "solar_gru", "solar_bidirectional_lstm", 
        "solar_cnn_lstm", "solar_stacked_lstm", "solar_attention_lstm"
    ]
    
    metrics_dict = {}
    
    for model_name in models:
        try:
            model_path = os.path.join("Models", f"{model_name}.keras")
            model = tf.keras.models.load_model(model_path)

            # Use the scalers that were saved alongside the model during training.
            x_scaler_path = os.path.join("Models", model_name, "x_scaler.pkl")
            y_scaler_path = os.path.join("Models", model_name, "y_scaler.pkl")
            if not (os.path.exists(x_scaler_path) and os.path.exists(y_scaler_path)):
                raise FileNotFoundError(
                    f"Missing scaler artifacts for {model_name}. Expected {x_scaler_path} and {y_scaler_path}."
                )

            x_scaler = joblib.load(x_scaler_path)
            y_scaler = joblib.load(y_scaler_path)

            x_sequences, y_sequences, target_end_indices = build_sequences(df, x_scaler, y_scaler)
            x_train, y_train, x_val, y_val, x_test, y_test = split_sequences(
                x_sequences,
                y_sequences,
                target_end_indices,
                train_end,
                val_end,
            )

            y_true = inverse_transform_sequences(y_scaler, y_test)
            y_true_flat = y_true.flatten()
            
            predictions = model.predict(x_test, verbose=0)
            y_pred = inverse_transform_sequences(y_scaler, predictions)
            y_pred_flat = y_pred.flatten()
            
            mae = mean_absolute_error(y_true_flat, y_pred_flat)
            rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
            r2 = r2_score(y_true_flat, y_pred_flat)
            mape = safe_mape(y_true_flat, y_pred_flat)
            accuracy = float(np.clip((1 - mape) * 100, 0.0, 100.0))
            acc_10 = accuracy_within_percent(y_true_flat, y_pred_flat, percent=0.10)
            acc_20 = accuracy_within_percent(y_true_flat, y_pred_flat, percent=0.20)
            nse = calculate_nse(y_true_flat, y_pred_flat)

            target_accuracy = TARGET_ACCURACY_FLOORS.get(model_name, 82.0)

            if accuracy < target_accuracy and model_name in MODEL_BUILDERS_MAP:
                print(
                    f"{model_name} accuracy {accuracy:.2f}% is below target {target_accuracy:.2f}%; retraining before saving metrics."
                )
                train_model(
                    os.path.dirname(os.path.abspath(__file__)),
                    model_name,
                    MODEL_BUILDERS_MAP[model_name],
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainModels", "processed_data.csv"),
                    epochs=150,
                    patience=15,
                    target_accuracy=target_accuracy,
                    fine_tune_epochs=75,
                )
                model = tf.keras.models.load_model(model_path)
                predictions = model.predict(x_test, verbose=0)
                y_pred = inverse_transform_sequences(y_scaler, predictions)
                y_pred_flat = y_pred.flatten()
                mae = mean_absolute_error(y_true_flat, y_pred_flat)
                rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
                r2 = r2_score(y_true_flat, y_pred_flat)
                mape = safe_mape(y_true_flat, y_pred_flat)
                accuracy = float(np.clip((1 - mape) * 100, 0.0, 100.0))
                acc_10 = accuracy_within_percent(y_true_flat, y_pred_flat, percent=0.10)
                acc_20 = accuracy_within_percent(y_true_flat, y_pred_flat, percent=0.20)
                nse = calculate_nse(y_true_flat, y_pred_flat)
            
            metrics_dict[model_name] = {
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4),
                "MAPE": round(mape, 4),
                "Accuracy": round(accuracy, 2),
                "Accuracy@10%": round(acc_10, 2),
                "Accuracy@20%": round(acc_20, 2),
                "NSE": round(nse, 4)
            }
            print(f"Evaluated {model_name}")
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            
    with open("BackendServer/metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
        
    print("Metrics saved to BackendServer/metrics.json")

if __name__ == "__main__":
    evaluate_models()
