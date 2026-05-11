import os
import joblib
import numpy as np
from flask import Flask, render_template, request, send_from_directory, abort
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__)

FEATURES = ["T2M", "RH2M", "WS2M", "CLOUD_AMT", "PRECTOTCORR", "day", "month", "year"]

MODELS_CONFIG = {
    "solar_lstm": {
        "model_path": os.path.join(BASE_DIR, "Models", "solar_lstm.keras"),
        "x_scaler": os.path.join(BASE_DIR, "Models", "solar_lstm", "x_scaler.pkl"),
        "y_scaler": os.path.join(BASE_DIR, "Models", "solar_lstm", "y_scaler.pkl"),
        "plots_dir": os.path.join(BASE_DIR, "solar_lstm_plots"),
        "time_steps": 7,
        "future_steps": 7,
        "features": FEATURES,
    },
    "solar_gru": {
        "model_path": os.path.join(BASE_DIR, "Models", "solar_gru.keras"),
        "x_scaler": os.path.join(BASE_DIR, "Models", "solar_gru", "x_scaler.pkl"),
        "y_scaler": os.path.join(BASE_DIR, "Models", "solar_gru", "y_scaler.pkl"),
        "plots_dir": os.path.join(BASE_DIR, "solar_gru_plots"),
        "time_steps": 7,
        "future_steps": 7,
        "features": FEATURES,
    },
    "solar_bidirectional_lstm": {
        "model_path": os.path.join(BASE_DIR, "Models", "solar_bidirectional_lstm.keras"),
        "x_scaler": os.path.join(BASE_DIR, "Models", "solar_bidirectional_lstm", "x_scaler.pkl"),
        "y_scaler": os.path.join(BASE_DIR, "Models", "solar_bidirectional_lstm", "y_scaler.pkl"),
        "plots_dir": os.path.join(BASE_DIR, "solar_bidirectional_lstm_plots"),
        "time_steps": 7,
        "future_steps": 7,
        "features": FEATURES,
    },
    "solar_cnn_lstm": {
        "model_path": os.path.join(BASE_DIR, "Models", "solar_cnn_lstm.keras"),
        "x_scaler": os.path.join(BASE_DIR, "Models", "solar_cnn_lstm", "x_scaler.pkl"),
        "y_scaler": os.path.join(BASE_DIR, "Models", "solar_cnn_lstm", "y_scaler.pkl"),
        "plots_dir": os.path.join(BASE_DIR, "solar_cnn_lstm_plots"),
        "time_steps": 7,
        "future_steps": 7,
        "features": FEATURES,
    },
    "solar_stacked_lstm": {
        "model_path": os.path.join(BASE_DIR, "Models", "solar_stacked_lstm.keras"),
        "x_scaler": os.path.join(BASE_DIR, "Models", "solar_stacked_lstm", "x_scaler.pkl"),
        "y_scaler": os.path.join(BASE_DIR, "Models", "solar_stacked_lstm", "y_scaler.pkl"),
        "plots_dir": os.path.join(BASE_DIR, "solar_stacked_lstm_plots"),
        "time_steps": 7,
        "future_steps": 7,
        "features": FEATURES,
    },
    "solar_attention_lstm": {
        "model_path": os.path.join(BASE_DIR, "Models", "solar_attention_lstm.keras"),
        "x_scaler": os.path.join(BASE_DIR, "Models", "solar_attention_lstm", "x_scaler.pkl"),
        "y_scaler": os.path.join(BASE_DIR, "Models", "solar_attention_lstm", "y_scaler.pkl"),
        "plots_dir": os.path.join(BASE_DIR, "solar_attention_lstm_plots"),
        "time_steps": 7,
        "future_steps": 7,
        "features": FEATURES,
    },
}

class ModelManager:
    def __init__(self):
        self.models = {}

    def get_model(self, model_name):
        if model_name not in MODELS_CONFIG:
            return None
        
        if model_name not in self.models:
            config = MODELS_CONFIG[model_name]
            self.models[model_name] = {
                "model": load_model(config["model_path"]),
                "x_scaler": joblib.load(config["x_scaler"]),
                "y_scaler": joblib.load(config["y_scaler"])
            }
        return self.models[model_name]

model_manager = ModelManager()

@app.route("/")
def home():
    return render_template("home.html", models=MODELS_CONFIG.keys())

@app.route("/<model_name>", methods=["GET", "POST"])
def predict(model_name):
    if model_name not in MODELS_CONFIG:
        return "Model not found", 404

    config = MODELS_CONFIG[model_name]
    prediction = None
    error = None

    if request.method == "POST":
        try:
            input_data = []
            for i in range(config["time_steps"]):
                row = []
                for feature in config["features"]:
                    val_str = request.form.get(f"{feature}_{i}")
                    if val_str is None or val_str.strip() == "":
                        raise ValueError(f"Missing required field: {feature} for day {i+1}")
                    row.append(float(val_str))
                input_data.append(row)

            reshaped_input = model_manager.get_model(model_name)["x_scaler"].transform(input_data).reshape((1, config["time_steps"], len(config["features"])))
            pred = model_manager.get_model(model_name)["model"].predict(reshaped_input)
            inv_pred = model_manager.get_model(model_name)["y_scaler"].inverse_transform(pred.reshape(-1, 1)).reshape(1, config["future_steps"])
            
            prediction = [round(float(p), 4) for p in inv_pred[0]]
            
        except ValueError as e:
            error = f"Validation Error: {str(e)}"
        except Exception as e:
            error = f"Prediction Error: {str(e)}"

    plot_files = [f for f in os.listdir(config["plots_dir"]) if f.endswith(".png")] if os.path.exists(config["plots_dir"]) else []

    metrics = {}
    metrics_path = os.path.join(BASE_DIR, "BackendServer", "metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    city = request.args.get("city", "Dakshina Kannada Region")

    return render_template(
        "predict.html", model_name=model_name, available_models=list(MODELS_CONFIG.keys()), features=config["features"],
        time_steps=config["time_steps"], prediction=prediction, error=error, plot_files=plot_files,
        city=city, metrics=metrics
    )

@app.route("/plots/<model_name>/<filename>")
def serve_plot(model_name, filename):
    if model_name not in MODELS_CONFIG: return abort(404)
    return send_from_directory(MODELS_CONFIG[model_name]["plots_dir"], filename)

if __name__ == "__main__":
    app.run(port=5000)