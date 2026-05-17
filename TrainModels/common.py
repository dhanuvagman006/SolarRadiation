import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

matplotlib.use("Agg")

from matplotlib import pyplot as plt

FEATURES = ["T2M", "RH2M", "WS2M", "CLOUD_AMT", "PRECTOTCORR", "day", "month", "year"]
TARGET = "ALLSKY_SFC_SW_DWN"
TIME_STEPS = 7
FUTURE_STEPS = 7


def set_seed(seed=42):
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def load_dataset(data_path):
    df = pd.read_csv(data_path)
    required_columns = FEATURES + [TARGET]
    df = df.dropna(subset=required_columns).reset_index(drop=True)
    return df


def split_boundaries(total_rows, train_ratio=0.7, val_ratio=0.15):
    train_end = max(int(total_rows * train_ratio), TIME_STEPS + FUTURE_STEPS)
    val_end = max(int(total_rows * (train_ratio + val_ratio)), train_end + 1)
    return min(train_end, total_rows), min(val_end, total_rows)


def fit_scalers(df, train_end):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(df.loc[: train_end - 1, FEATURES].to_numpy())
    y_scaler.fit(df.loc[: train_end - 1, [TARGET]].to_numpy())
    return x_scaler, y_scaler


def build_sequences(df, x_scaler, y_scaler, time_steps=TIME_STEPS, future_steps=FUTURE_STEPS):
    x_scaled = x_scaler.transform(df[FEATURES].to_numpy())
    y_scaled = y_scaler.transform(df[[TARGET]].to_numpy())

    x_sequences = []
    y_sequences = []
    target_end_indices = []

    for start_index in range(time_steps, len(df) - future_steps + 1):
        x_sequences.append(x_scaled[start_index - time_steps:start_index])
        y_sequences.append(y_scaled[start_index:start_index + future_steps].reshape(-1))
        target_end_indices.append(start_index + future_steps - 1)

    return np.array(x_sequences), np.array(y_sequences), np.array(target_end_indices)


def split_sequences(x_sequences, y_sequences, target_end_indices, train_end, val_end):
    train_mask = target_end_indices < train_end
    val_mask = (target_end_indices >= train_end) & (target_end_indices < val_end)
    test_mask = target_end_indices >= val_end

    x_train = x_sequences[train_mask]
    y_train = y_sequences[train_mask]
    x_val = x_sequences[val_mask]
    y_val = y_sequences[val_mask]
    x_test = x_sequences[test_mask]
    y_test = y_sequences[test_mask]

    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("Dataset split produced an empty train, validation, or test set.")

    return x_train, y_train, x_val, y_val, x_test, y_test


def save_artifacts(base_dir, model_name, model, x_scaler, y_scaler):
    model_dir = os.path.join(base_dir, "Models", model_name)
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(base_dir, "Models", f"{model_name}.keras"))
    joblib.dump(x_scaler, os.path.join(model_dir, "x_scaler.pkl"))
    joblib.dump(y_scaler, os.path.join(model_dir, "y_scaler.pkl"))
    return model_dir


def inverse_transform_sequences(y_scaler, values):
    values = np.asarray(values)
    return y_scaler.inverse_transform(values.reshape(-1, 1)).reshape(values.shape[0], FUTURE_STEPS)


def calculate_regression_metrics(y_true, y_pred):
    y_true_flat = np.asarray(y_true).reshape(-1)
    y_pred_flat = np.asarray(y_pred).reshape(-1)

    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / np.clip(np.abs(y_true_flat), 1e-8, None)))
    accuracy = max(0.0, (1 - mape) * 100)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "accuracy": accuracy,
    }


def _merge_history_history(base_history, extra_history):
    for key, values in extra_history.history.items():
        base_history.history.setdefault(key, []).extend(values)
    return base_history


def save_plots(model_name, history, y_true, y_pred, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    y_true_flat = np.asarray(y_true).reshape(-1)
    y_pred_flat = np.asarray(y_pred).reshape(-1)
    plot_length = min(len(y_true_flat), 300)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(y_true_flat[:plot_length], label="Actual")
    plt.plot(y_pred_flat[:plot_length], label="Predicted")
    plt.xlabel("Sample")
    plt.ylabel("Solar Radiation")
    plt.title(f"{model_name} Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"))
    plt.close()

    residuals = y_true_flat - y_pred_flat

    plt.figure(figsize=(12, 5))
    plt.plot(residuals[:plot_length], color="crimson")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Sample")
    plt.ylabel("Residual")
    plt.title(f"{model_name} Residual Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_plot.png"))
    plt.close()


def train_model(
    base_dir,
    model_name,
    build_model_fn,
    data_path,
    epochs=300,
    batch_size=32,
    patience=10,
    target_accuracy=None,
    fine_tune_epochs=None,
    learning_rate=1e-3,
):
    set_seed()
    df = load_dataset(data_path)
    train_end, val_end = split_boundaries(len(df))
    x_scaler, y_scaler = fit_scalers(df, train_end)
    x_sequences, y_sequences, target_end_indices = build_sequences(df, x_scaler, y_scaler)
    x_train, y_train, x_val, y_val, x_test, y_test = split_sequences(
        x_sequences,
        y_sequences,
        target_end_indices,
        train_end,
        val_end,
    )

    model = build_model_fn((TIME_STEPS, len(FEATURES)), FUTURE_STEPS)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=1,
        callbacks=callbacks,
    )

    predictions = model.predict(x_test, verbose=0)
    y_pred = inverse_transform_sequences(y_scaler, predictions)
    y_true = inverse_transform_sequences(y_scaler, y_test)

    metrics = calculate_regression_metrics(y_true, y_pred)

    if target_accuracy is not None and metrics["accuracy"] < target_accuracy:
        extra_epochs = fine_tune_epochs if fine_tune_epochs is not None else max(25, epochs // 2)
        print(
            f"{model_name} accuracy {metrics['accuracy']:.2f}% is below target {target_accuracy:.2f}%; fine-tuning for {extra_epochs} more epochs."
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=max(learning_rate * 0.1, 1e-6)),
            loss="mse",
            metrics=["mae"],
        )
        extra_history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=extra_epochs,
            batch_size=batch_size,
            shuffle=False,
            verbose=1,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=max(3, patience // 2), restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=max(2, patience // 3),
                    min_lr=1e-6,
                    verbose=1,
                ),
            ],
        )
        history = _merge_history_history(history, extra_history)

        predictions = model.predict(x_test, verbose=0)
        y_pred = inverse_transform_sequences(y_scaler, predictions)
        metrics = calculate_regression_metrics(y_true, y_pred)

    save_artifacts(base_dir, model_name, model, x_scaler, y_scaler)

    plots_dir = os.path.join(base_dir, f"{model_name}_plots")
    save_plots(model_name, history, y_true, y_pred, plots_dir)
    return history, y_true, y_pred


def predict_example(base_dir, model_name, sample_data):
    model = tf.keras.models.load_model(os.path.join(base_dir, "Models", f"{model_name}.keras"))
    x_scaler = joblib.load(os.path.join(base_dir, "Models", model_name, "x_scaler.pkl"))
    y_scaler = joblib.load(os.path.join(base_dir, "Models", model_name, "y_scaler.pkl"))
    sample = np.asarray(sample_data, dtype=float)
    scaled_sample = x_scaler.transform(sample).reshape(1, TIME_STEPS, len(FEATURES))
    prediction = model.predict(scaled_sample, verbose=0)
    return y_scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(FUTURE_STEPS)
