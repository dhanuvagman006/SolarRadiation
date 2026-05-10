import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_all_plots(
    model_name,
    history,
    y_test_actual,
    predictions,
    x_train,
    df,
    output_dir=None
):
    y_test_actual = np.array(y_test_actual).flatten()
    predictions = np.array(predictions).flatten()
    folder_name = output_dir if output_dir else f"{model_name}_plots"

    os.makedirs(folder_name, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{folder_name}/loss_plot.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.savefig(f"{folder_name}/mae_plot.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual[:200], label='Actual')
    plt.plot(predictions[:200], label='Predicted')
    plt.xlabel("Samples")
    plt.ylabel("Solar Radiation")
    plt.legend()
    plt.savefig(f"{folder_name}/actual_vs_predicted.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_actual, predictions)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig(f"{folder_name}/scatter_plot.png")
    plt.close()

    residuals = y_test_actual - predictions

    plt.figure(figsize=(10, 5))
    plt.plot(residuals)
    plt.xlabel("Samples")
    plt.ylabel("Residual Error")
    plt.savefig(f"{folder_name}/residual_plot.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual Error")
    plt.ylabel("Frequency")
    plt.savefig(f"{folder_name}/residual_histogram.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.savefig(f"{folder_name}/correlation_heatmap.png")
    plt.close()

    sample = x_train[0]

    plt.figure(figsize=(10, 5))

    for i in range(sample.shape[1]):
        plt.plot(sample[:, i], label=f'Feature {i}')

    plt.legend()
    plt.savefig(f"{folder_name}/sequence_window.png")
    plt.close()

    errors = np.abs(y_test_actual - predictions)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_actual, errors)
    plt.xlabel("Actual")
    plt.ylabel("Absolute Error")
    plt.savefig(f"{folder_name}/error_vs_actual.png")
    plt.close()

    print(f"All plots saved in '{folder_name}'")