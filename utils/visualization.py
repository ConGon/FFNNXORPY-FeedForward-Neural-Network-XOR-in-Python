# written by: ConGon
# Date: 2026-02-27

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss(loss_history, output_folder="output_graphs"):
    os.makedirs(output_folder, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "training_loss.png"))
    plt.close()


def plot_predictions(y_true, y_pred, output_folder="output_graphs"):
    os.makedirs(output_folder, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.scatter(range(len(y_true)), y_true, label="Actual")
    plt.scatter(range(len(y_pred)), y_pred, label="Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Output")
    plt.title("Final XOR Predictions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "final_predictions.png"))
    plt.close()


def plot_epoch_predictions(y_true, prediction_history, output_folder="output_graphs"):
    """
    Visualizes how predictions evolve over epochs.
    Each XOR sample gets its own prediction curve.
    """
    os.makedirs(output_folder, exist_ok=True)

    prediction_history = np.array(prediction_history)  # shape: (epochs, 4, 1)
    epochs = prediction_history.shape[0]

    plt.figure(figsize=(10, 6))

    for sample_index in range(len(y_true)):
        plt.plot(
            range(epochs),
            prediction_history[:, sample_index, 0],
            label=f"Sample {sample_index} (Target={y_true[sample_index][0]})"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Predicted Output")
    plt.title("Predicted Output vs Actual Across Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "epoch_predictions.png"))
    plt.close()