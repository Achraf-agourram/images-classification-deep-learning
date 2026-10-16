from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.data.prepare_dataset import create_datasets
from src.models.cnn import build_cnn


EPOCHS = 30

MODEL_PATH = Path("models/cnn.keras")
HISTORY_PATH = Path("reports/metrics/training_history.csv")
PLOT_PATH = Path("reports/figures/training_curves.png")


def plot_training_history(history):
    history_data = pd.DataFrame(history.history)
    history_data.index = range(1, len(history_data) + 1)

    epochs = history_data.index

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        epochs,
        history_data["accuracy"],
        label="Training accuracy"
    )
    axes[0].plot(
        epochs,
        history_data["val_accuracy"],
        label="Validation accuracy"
    )
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        epochs,
        history_data["loss"],
        label="Training loss"
    )
    axes[1].plot(
        epochs,
        history_data["val_loss"],
        label="Validation loss"
    )
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150)
    plt.show()
    plt.close(fig)


def train_model():
    train_ds, validation_ds, _, class_names = create_datasets()

    model = build_cnn(num_classes=len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    model.save(MODEL_PATH)

    history_data = pd.DataFrame(history.history)
    history_data.index = range(1, len(history_data) + 1)
    history_data.to_csv(HISTORY_PATH, index_label="epoch")

    plot_training_history(history)

    best_loss_epoch = int(
        history_data["val_loss"].idxmin()
    )

    best_accuracy_epoch = int(
        history_data["val_accuracy"].idxmax()
    )

    print(f"Classes: {class_names}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training history saved to: {HISTORY_PATH}")
    print(f"Training curves saved to: {PLOT_PATH}")

    print(f"Best validation loss: {history_data['val_loss'].min():.4f}")
    print(f"Epoch with best validation loss: {best_loss_epoch}")

    print(
        f"Best validation accuracy: "
        f"{history_data['val_accuracy'].max():.4f}"
    )
    print(f"Epoch with best validation accuracy: {best_accuracy_epoch}")


if __name__ == "__main__":
    train_model()