from pathlib import Path
import pandas as pd
import tensorflow as tf

from src.data.prepare_dataset import create_datasets
from src.models.cnn import build_cnn


EPOCHS = 30

MODEL_PATH = Path("models/cnn.keras")
HISTORY_PATH = Path("reports/metrics/training_history.csv")
PLOT_PATH = Path("reports/figures/training_curves.png")

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



if __name__ == "__main__":
    train_model()
