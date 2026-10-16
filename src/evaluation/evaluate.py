from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.data.prepare_dataset import create_datasets, SPLIT_DIR, IMAGE_SIZE

MODEL_PATH = Path("models/cnn.keras")

METRICS_PATH = Path("reports/metrics/test_metrics.json")
REPORT_PATH = Path("reports/metrics/classification_report.csv")
CONFUSION_PATH = Path("reports/metrics/confusion_matrix.csv")
COMMON_CONFUSIONS_PATH = Path("reports/metrics/common_confusions.csv")
MISCLASSIFIED_PATH = Path("reports/metrics/misclassified_images.csv")

CONFUSION_PLOT_PATH = Path("reports/figures/confusion_matrix.png")
MISCLASSIFIED_PLOT_PATH = Path("reports/figures/misclassified_images.png")

def evaluate_model():
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train the model first.")

    _, _, test_dataset, class_names = create_datasets()

    model = tf.keras.models.load_model(MODEL_PATH)

    test_data = pd.read_csv(SPLIT_DIR / "test.csv")

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    true_labels = test_data["label"].map(class_to_index)

    if true_labels.isna().any():
        raise ValueError("Unknown labels found in the test dataset.")

    y_true = true_labels.astype("int32").to_numpy()

    probabilities = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "The number of test labels does not match the predictions."
        )

    labels = np.arange(len(class_names))


if __name__ == "__main__":
    evaluate_model()