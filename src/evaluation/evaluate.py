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

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true, y_pred, average="macro", zero_division=0
            )
        ),
        "precision_weighted": float(
            precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        )
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    pd.DataFrame(report).T.to_csv(REPORT_PATH, index=True)

    pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names
    ).to_csv(CONFUSION_PATH, index=True)

    misclassified_indices = np.flatnonzero(y_true != y_pred)

    misclassified = pd.DataFrame({
        "path": test_data.iloc[misclassified_indices]["path"].to_numpy(),
        "true_label": [
            class_names[y_true[index]]
            for index in misclassified_indices
        ],
        "predicted_label": [
            class_names[y_pred[index]]
            for index in misclassified_indices
        ],
        "confidence": [
            probabilities[index, y_pred[index]]
            for index in misclassified_indices
        ]
    })

    misclassified = misclassified.sort_values(
        "confidence",
        ascending=False
    )

    misclassified.to_csv(MISCLASSIFIED_PATH, index=False)

    confusion_rows = []

    for true_index in range(len(class_names)):
        for predicted_index in range(len(class_names)):
            count = int(matrix[true_index, predicted_index])

            if true_index != predicted_index and count > 0:
                confusion_rows.append({
                    "true_class": class_names[true_index],
                    "predicted_as": class_names[predicted_index],
                    "count": count
                })

    common_confusions = pd.DataFrame(
        confusion_rows,
        columns=["true_class", "predicted_as", "count"]
    )

    if not common_confusions.empty:
        common_confusions = common_confusions.sort_values(
            "count",
            ascending=False
        )

    common_confusions.to_csv(COMMON_CONFUSIONS_PATH, index=False)

    print("\nTest Metrics")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nPerformance by Class")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0
        )
    )

    print("\nMost Common Confusions")

    if common_confusions.empty:
        print("No class confusions found.")
    else:
        for row in common_confusions.head(10).itertuples(index=False):
            print(
                f"{row.true_class} predicted as "
                f"{row.predicted_as}: {row.count} images"
            )

    print(f"\nMetrics saved to: {METRICS_PATH}")
    print(f"Class report saved to: {REPORT_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_PATH}")
    print(f"Confusion plot saved to: {CONFUSION_PLOT_PATH}")
    print(f"Misclassified images saved to: {MISCLASSIFIED_PATH}")
    print(f"Misclassified images plot: {MISCLASSIFIED_PLOT_PATH}")


if __name__ == "__main__":
    evaluate_model()