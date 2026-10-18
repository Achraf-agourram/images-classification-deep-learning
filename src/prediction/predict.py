import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.prepare_dataset import (
    load_and_preprocess_image,
    SPLIT_DIR
)


MODEL_PATH = Path("models/cnn.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")


def load_prediction_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Train the model first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            class_names = json.load(file)
    else:
        train_data = pd.read_csv(SPLIT_DIR / "train.csv")
        class_names = sorted(train_data["label"].unique())

    return model, class_names


def predict_image(image_path, model, class_names):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image, _ = load_and_preprocess_image(
        str(image_path),
        0
    )

    image_batch = tf.expand_dims(image, axis=0)

    probabilities = model.predict(
        image_batch,
        verbose=0
    )[0]

    if len(probabilities) != len(class_names):
        raise ValueError(
            "The number of model outputs does not match the class names."
        )

    predicted_index = int(np.argmax(probabilities))

    predicted_class = class_names[predicted_index]

    probability = float(probabilities[predicted_index])

    print(f"Predicted class: {predicted_class}")
    print(f"Probability: {probability:.2%}")

    plt.figure(figsize=(6, 5))
    plt.imshow(image.numpy())
    plt.title(
        f"Prediction: {predicted_class}\n"
        f"Probability: {probability:.2%}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return {
        "image_path": str(image_path),
        "predicted_class": predicted_class,
        "probability": probability
    }


