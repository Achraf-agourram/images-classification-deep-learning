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

