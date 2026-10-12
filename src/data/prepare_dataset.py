from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


DATASET_DIR = Path("data/intel_image_classification")
OUTPUT_DIR = Path("data/splits")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif"
}

EXPECTED_CLASSES = {
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
}

RANDOM_STATE = 42