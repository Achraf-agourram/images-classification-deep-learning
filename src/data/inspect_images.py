from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image
from src.data.explore_dataset import find_class_directory, contains_images


DATASET_DIR = Path("data/intel_image_classification")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff"
}

EXPECTED_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".webp": "WEBP",
    ".tif": "TIFF",
    ".tiff": "TIFF"
}


def calculate_hash(image_path):
    digest = hashlib.sha256()

    with image_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()

