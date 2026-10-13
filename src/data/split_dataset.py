from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.explore_dataset import find_class_directory, contains_images

DATASET_DIR = Path("data/intel_image_classification")
OUTPUT_DIR = Path("data/splits")
AUDIT_PATH = Path("reports/metrics/image_audit.csv")

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

RANDOM_STATE = 42

def collect_images(class_directory):
    records = []

    for class_folder in sorted(class_directory.iterdir()):
        if not class_folder.is_dir() or not contains_images(class_folder):
            continue

        for image_path in sorted(class_folder.rglob("*")):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                records.append({
                    "path": str(image_path),
                    "label": class_folder.name
                })

    return pd.DataFrame(records)
