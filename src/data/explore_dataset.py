from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


DATASET_DIR = Path("data/intel_image_classification")
IMAGE_EXTENSIONS = {".jpg",  ".jpeg",  ".png"}


def contains_images(folder):

    return list(file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS  for file in folder.rglob("*"))



def find_class_directory():

    for directory in [DATASET_DIR, *DATASET_DIR.rglob("*")]:
        if not directory.is_dir():
            continue

        class_folders = [
            folder
            for folder in directory.iterdir()
            if folder.is_dir() and contains_images(folder)
        ]

        if len(class_folders) >= 3:
            return directory

    raise FileNotFoundError("Could not find a directory containing class folders.")
