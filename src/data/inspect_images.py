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


def inspect_image(image_path, class_name):
    file_size = image_path.stat().st_size

    result = {
        "path": str(image_path),
        "class": class_name,
        "extension": image_path.suffix.lower(),
        "format": None,
        "width": None,
        "height": None,
        "mode": None,
        "size_bytes": file_size,
        "sha256": None,
        "valid": False,
        "issues": "",
        "error": ""
    }

    issues = []

    if file_size == 0:
        issues.append("empty_file")

    try:
        with Image.open(image_path) as image:
            result["format"] = image.format
            result["width"], result["height"] = image.size
            result["mode"] = image.mode
            image.load()

        result["valid"] = True

    except Exception as error:
        issues.append("unreadable_image")
        result["error"] = str(error)

    expected_format = EXPECTED_FORMATS.get(
        result["extension"]
    )

    if (
        result["format"] is not None
        and expected_format is not None
        and result["format"] != expected_format
    ):
        issues.append("extension_format_mismatch")

    if result["valid"]:
        if result["width"] < 64 or result["height"] < 64:
            issues.append("small_dimensions")

        if file_size < 1024:
            issues.append("small_file_size")

        result["sha256"] = calculate_hash(image_path)

    result["issues"] = "; ".join(issues)

    return result

