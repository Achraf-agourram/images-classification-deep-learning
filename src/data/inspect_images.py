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


def inspect_dataset():
    class_directory = find_class_directory()
    records = []

    for class_folder in sorted(class_directory.iterdir()):
        if not class_folder.is_dir():
            continue

        if not contains_images(class_folder):
            continue

        image_paths = [
            path
            for path in class_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image_path in image_paths:
            records.append(
                inspect_image(image_path, class_folder.name)
            )

    if not records:
        raise ValueError("No images were found.")

    audit_df = pd.DataFrame(records)

    valid_df = audit_df[audit_df["valid"]].copy()

    duplicate_hashes = set(
        valid_df["sha256"][
            valid_df["sha256"].duplicated(keep=False)
        ].dropna()
    )

    duplicate_mask = audit_df["sha256"].isin(duplicate_hashes)

    for index in audit_df.index[duplicate_mask]:
        current_issues = audit_df.at[index, "issues"]

        if current_issues:
            audit_df.at[index, "issues"] = (
                current_issues + "; exact_duplicate"
            )
        else:
            audit_df.at[index, "issues"] = "exact_duplicate"

    reports_dir = Path("reports/metrics")
    reports_dir.mkdir(parents=True, exist_ok=True)

    audit_path = reports_dir / "image_audit.csv"
    problems_path = reports_dir / "problematic_images.csv"

    audit_df.to_csv(audit_path, index=False)

    problematic_df = audit_df[
        audit_df["issues"] != ""
    ].copy()

    problematic_df.to_csv(problems_path, index=False)

    print(f"Class directory: {class_directory}")
    print(f"Total images inspected: {len(audit_df)}")
    print(f"Readable images: {len(valid_df)}")
    print(f"Unreadable images: {(~audit_df['valid']).sum()}")

    print("\nImage formats:")
    print(
        valid_df["format"]
        .value_counts()
        .to_string()
    )

    print("\nImage dimensions:")
    if not valid_df.empty:
        dimensions = (
            valid_df["width"].astype(int).astype(str)
            + " x "
            + valid_df["height"].astype(int).astype(str)
        )

        print(dimensions.value_counts().head(15).to_string())

        print("\nDimension summary:")
        print(
            valid_df[["width", "height"]]
            .describe()
            .round(2)
            .to_string()
        )

    print("\nImage modes:")
    print(valid_df["mode"].value_counts().to_string())

    print("\nData quality issues:")

    issue_names = [
        "empty_file",
        "unreadable_image",
        "extension_format_mismatch",
        "small_dimensions",
        "small_file_size",
        "exact_duplicate"
    ]

    for issue in issue_names:
        count = audit_df["issues"].apply(
            lambda value: issue in value.split("; ")
        ).sum()

        print(f"{issue}: {count}")

    print(f"\nFull report: {audit_path}")
    print(f"Problematic images: {problems_path}")

    return audit_df


if __name__ == "__main__":
    inspect_dataset()