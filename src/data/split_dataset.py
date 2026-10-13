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


def remove_unreadable_images(data):
    if not AUDIT_PATH.exists():
        print("Image audit not found. Skipping the readability filter.")
        return data

    audit = pd.read_csv(AUDIT_PATH)

    if not {"path", "valid"}.issubset(audit.columns):
        print("Image audit has an unexpected format. Skipping the filter.")
        return data

    valid_status = (
        audit["valid"]
        .astype(str)
        .str.lower()
        .isin({"true", "1"})
    )

    invalid_paths = set(
        audit.loc[~valid_status, "path"].astype(str)
    )

    filtered_data = data[~data["path"].isin(invalid_paths)].copy()

    print(f"Unreadable images excluded: {len(data) - len(filtered_data)}")

    return filtered_data


def save_split(data, split_name):

    for row in data.itertuples(index=False):

        source = Path(row.path)
        destination_folder = OUTPUT_DIR / split_name / row.label
        destination_folder.mkdir(parents=True, exist_ok=True)

        destination = destination_folder / source.name

        if destination.exists():
            raise FileExistsError(f"Two images have the same filename: {destination}")

        shutil.copy2(source, destination)


def split_dataset():

    class_directory = find_class_directory()
    data = collect_images(class_directory)

    if data.empty:
        raise ValueError("No labeled images were found.")

    data = remove_unreadable_images(data)

    if data.empty:
        raise ValueError("No valid images remain.")

    class_counts = data["label"].value_counts()

    if (class_counts < 3).any():
        raise ValueError("Every class must contain at least three valid images.")

    train_data, temporary_data = train_test_split(
        data,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=data["label"]
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temporary_data["label"]
    )

    splits = {
        "train": train_data,
        "validation": validation_data,
        "test": test_data
    }

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split_name, split_data in splits.items():
        save_split(split_data, split_name)

    counts = pd.crosstab(data["label"], data.index.map(
        lambda index: (
            "train"
            if index in train_data.index
            else "validation"
            if index in validation_data.index
            else "test"
        )
    ))

    counts = counts.reindex(columns=["train", "validation", "test"], fill_value=0)

    distribution = counts.copy()

    for split_name, split_data in splits.items():
        distribution[f"{split_name}_percent"] = (counts[split_name] / len(split_data) * 100).round(2)

    distribution["total"] = counts.sum(axis=1)

    total_images = sum(len(split_data) for split_data in splits.values())

    print(f"Source directory: {class_directory}")
    print(f"Total labeled images: {total_images}")

    print("\nSplit summary:")

    for split_name, split_data in splits.items():
        percentage = len(split_data) / total_images * 100

        print(
            f"{split_name}: {len(split_data)} images "
            f"({percentage:.2f}%)"
        )

    print("\nClass distribution:")
    print(distribution.to_string())

    print(f"\nSplits saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    split_dataset()