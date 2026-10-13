from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.explore_dataset import find_class_directory, contains_images

DATASET_DIR = Path("data/intel_image_classification")
OUTPUT_DIR = Path("data/splits")
AUDIT_PATH = Path("reports/metrics/image_audit.csv")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

RANDOM_STATE = 42


def collect_images():
    class_directory = find_class_directory()
    records = []

    for class_folder in class_directory.iterdir():
        if not class_folder.is_dir():
            continue

        for image_path in class_folder.rglob("*"):
            if (image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS):
                records.append({
                    "path": image_path.as_posix(),
                    "label": class_folder.name
                })

    data = pd.DataFrame(records)

    if data.empty:
        raise ValueError("No images were found in the dataset.")

    return data


def remove_unreadable_images(data):
    if not AUDIT_PATH.exists():
        return data

    audit = pd.read_csv(AUDIT_PATH)

    if not {"path", "valid"}.issubset(audit.columns):
        return data

    data = data.copy()
    audit = audit.copy()

    data["path"] = data["path"].astype(str).str.replace("\\", "/", regex=False)
    audit["path"] = audit["path"].astype(str).str.replace("\\", "/", regex=False)

    valid_values = audit["valid"].astype(str).str.lower().eq("true")
    invalid_paths = set(audit.loc[~valid_values, "path"])

    initial_count = len(data)
    data = data[~data["path"].isin(invalid_paths)].copy()

    print(f"Unreadable images removed: {initial_count - len(data)}")

    return data


def create_splits(data):
    X = data["path"]
    y = data["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    train_data = pd.DataFrame({
        "path": X_train.to_numpy(),
        "label": y_train.to_numpy()
    })

    validation_data = pd.DataFrame({
        "path": X_validation.to_numpy(),
        "label": y_validation.to_numpy()
    })

    test_data = pd.DataFrame({
        "path": X_test.to_numpy(),
        "label": y_test.to_numpy()
    })

    return train_data, validation_data, test_data


def save_splits(train_data, validation_data, test_data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "validation", "test"]:
        old_directory = OUTPUT_DIR / split_name

        if old_directory.is_dir():
            shutil.rmtree(old_directory)

    splits = {
        "train": train_data,
        "validation": validation_data,
        "test": test_data
    }

    for split_name, split_data in splits.items():
        output_path = OUTPUT_DIR / f"{split_name}.csv"
        split_data.to_csv(output_path, index=False)

    distribution = pd.concat(
        [
            split_data[["label"]].assign(split=split_name)
            for split_name, split_data in splits.items()
        ],
        ignore_index=True
    )

    counts = pd.crosstab(
        distribution["label"],
        distribution["split"]
    )

    counts = counts.reindex(
        columns=["train", "validation", "test"],
        fill_value=0
    )

    counts["total"] = counts[["train", "validation", "test"]].sum(axis=1)

    for split_name in ["train", "validation", "test"]:
        counts[f"{split_name}_percentage"] = (
            counts[split_name] / counts["total"] * 100
        ).round(2)

    counts.to_csv(OUTPUT_DIR / "class_distribution.csv")

    print("\nSplit results:")

    for split_name, split_data in splits.items():
        percentage = len(split_data) / len(distribution) * 100
        print(
            f"{split_name}: {len(split_data)} images "
            f"({percentage:.2f}%)"
        )

    print("\nClass distribution:")
    print(counts)


def main():
    data = collect_images()
    data = remove_unreadable_images(data)

    if data.empty:
        raise ValueError("No valid images remain after filtering.")

    print(f"Total valid images: {len(data)}")
    print(f"Number of classes: {data['label'].nunique()}")

    train_data, validation_data, test_data = create_splits(data)

    save_splits(train_data, validation_data, test_data)


if __name__ == "__main__":
    main()