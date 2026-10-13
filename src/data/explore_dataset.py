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


def explore_dataset():

    class_directory = find_class_directory()
    class_counts = {}

    for class_folder in sorted(class_directory.iterdir()):
        if not class_folder.is_dir() or not contains_images(class_folder):
            continue

        image_count = sum(
            1
            for file in class_folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        class_counts[class_folder.name] = image_count

    if not class_counts:
        raise ValueError("No images were found in the dataset.")

    total_images = sum(class_counts.values())

    distribution = pd.DataFrame({
        "class": list(class_counts.keys()),
        "image_count": list(class_counts.values())
    })

    distribution["percentage"] = (distribution["image_count"] / total_images * 100).round(2)

    distribution = distribution.sort_values("image_count", ascending=False).reset_index(drop=True)

    print(f"Dataset directory: {class_directory}")
    print(f"\nNumber of classes: {len(distribution)}")
    print(f"Total number of images: {total_images}")
    print("\nClass distribution:")
    print(distribution.to_string(index=False))

    output_dir = Path("reports")
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    distribution.to_csv(
        metrics_dir / "class_distribution.csv",
        index=False
    )

    plt.figure(figsize=(10, 5))
    plt.bar(
        distribution["class"],
        distribution["image_count"]
    )
    plt.title("Number of Images per Class")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        figures_dir / "class_distribution.png",
        dpi=300
    )


if __name__ == "__main__":
    explore_dataset()