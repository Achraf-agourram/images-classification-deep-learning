from pathlib import Path
import pandas as pd
import tensorflow as tf


SPLIT_DIR = Path("data/splits")
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
RANDOM_STATE = 42
AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_dataset(csv_path, class_to_index, training=False):
    data = pd.read_csv(csv_path)

    data["label"] = data["label"].map(class_to_index)

    if data["label"].isna().any():
        raise ValueError(f"Unknown labels found in {csv_path}")

    paths = data["path"].astype(str).to_numpy()
    labels = data["label"].astype("int32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(data),
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    if training:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10)
        ])

        dataset = dataset.map(
            lambda images, labels: (
                augmentation(images, training=True),
                labels
            ),
            num_parallel_calls=AUTOTUNE
        )

    return dataset.prefetch(AUTOTUNE)


def create_datasets():
    train_data = pd.read_csv(SPLIT_DIR / "train.csv")

    class_names = sorted(train_data["label"].unique())

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    train_dataset = create_dataset(
        SPLIT_DIR / "train.csv",
        class_to_index,
        training=True
    )

    validation_dataset = create_dataset(
        SPLIT_DIR / "validation.csv",
        class_to_index
    )

    test_dataset = create_dataset(
        SPLIT_DIR / "test.csv",
        class_to_index
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    )


if __name__ == "__main__":
    train_ds, validation_ds, test_ds, class_names = create_datasets()

    print("Classes:", class_names)
    print("Preprocessing completed.")

    images, labels = next(iter(train_ds))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Minimum pixel value:", tf.reduce_min(images).numpy())
    print("Maximum pixel value:", tf.reduce_max(images).numpy())