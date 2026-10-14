
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.data.prepare_dataset import (
    SPLIT_DIR,
    load_and_preprocess_image,
    IMAGE_SIZE
)


def visualize_augmentation():
    data = pd.read_csv(SPLIT_DIR / "train.csv")
    samples = data.sample(n=5, random_state=42)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.10, 0.10),
        tf.keras.layers.RandomContrast(0.10)
    ])

    fig, axes = plt.subplots(
        len(samples), 2,
        figsize=(8, 3 * len(samples)),
        squeeze=False
    )

    for row, (_, sample) in enumerate(samples.iterrows()):
        image, label = load_and_preprocess_image(
            sample["path"],
            sample["label"]
        )

        augmented_image = augmentation(
            tf.expand_dims(image, axis=0),
            training=True
        )[0]

        axes[row, 0].imshow(image.numpy())
        axes[row, 0].set_title(f"Original: {sample['label']}")

        axes[row, 1].imshow(
            tf.clip_by_value(augmented_image, 0.0, 1.0).numpy()
        )
        axes[row, 1].set_title("After augmentation")

        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_augmentation()
