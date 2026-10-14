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