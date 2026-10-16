import tensorflow as tf


def build_cnn(input_shape=(150, 150, 3), num_classes=6):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same"
        ),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same"
        ),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same"
        ),
        tf.keras.layers.ReLU(),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(128),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.30),

        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


if __name__ == "__main__":
    model = build_cnn()
    model.summary()
