from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    tf.keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32)[..., np.newaxis] / 255.0
    x_test = x_test.astype(np.float32)[..., np.newaxis] / 255.0

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2), padding="same"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping],
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    model_path = Path(__file__).resolve().parent / "numerosC2.keras"
    model.save(model_path)

    print(f"Modelo guardado en: {model_path}")
    print(f"Exactitud de prueba: {test_accuracy:.2%}")
    print(f"Pérdida de prueba: {test_loss:.4f}")


if __name__ == "__main__":
    main()
