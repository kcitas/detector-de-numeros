"""Entrena la CNN con MNIST y deformaciones para reconocer dígitos mal escritos.

Cada imagen de entrenamiento se deforma al azar: en espejo, volteada, girada,
inclinada, desplazada, con zoom y con trazo más grueso o más delgado.
Los 6 y 9 no se voltean de arriba abajo porque se confundirían entre sí.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf


BATCH_SIZE = 128
MIRROR_PROBABILITY = 0.3
UPSIDE_DOWN_PROBABILITY = 0.2

geometric_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.1, fill_mode="constant"),
        tf.keras.layers.RandomZoom((-0.2, 0.2), fill_mode="constant"),
        tf.keras.layers.RandomTranslation(0.12, 0.12, fill_mode="constant"),
    ]
)


def random_shear(images: tf.Tensor) -> tf.Tensor:
    shear = tf.random.uniform([tf.shape(images)[0], 1], -0.4, 0.4)
    zeros = tf.zeros_like(shear)
    ones = tf.ones_like(shear)
    # Transformación afín [a0, a1, a2, b0, b1, b2, c0, c1] centrada en la imagen.
    transforms = tf.concat([ones, shear, -shear * 14.0, zeros, ones, zeros, zeros, zeros], axis=1)
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transforms,
        output_shape=tf.shape(images)[1:3],
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
    )


def random_stroke_width(images: tf.Tensor) -> tf.Tensor:
    thicker = tf.nn.max_pool2d(images, ksize=3, strides=1, padding="SAME")
    thinner = -tf.nn.max_pool2d(-images, ksize=2, strides=1, padding="SAME")
    choice = tf.random.uniform([tf.shape(images)[0], 1, 1, 1])
    return tf.where(choice < 0.25, thicker, tf.where(choice < 0.35, thinner, images))


def random_flips(images: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    batch = tf.shape(images)[0]
    mirror = tf.random.uniform([batch]) < MIRROR_PROBABILITY
    can_turn = tf.logical_and(labels != 6, labels != 9)
    upside_down = tf.logical_and(tf.random.uniform([batch]) < UPSIDE_DOWN_PROBABILITY, can_turn)
    images = tf.where(mirror[:, None, None, None], tf.reverse(images, axis=[2]), images)
    return tf.where(upside_down[:, None, None, None], tf.reverse(images, axis=[1]), images)


def augment(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    images = random_flips(images, labels)
    images = random_shear(images)
    images = geometric_augmentation(images, training=True)
    images = random_stroke_width(images)
    return tf.clip_by_value(images, 0.0, 1.0), labels


def build_model() -> tf.keras.Model:
    def conv_block(filters: int) -> list[tf.keras.layers.Layer]:
        return [
            tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ]

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            *conv_block(32),
            *conv_block(32),
            tf.keras.layers.MaxPooling2D(),
            *conv_block(64),
            *conv_block(64),
            tf.keras.layers.MaxPooling2D(),
            *conv_block(128),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )


def main() -> None:
    tf.keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32)[..., np.newaxis] / 255.0
    x_test = x_test.astype(np.float32)[..., np.newaxis] / 255.0

    train_data = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train.astype(np.int32)))
        .shuffle(len(x_train))
        .batch(BATCH_SIZE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(
        train_data,
        epochs=12,
        validation_data=(x_test, y_test),
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ],
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    model_path = Path(__file__).resolve().parent / "numerosC2.keras"
    model.save(model_path)
    print(f"Precisión en MNIST test: {test_accuracy:.2%} — modelo guardado en {model_path.name}")


if __name__ == "__main__":
    main()
