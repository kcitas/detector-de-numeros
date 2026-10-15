"""Entrena la CNN con MNIST + EMNIST Digits y deformaciones para reconocer dígitos mal escritos.

Datos: MNIST (70.000 dígitos) y EMNIST Digits del NIST (280.000 dígitos de otras
personas, se descarga la primera vez, ~560 MB).

Cada imagen de entrenamiento se deforma al azar: en espejo, volteada, girada,
inclinada, desplazada, con zoom y con trazo más grueso o más delgado.
Los 6 y 9 no se voltean de arriba abajo porque se confundirían entre sí.

Uso:
    python train_model.py              # entrena desde cero
    python train_model.py --fine-tune  # continúa desde numerosC2.keras (más rápido)
"""

import argparse
import gzip
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf


MODEL_PATH = Path(__file__).resolve().parent / "numerosC2.keras"
EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
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


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        data = file.read()
    dimensions = data[3]
    shape = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(dimensions)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * dimensions).reshape(shape)


def load_emnist_digits() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    archive = Path(tf.keras.utils.get_file("emnist-gzip.zip", EMNIST_URL))
    folder = archive.parent / "emnist-digits"
    folder.mkdir(exist_ok=True)
    splits = {}
    for split in ("train", "test"):
        arrays = []
        for kind in ("images-idx3", "labels-idx1"):
            name = f"emnist-digits-{split}-{kind}-ubyte.gz"
            if not (folder / name).exists():
                with zipfile.ZipFile(archive) as zipped:
                    (folder / name).write_bytes(zipped.read(f"gzip/{name}"))
            arrays.append(read_idx(folder / name))
        images, labels = arrays
        # EMNIST guarda las imágenes transpuestas respecto a MNIST.
        splits[split] = (images.transpose(0, 2, 1), labels)
    return splits["train"], splits["test"]


def to_model_input(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32)[..., np.newaxis] / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine-tune", action="store_true", help="continúa desde el modelo guardado")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(42)

    (mnist_x, mnist_y), (mnist_test_x, mnist_test_y) = tf.keras.datasets.mnist.load_data()
    (emnist_x, emnist_y), (emnist_test_x, emnist_test_y) = load_emnist_digits()
    x_train = to_model_input(np.concatenate([mnist_x, emnist_x]))
    y_train = np.concatenate([mnist_y, emnist_y]).astype(np.int32)
    x_test = to_model_input(np.concatenate([mnist_test_x, emnist_test_x]))
    y_test = np.concatenate([mnist_test_y, emnist_test_y]).astype(np.int32)
    print(f"Entrenando con {len(x_train):,} dígitos, validando con {len(x_test):,}")

    train_data = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train))
        .batch(BATCH_SIZE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    if args.fine_tune:
        model = tf.keras.models.load_model(MODEL_PATH)
        learning_rate = 3e-4
    else:
        model = build_model()
        learning_rate = 1e-3
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.fit(
        train_data,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ],
    )

    _, mnist_accuracy = model.evaluate(to_model_input(mnist_test_x), mnist_test_y, verbose=0)
    _, emnist_accuracy = model.evaluate(to_model_input(emnist_test_x), emnist_test_y, verbose=0)
    model.save(MODEL_PATH)
    print(
        f"Precisión en MNIST test: {mnist_accuracy:.2%}, EMNIST test: {emnist_accuracy:.2%} "
        f"— modelo guardado en {MODEL_PATH.name}"
    )


if __name__ == "__main__":
    main()
