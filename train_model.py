"""Entrena la CNN con MNIST + EMNIST Digits y deformaciones para reconocer dígitos mal escritos.

Datos: MNIST (70.000 dígitos) y EMNIST Digits del NIST (280.000 dígitos de otras
personas, se descarga la primera vez, ~560 MB).

Cada imagen de entrenamiento se deforma al azar: en espejo, volteada, girada,
inclinada, estirada (muy ancha o muy alta), temblorosa, con trazo más grueso o
más delgado, con huecos en la línea o un poco cortada por el borde.
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
from scipy import ndimage


MODEL_PATH = Path(__file__).resolve().parent / "numerosC2.keras"
EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
BATCH_SIZE = 128
MIRROR_PROBABILITY = 0.3
UPSIDE_DOWN_PROBABILITY = 0.2
WORK_SIZE = 48  # lienzo intermedio para que las deformaciones no se salgan del borde


def recenter(image: np.ndarray) -> np.ndarray:
    """Recorta el dígito y lo centra en 20×20 dentro de 28×28, igual que la app."""
    rows, cols = np.nonzero(image > 0.2)
    if len(rows) == 0:
        return np.zeros((28, 28), np.float32)
    digit = image[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    scale = 20 / max(digit.shape)
    digit = np.clip(ndimage.zoom(digit, scale, order=1), 0, 1)[:20, :20]
    result = np.zeros((28, 28), np.float32)
    top, left = (28 - digit.shape[0]) // 2, (28 - digit.shape[1]) // 2
    result[top:top + digit.shape[0], left:left + digit.shape[1]] = digit
    return result


def augment_digit(image: np.ndarray, label: int, rng: np.random.Generator) -> np.ndarray:
    """Deforma un dígito al azar para que parezca mal escrito."""
    if rng.random() < MIRROR_PROBABILITY:
        image = image[:, ::-1]
    if label not in (6, 9) and rng.random() < UPSIDE_DOWN_PROBABILITY:
        image = image[::-1, :]

    # Giro, inclinación y estiramiento (muy ancho o muy alto) en una sola transformación.
    angle = np.radians(rng.uniform(-30, 30))
    shear = rng.uniform(-0.4, 0.4)
    stretch = np.exp(rng.uniform(np.log(0.45), np.log(1.6), size=2)) if rng.random() < 0.4 else rng.uniform(0.85, 1.15, 2)
    # Inversa 2×2 calculada a mano: np.linalg y las multiplicaciones de matrices usan
    # Accelerate en macOS, que se cae (Bus error) si varios hilos lo llaman a la vez.
    cos, sin = np.cos(angle), np.sin(angle)
    a, b = (cos - sin * shear) * stretch[0], -sin * stretch[1]
    c, d = (sin + cos * shear) * stretch[0], cos * stretch[1]
    determinant = a * d - b * c
    rows, cols = np.mgrid[0:WORK_SIZE, 0:WORK_SIZE].astype(np.float32) - WORK_SIZE / 2
    coords = np.stack([
        (d * rows - b * cols) / determinant + 14,
        (-c * rows + a * cols) / determinant + 14,
    ]).reshape(2, -1)

    # Trazo tembloroso u ondulado.
    if rng.random() < 0.5:
        alpha, sigma = rng.uniform(1.5, 3.5), rng.uniform(3, 5)
        for axis in range(2):
            field = ndimage.gaussian_filter(rng.uniform(-1, 1, (WORK_SIZE, WORK_SIZE)), sigma)
            coords[axis] += (field / (np.abs(field).max() + 1e-6) * alpha).ravel()
    image = ndimage.map_coordinates(image, coords, order=1, cval=0).reshape(WORK_SIZE, WORK_SIZE)

    # Trazo más grueso o más delgado.
    choice = rng.random()
    if choice < 0.25:
        image = ndimage.grey_dilation(image, size=(int(rng.integers(2, 4)),) * 2)
    elif choice < 0.35:
        thinner = ndimage.grey_erosion(image, size=(2, 2))
        image = thinner if (thinner > 0.2).sum() > 15 else image

    # Trazo cortado: huecos pequeños en la línea.
    if rng.random() < 0.2:
        rows, cols = np.nonzero(image > 0.5)
        for index in rng.choice(len(rows), size=min(len(rows), int(rng.integers(1, 3))), replace=False):
            size = int(rng.integers(2, 4))
            image[rows[index] - size // 2:rows[index] + size, cols[index] - size // 2:cols[index] + size] = 0

    # Número que toca el borde de la pantalla y queda un poco cortado.
    if rng.random() < 0.1:
        rows, cols = np.nonzero(image > 0.2)
        if len(rows):
            cut = rng.uniform(0.05, 0.15)
            side = rng.integers(4)
            if side == 0:
                image[:rows.min() + int(cut * (rows.max() - rows.min())), :] = 0
            elif side == 1:
                image[rows.max() - int(cut * (rows.max() - rows.min())):, :] = 0
            elif side == 2:
                image[:, :cols.min() + int(cut * (cols.max() - cols.min()))] = 0
            else:
                image[:, cols.max() - int(cut * (cols.max() - cols.min())):] = 0

    return recenter(np.clip(image, 0, 1))


def augment_batch(images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng()
    return np.stack([augment_digit(image[..., 0], int(label), rng) for image, label in zip(images, labels)])[
        ..., np.newaxis
    ].astype(np.float32)


def augment(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    augmented = tf.numpy_function(augment_batch, [images, labels], tf.float32)
    augmented.set_shape([None, 28, 28, 1])
    return augmented, labels


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
        .map(augment, num_parallel_calls=1)
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
