import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy import ndimage
from streamlit_drawable_canvas import st_canvas


MODEL_PATH = Path(__file__).resolve().parent / "numerosC2.keras"
CLASS_NAMES = [str(digit) for digit in range(10)]
CANVAS_SIZE = 300
LOW_CONFIDENCE = 0.6
SPECK_FRACTION = 0.08

st.set_page_config(page_title="Detector de números", page_icon="✍️", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    .hero h1 { font-size: 2.4rem; margin-bottom: 0; }
    .hero p { color: #9AA3B5; margin-top: 0.25rem; }
    .result-card {
        background: #1A1F2B;
        border: 1px solid #2A3142;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        color: #E8EAF0;
    }
    .result-card b { color: #FFFFFF; }
    .result-card .label { color: #9AA3B5; font-size: 0.9rem; letter-spacing: 0.08em; }
    .result-card .digit { font-size: 6rem; font-weight: 800; line-height: 1.1; color: #F5B700; }
    .result-card .confidence { font-size: 1.1rem; }
    .prob-row { display: flex; align-items: center; gap: 0.6rem; margin: 0.3rem 0; }
    .prob-row .name { width: 1rem; font-weight: 700; }
    .prob-row .track { flex: 1; height: 12px; background: #2A3142; border-radius: 6px; overflow: hidden; }
    .prob-row .fill { height: 100%; background: #4C6EF5; border-radius: 6px; }
    .prob-row.top .fill { background: #F5B700; }
    .prob-row .value { width: 3.5rem; text-align: right; color: #9AA3B5; font-variant-numeric: tabular-nums; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>✍️ Detector de números</h1>
      <p>Una red neuronal convolucional entrenada con 300.000 dígitos (MNIST y EMNIST) reconoce del 0 al 9, incluso mal escritos.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file():
    st.error(
        f"No se encontró `{MODEL_PATH.name}` junto a `app.py`. "
        "Descarga el modelo entrenado y agrégalo al repositorio."
    )
    st.stop()


@st.cache_data
def model_fingerprint(model_path: str) -> str:
    return hashlib.sha256(Path(model_path).read_bytes()).hexdigest()[:8]


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


try:
    model = load_model(str(MODEL_PATH))
except (OSError, ValueError) as error:
    st.error(f"No se pudo cargar `{MODEL_PATH.name}`: {error}")
    st.stop()


def remove_specks(pixels: np.ndarray) -> np.ndarray:
    """Borra manchas y puntos sueltos mucho más pequeños que el trazo principal."""
    mask = pixels > 50
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count <= 1:
        return pixels
    sizes = ndimage.sum(mask, labels, index=range(1, count + 1))
    keep = np.flatnonzero(sizes >= SPECK_FRACTION * sizes.max()) + 1
    return np.where(np.isin(labels, keep), pixels, 0).astype(pixels.dtype)


def to_mnist_format(image: Image.Image) -> Image.Image | None:
    """Recorta el dígito (blanco sobre negro) y lo centra en 28×28 como en MNIST."""
    image = Image.fromarray(remove_specks(np.asarray(image)))
    digit_mask = image.point(lambda pixel: 255 if pixel > 50 else 0)
    bounding_box = digit_mask.getbbox()
    if bounding_box is None:
        return None

    digit = image.crop(bounding_box)
    scale = 20 / max(digit.width, digit.height)
    digit = digit.resize(
        (max(1, round(digit.width * scale)), max(1, round(digit.height * scale))),
        Image.Resampling.LANCZOS,
    )
    mnist_image = Image.new("L", (28, 28), color=0)
    mnist_image.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    return mnist_image


def prepare_canvas_image(image_data: np.ndarray) -> Image.Image | None:
    image = Image.fromarray(image_data.astype(np.uint8), mode="RGBA").convert("L")
    return to_mnist_format(image)


def prepare_uploaded_image(uploaded_file, background: str) -> Image.Image | None:
    image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("L")
    if background == "Automático":
        has_light_background = float(np.median(np.asarray(image))) > 127
    else:
        has_light_background = background == "Claro"
    if has_light_background:
        image = ImageOps.invert(image)
    return to_mnist_format(image)


def render_result(image: Image.Image) -> None:
    model_input = (np.asarray(image, dtype=np.float32) / 255.0)[np.newaxis, ..., np.newaxis]
    probabilities = model.predict(model_input, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))

    st.markdown(
        f"""
        <div class="result-card">
          <div class="label">EL MODELO VE UN</div>
          <div class="digit">{CLASS_NAMES[predicted_digit]}</div>
          <div class="confidence">Confianza: <b>{float(probabilities[predicted_digit]):.1%}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if probabilities[predicted_digit] < LOW_CONFIDENCE:
        st.warning(
            "El modelo no está muy seguro. Prueba dibujar el número más grande "
            "y centrado, o con un trazo más grueso."
        )

    st.markdown("#### Probabilidad por dígito")
    rows = "".join(
        f"""
        <div class="prob-row{' top' if index == predicted_digit else ''}">
          <span class="name">{name}</span>
          <div class="track"><div class="fill" style="width:{probability:.1%}"></div></div>
          <span class="value">{probability:.1%}</span>
        </div>
        """
        for index, (name, probability) in enumerate(zip(CLASS_NAMES, map(float, probabilities)))
    )
    st.markdown(rows, unsafe_allow_html=True)

    with st.expander("Ver imagen de 28×28 que recibe el modelo"):
        st.image(image.resize((140, 140), Image.Resampling.NEAREST))


def render_placeholder(message: str) -> None:
    st.markdown(
        f"""
        <div class="result-card">
          <div class="label">ESPERANDO ENTRADA</div>
          <div class="digit" style="color:#2A3142">?</div>
          <div class="confidence">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.header("Entrada")
    input_mode = st.radio(
        "¿Cómo quieres ingresar el número?",
        ("Dibujar", "Subir imagen"),
        label_visibility="collapsed",
    )
    if input_mode == "Dibujar":
        stroke_width = st.slider("Grosor del trazo", 10, 30, 20)
        if st.button("🗑️ Borrar lienzo", use_container_width=True):
            st.session_state.canvas_version = st.session_state.get("canvas_version", 0) + 1
        st.caption("También puedes deshacer trazos con los íconos bajo el lienzo.")
    else:
        background = st.radio(
            "Fondo de la imagen",
            ("Automático", "Oscuro", "Claro"),
            help="En automático se detecta si el dígito es oscuro sobre fondo claro.",
        )
    st.divider()
    st.caption(
        "Modelo: CNN Keras (`numerosC2.keras`) entrenada con MNIST y EMNIST. "
        f"Versión: `{model_fingerprint(str(MODEL_PATH))}`"
    )

input_column, result_column = st.columns([1, 1], gap="large")

with input_column:
    if input_mode == "Dibujar":
        st.subheader("Dibuja un dígito")
        canvas_result = st_canvas(
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            display_toolbar=True,
            update_streamlit=True,
            key=f"digit_canvas_{st.session_state.get('canvas_version', 0)}",
        )
        digit_image = None
        if canvas_result.image_data is not None:
            digit_image = prepare_canvas_image(canvas_result.image_data)
        placeholder_message = "Dibuja un número en el lienzo."
    else:
        st.subheader("Sube una imagen")
        uploaded_file = st.file_uploader(
            "Imagen PNG, JPG, JPEG o BMP con un solo dígito",
            type=("png", "jpg", "jpeg", "bmp"),
        )
        digit_image = None
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Imagen original", width=CANVAS_SIZE)
            try:
                digit_image = prepare_uploaded_image(uploaded_file, background)
            except (UnidentifiedImageError, OSError) as error:
                st.error(f"No se pudo abrir la imagen: {error}")
        placeholder_message = (
            "No se encontró un dígito en la imagen. Prueba otro fondo en el panel lateral."
            if uploaded_file is not None
            else "Sube una imagen para analizarla."
        )

with result_column:
    st.subheader("Resultado")
    if digit_image is None:
        render_placeholder(placeholder_message)
    else:
        render_result(digit_image)
