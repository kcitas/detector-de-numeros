# Detector de números

Aplicación web en Streamlit que reconoce dígitos manuscritos del 0 al 9, incluso
cuando están mal escritos: en espejo, volteados, girados, inclinados, estirados,
temblorosos, muy gruesos o muy delgados.

App publicada: <https://detector-de-numeros-etbjfv4pcktcfo9343uqc4.streamlit.app/>

## Usar la aplicación

En el panel lateral elige:

- **Dibujar**: escribe un número en blanco sobre el lienzo negro. Puedes ajustar
  el grosor del trazo y borrar con el botón **Borrar lienzo**. La predicción se
  actualiza al terminar cada trazo.
- **Subir imagen**: carga un PNG, JPG, JPEG o BMP con un solo dígito. El fondo
  (claro u oscuro) se detecta solo; si falla, elígelo a mano en el panel lateral.

El resultado muestra el dígito, la confianza y la probabilidad de cada dígito.
Si la confianza es menor al 60 % aparece un aviso. Debajo del resultado,
**Ver imagen de 28×28** muestra exactamente lo que recibe el modelo.

Al final del panel lateral aparece la **versión del modelo** (los primeros 8
caracteres del SHA-256 de `numerosC2.keras`), para comprobar qué modelo está
desplegado.

## Cómo funciona

### Preprocesamiento (`app.py`)

Antes de pasar la imagen al modelo:

1. Se convierte a escala de grises con el dígito en blanco sobre negro.
2. Se borran manchas y puntos sueltos mucho más pequeños que el trazo principal.
3. Se recorta el dígito y se escala para que su lado mayor mida 20 píxeles,
   sin deformarlo.
4. Se centra en una imagen de 28×28, el mismo formato que MNIST.

### Modelo (`numerosC2.keras`)

Red neuronal convolucional (Keras) con 5 capas convolucionales con
normalización por lotes, *global average pooling* y dropout.

### Datos de entrenamiento (`train_model.py`)

- **MNIST**: 70.000 dígitos.
- **EMNIST Digits** (NIST): 280.000 dígitos de otras personas. Se descarga sola
  la primera vez (~560 MB).

Cada imagen se deforma al azar en cada época para simular números mal escritos:

| Deformación | Detalle |
|---|---|
| Espejo | 30 % de las imágenes |
| Volteado de arriba abajo | 20 %, excepto 6 y 9 (volteados se confunden entre sí) |
| Giro | hasta ±30° |
| Inclinación | hasta ±0,4 |
| Estiramiento | muy ancho o muy alto (de 0,45× a 1,6× por eje) |
| Trazo tembloroso | ondulaciones suaves en el 50 % de las imágenes |
| Grosor | más grueso (25 %) o más delgado (10 %) |
| Trazo cortado | huecos pequeños en la línea (20 %) |
| Borde | un pedazo cortado, como si tocara el borde (10 %) |

Después de deformarla, cada imagen se recorta y centra igual que en la app.

Las imágenes de prueba de MNIST y EMNIST no se usan para entrenar. Además se
evalúa con **USPS** (dígitos escaneados de sobres de correo), una fuente
distinta que tampoco se usa para entrenar.

### Limitaciones

- Un **6 o un 9 volteado de cabeza** no se puede distinguir: un 6 al revés es un 9.
- Reconoce **un solo dígito** a la vez, no números de varias cifras.

## Ejecutar localmente

Requiere Python 3.10–3.12 (TensorFlow 2.16 no soporta versiones más nuevas).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

En Mac con Apple Silicon no existe `tensorflow-cpu`; instala `tensorflow==2.16.1`
en su lugar.

## Reentrenar el modelo

```bash
python train_model.py              # desde cero (~8 min por época en CPU)
python train_model.py --fine-tune  # continúa desde numerosC2.keras
python train_model.py --fine-tune --epochs 6
```

El modelo se guarda en `numerosC2.keras`.

## Archivos del proyecto

- `app.py`: interfaz, preprocesamiento y predicción.
- `.streamlit/config.toml`: tema oscuro.
- `numerosC2.keras`: modelo entrenado.
- `requirements.txt`: dependencias.
- `train_model.py`: descarga de datos, deformaciones y entrenamiento.

## Deployment

Desplegado con [Streamlit Community Cloud](https://share.streamlit.io/) desde la
rama `main` de [kcitas/detector-de-numeros](https://github.com/kcitas/detector-de-numeros),
con `app.py` como archivo principal y Python 3.12. Cada push a `main` se
despliega automáticamente.

En el plan gratis la app se duerme si nadie la usa por un tiempo; al abrirla
muestra "Your app is waking up!" y tarda uno o dos minutos en arrancar.
