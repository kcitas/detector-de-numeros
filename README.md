# Detector de números

Aplicación web en Streamlit que reconoce dígitos manuscritos del 0 al 9 con una
CNN entrenada con MNIST (`numerosC2.keras`). El modelo se entrenó con
deformaciones (dígitos en espejo, volteados, girados, inclinados, gruesos y
delgados), así que también reconoce números mal escritos. La interfaz tiene tema oscuro,
panel lateral para elegir la entrada y resultados en dos columnas con barras de
probabilidad.

## Usar la aplicación

En el panel lateral elige:

- **Dibujar**: escribe un número en blanco sobre el lienzo negro. Puedes ajustar
  el grosor del trazo. La predicción se actualiza al terminar cada trazo.
- **Subir imagen**: carga un PNG, JPG, JPEG o BMP con un solo dígito. El
  dígito se recorta y centra automáticamente, y el fondo (claro u oscuro) se
  detecta solo; si falla, elige el fondo a mano en el panel lateral.

## Ejecutar localmente

Requiere Python 3.10–3.12 (TensorFlow 2.16 no soporta versiones más nuevas).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Archivos del proyecto

- `app.py`: interfaz y predicción.
- `.streamlit/config.toml`: tema visual.
- `numerosC2.keras`: modelo Keras entrenado.
- `requirements.txt`: dependencias.
- `train_model.py`: script opcional para reentrenar el modelo con MNIST.

## Deployment

Desplegado con [Streamlit Community Cloud](https://share.streamlit.io/) desde
la rama `main` de este repositorio, con `app.py` como archivo principal y
Python 3.12. Cada push a `main` se despliega automáticamente.
