import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pandas as pd

# 1. Configuración inicial
st.set_page_config(page_title="IA Vision Lab", page_icon="🧠", layout="wide")

# 2. CSS Simple (Evita el TypeError)
st.markdown("<style>.stMetric {background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #4a4a4a;}</style>", unsafe_allow_html=True)

# 3. Carga de Modelos
@st.cache_resource
def load_models():
    try:
        m1 = tf.keras.models.load_model("sistema_prediccion_digits.keras")
        m2 = tf.keras.models.load_model("sistema_prediccion_fashion.keras")
        return m1, m2
    except Exception as e:
        st.error(f"Error cargando archivos .keras: {e}")
        return None, None

model_mnist, model_fashion = load_models()

# 4. Sidebar
with st.sidebar:
    st.title("Configuración")
    tipo_modelo = st.radio("Dataset:", ("Dígitos (MNIST)", "Moda (Fashion MNIST)"))

# 5. Cuerpo Principal
st.title("🧠 Laboratorio de Visión Artificial")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Entrada")
    archivo = st.file_uploader("Subir imagen", type=["png", "jpg", "jpeg"])
    if archivo:
        img = Image.open(archivo)
        st.image(img, width=300)

with col2:
    st.subheader("📊 Predicción")
    if archivo and st.button("🚀 Analizar"):
        # Preprocesamiento
        img_gray = ImageOps.grayscale(img).resize((28, 28))
        img_array = np.array(img_gray).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Inferencia
        if tipo_modelo == "Dígitos (MNIST)":
            res = model_mnist.predict(img_array)
            labels = [str(i) for i in range(10)]
        else:
            res = model_fashion.predict(img_array)
            labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        cls_idx = np.argmax(res)
        conf = np.max(res)

        # Mostrar métricas
        st.metric("Clasificación", labels[cls_idx])
        st.metric("Confianza", f"{conf:.2%}")
        st.progress(float(conf))
