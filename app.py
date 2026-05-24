import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(page_title="Clasificador MNIST", page_icon="🧠", layout="wide")

# ============================================================
# CARGA DE MODELOS (Caché para no saturar la RAM)
# ============================================================
@st.cache_resource
def load_models():
    # Asegúrate de que los nombres coincidan con tus archivos .keras
    mnist = tf.keras.models.load_model("mnist_model.keras")
    fashion = tf.keras.models.load_model("fashion_mnist_model.keras")
    return mnist, fashion

model_mnist, model_fashion = load_models()

# ============================================================
# HEADER
# ============================================================
st.title("🧠 Clasificador de Imágenes con Redes Neuronales")

# Selector de modelo (estilo tu selectbox anterior)
tipo_modelo = st.sidebar.radio(
    "Selecciona el Dataset:",
    ("Dígitos (MNIST)", "Ropa (Fashion MNIST)")
)

# ============================================================
# ÁREA DE CARGA DE IMAGEN
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🖼️ Cargar Imagen")
    archivo_subido = st.file_uploader("Sube una imagen (28x28 ideal)", type=["png", "jpg", "jpeg"])
    
    if archivo_subido:
        img = Image.open(archivo_subido).convert('L') # Convertir a escala de grises
        st.image(img, caption="Imagen cargada", width=200)

# ============================================================
# PREDICCIÓN
# ============================================================
with col2:
    st.subheader("📊 Resultado de la IA")
    
    if archivo_subido and st.button("🔍 Predecir"):
        # --- PREPROCESAMIENTO ---
        img_res = img.resize((28, 28))
        img_array = np.array(img_res) / 255.0  # Normalizar
        img_array = img_array.reshape(1, 28, 28, 1) # Ajustar forma para el modelo

        # --- SELECCIÓN DE MODELO Y CLASES ---
        if tipo_modelo == "Dígitos (MNIST)":
            pred = model_mnist.predict(img_array)
            clases = [str(i) for i in range(10)]
        else:
            pred = model_fashion.predict(img_array)
            clases = ['Camiseta', 'Pantalón', 'Pullover', 'Vestido', 'Abrigo', 
                      'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']

        # --- MOSTRAR MÉTRICAS ---
        clase_id = np.argmax(pred)
        probabilidad = np.max(pred)

        st.metric("Clase Predicha", clases[clase_id])
        st.metric("Confianza", f"{probabilidad:.2%}")
        
        st.progress(float(probabilidad))