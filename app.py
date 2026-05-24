import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="IA Vision Lab",
    page_icon="🧠",
    layout="wide"
)

# Estilo CSS personalizado para mejorar el "Frontend"
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4a4a4a;
    }
    </style>
    """, unsafe_allow_stdio=True)

# ============================================================
# CARGA DE MODELOS (Nombres exactos de tu Repo)
# ============================================================
@st.cache_resource
def load_models():
    try:
        mnist = tf.keras.models.load_model("sistema_prediccion_digits.keras")
        fashion = tf.keras.models.load_model("sistema_prediccion_fashion.keras")
        return mnist, fashion
    except Exception as e:
        st.error(f"Error al cargar modelos: {e}")
        return None, None

model_mnist, model_fashion = load_models()

# ============================================================
# SIDEBAR / CONFIGURACIÓN
# ============================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Configuración")
    tipo_modelo = st.radio(
        "Seleccione el Cerebro de IA:",
        ("Dígitos (MNIST)", "Moda (Fashion MNIST)")
    )
    st.divider()
    st.info("Este sistema utiliza redes neuronales convolucionales (CNN) para clasificar imágenes de 28x28 píxeles.")

# ============================================================
# HEADER
# ============================================================
st.title("🧠 Laboratorio de Visión Artificial")
st.subheader(f"Modelo actual: {tipo_modelo}")

# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📥 Entrada de Imagen")
    archivo_subido = st.file_uploader(
        "Arrastra o selecciona una imagen...", 
        type=["png", "jpg", "jpeg"]
    )

    if archivo_subido:
        # Mostrar imagen original de forma estética
        img = Image.open(archivo_subido)
        st.image(img, caption="Imagen cargada", use_column_width=True)
    else:
        st.warning("Esperando imagen para analizar...")

with col2:
    st.markdown("### 📊 Análisis de la Red Neuronal")
    
    if archivo_subido and st.button("🚀 Ejecutar Predicción", use_container_width=True):
        if model_mnist and model_fashion:
            # 1. Preprocesamiento Técnico
            # Convertir a escala de grises y redimensionar a 28x28
            img_processed = ImageOps.grayscale(img).resize((28, 28))
            img_array = np.array(img_processed) / 255.0  # Normalización
            
            # Ajustar dimensiones (Batch, Ancho, Alto, Canales)
            img_array = img_array.reshape(1, 28, 28, 1)

            # 2. Inferencia
            if tipo_modelo == "Dígitos (MNIST)":
                prediction = model_mnist.predict(img_array)
                clases = [str(i) for i in range(10)]
            else:
                prediction = model_fashion.predict(img_array)
                clases = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

            # 3. Resultados
            clase_id = np.argmax(prediction)
            confianza = np.max(prediction)

            # Visualización de métricas
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Resultado", clases[clase_id])
            with m_col2:
                st.metric("Confianza", f"{confianza:.2%}")

            st.progress(float(confianza))

            # 4. Detalle de todas las probabilidades
            with st.expander("Ver probabilidades completas"):
                import pandas as pd
                chart_data = pd.DataFrame(
                    prediction[0], 
                    index=clases, 
                    columns=["Probabilidad"]
                )
                st.bar_chart(chart_data)
        else:
            st.error("Los modelos no están disponibles.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Desarrollado para el Taller de CNN - 2026")