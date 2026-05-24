import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pandas as pd

# ============================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(page_title="IA Vision Lab Pro", page_icon="🧠", layout="wide")

# CSS para que se vea profesional como tu trabajo anterior
st.markdown("""
    <style>
    .stMetric {background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #4a4a4a;}
    [data-testid="stSidebar"] {background-color: #0e1117;}
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 2. CARGA DE MODELOS
# ============================================================
@st.cache_resource
def load_models():
    try:
        # Usando los nombres exactos de tu repositorio
        m1 = tf.keras.models.load_model("sistema_prediccion_digits.keras")
        m2 = tf.keras.models.load_model("sistema_prediccion_fashion.keras")
        return m1, m2
    except Exception as e:
        st.error(f"Error cargando los modelos: {e}")
        return None, None

model_mnist, model_fashion = load_models()

# ============================================================
# 3. BARRA LATERAL (SIDEBAR)
# ============================================================
with st.sidebar:
    st.header("🧠 Panel de Control")
    tipo_modelo = st.radio(
        "Selecciona el Cerebro de IA:",
        ("Dígitos (MNIST)", "Moda (Fashion MNIST)")
    )
    st.divider()
    st.write("**Instrucciones:**")
    st.caption("1. Sube una imagen clara.")
    st.caption("2. El sistema centrará y ajustará los colores automáticamente.")
    st.caption("3. Presiona Analizar.")

# ============================================================
# 4. CUERPO PRINCIPAL
# ============================================================
st.title("🧪 Laboratorio de Visión Artificial")
st.markdown(f"### Analizando con: **{tipo_modelo}**")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Entrada de Imagen")
    archivo = st.file_uploader("Cargar imagen (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    
    if archivo:
        img_original = Image.open(archivo)
        st.image(img_original, caption="Imagen Original", use_column_width=True)

with col2:
    st.subheader("📊 Resultados de Predicción")
    
    if archivo and st.button("🚀 Ejecutar Análisis", use_container_width=True):
        if model_mnist and model_fashion:
            with st.spinner("La IA está pensando..."):
                # --- PREPROCESAMIENTO AVANZADO ---
                # 1. Convertir a escala de grises
                img_gray = ImageOps.grayscale(img_original)
                
                # 2. Inversión Inteligente:
                # Si el promedio de píxeles es alto (>127), el fondo es claro.
                # MNIST necesita fondo NEGRO, así que invertimos si es claro.
                if np.mean(np.array(img_gray)) > 127:
                    img_gray = ImageOps.invert(img_gray)
                
                # 3. Redimensionar a 28x28 (Tamaño exacto que espera Keras)
                img_res = img_gray.resize((28, 28))
                
                # 4. Normalizar y dar forma (Shape: 1, 28, 28, 1)
                img_array = np.array(img_res).astype('float32') / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)

                # --- PREDICCIÓN ---
                if tipo_modelo == "Dígitos (MNIST)":
                    prediction = model_mnist.predict(img_array)
                    labels = [str(i) for i in range(10)]
                else:
                    prediction = model_fashion.predict(img_array)
                    labels = ['Camiseta', 'Pantalón', 'Pullover', 'Vestido', 'Abrigo', 
                              'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']

                # --- MOSTRAR RESULTADOS ---
                idx = np.argmax(prediction)
                prob = np.max(prediction)

                # Métrica Principal
                st.metric(label="Clasificación Final", value=labels[idx], delta=f"{prob:.2%} de confianza")
                
                # Barra de progreso
                st.progress(float(prob))

                # Debug visual: Ver qué está viendo la IA realmente
                with st.expander("🔎 Ver procesamiento interno"):
                    st.write("Así es como la IA 've' tu imagen después del ajuste:")
                    st.image(img_res, width=150)
                    
                    # Gráfico de barras con todas las probabilidades
                    df_prob = pd.DataFrame({'Clase': labels, 'Probabilidad': prediction[0]})
                    st.bar_chart(df_prob.set_index('Clase'))
        else:
            st.error("Modelos no cargados. Revisa los archivos en GitHub.")
    else:
        st.info("Sube una imagen y presiona el botón para comenzar.")

# ============================================================
# 5. FOOTER
# ============================================================
st.divider()
st.caption("© 2026 Taller CNN - Sistema de Clasificación Multimodal")
