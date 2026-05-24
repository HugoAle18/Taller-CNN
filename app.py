import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import plotly.express as px

# ============================================================
# 1. CONFIGURACIÓN ESTÉTICA (LIGHT THEME)
# ============================================================
st.set_page_config(
    page_title="Vision Lab Pro 2.0",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS para un acabado de "Consultoría de IA"
st.markdown("""
    <style>
    /* Fondo principal y fuentes */
    .stApp {
        background: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    
    /* Tarjetas dinámicas (Glassmorphism Light) */
    div.stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.4);
    }
    
    /* Contenedores personalizados */
    .metric-card {
        background: white;
        padding: 24px;
        border-radius: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
    }
    
    h1, h2, h3 { color: #1E293B !important; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 2. LÓGICA DE CARGA DE MODELOS (NUEVOS NOMBRES)
# ============================================================
@st.cache_resource
def load_vision_engines():
    try:
        # Usando tus nuevos nombres de archivo
        m1 = tf.keras.models.load_model("mnist_cnn_model.keras")
        m2 = tf.keras.models.load_model("fashion_mnist_cnn_model.keras")
        return m1, m2
    except Exception as e:
        st.sidebar.error(f"⚠️ Error cargando modelos: {e}")
        return None, None

model_mnist, model_fashion = load_vision_engines()

# ============================================================
# 3. INTERFAZ Y NAVEGACIÓN
# ============================================================
with st.sidebar:
    st.image("http://googleusercontent.com/image_collection/image_retrieval/17860148785345437328", width=120)
    st.title("Vision Lab Pro")
    st.markdown("---")
    
    engine_choice = st.segmented_control(
        "Cerebro Activo:",
        options=["Números (MNIST)", "Moda (Fashion)"],
        default="Números (MNIST)"
    )
    
    st.info("💡 **Tip:** El sistema invierte automáticamente los colores si detecta un fondo blanco para máxima precisión.")

# ============================================================
# 4. CUERPO PRINCIPAL (DINÁMICO)
# ============================================================
st.title("🔮 Laboratorio de Clasificación Inteligente")
st.write(f"Inferencia en tiempo real con motor de **{engine_choice}**")

# Layout de dos columnas principales
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown("### 📸 Captura de Datos")
    uploaded_file = st.file_uploader("Arrastra una imagen", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        img_raw = Image.open(uploaded_file)
        st.image(img_raw, caption="Imagen de entrada", use_container_width=True)
    else:
        st.empty()

with col_result:
    st.markdown("### ⚡ Análisis Dinámico")
    
    if uploaded_file and st.button("Analizar Patrones"):
        if model_mnist and model_fashion:
            # --- PROCESAMIENTO ---
            img_gray = ImageOps.grayscale(img_raw)
            
            # Auto-inversión (MNIST necesita blanco sobre negro)
            if np.mean(np.array(img_gray)) > 120:
                img_gray = ImageOps.invert(img_gray)
            
            img_final = img_gray.resize((28, 28))
            img_tensor = np.array(img_final).astype('float32') / 255.0
            img_tensor = img_tensor.reshape(1, 28, 28, 1)

            # --- INFERENCIA ---
            if engine_choice == "Números (MNIST)":
                preds = model_mnist.predict(img_tensor)
                labels = [str(i) for i in range(10)]
            else:
                preds = model_fashion.predict(img_tensor)
                labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

            # --- RESULTADOS ---
            top_idx = np.argmax(preds)
            confidence = np.max(preds)

            # Visualización de métricas en tarjetas
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Predicción", labels[top_idx])
            with m2:
                st.metric("Confianza", f"{confidence:.2%}")

            # Gráfico dinámico de probabilidades (Plotly)
            df_chart = pd.DataFrame({'Clase': labels, 'Confianza': preds[0]})
            fig = px.bar(df_chart, x='Confianza', y='Clase', orientation='h', 
                         title="Distribución de Probabilidades",
                         color='Confianza', color_continuous_scale='Purples')
            fig.update_layout(showlegend=False, height=400, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("No se detectaron los modelos en el repositorio.")

# Footer
st.markdown("---")
st.caption("🔍 Vision Lab Pro | Powered by TensorFlow & Streamlit Cloud")