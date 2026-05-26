import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas

# ============================================================
# 1. CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Vision Lab Pro 2.0",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ocultar elementos innecesarios pero MANTENER el botón de sidebar
st.markdown("""
<style>
#MainMenu { display: none !important; }
footer { display: none !important; }
.stApp header > div:last-child { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. CSS PROFESIONAL — DARK THEME
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

.stApp { background: #080B14 !important; font-family: 'Syne', sans-serif !important; }

/* SIDEBAR */
[data-testid="stSidebar"] { background: #0D1120 !important; border-right: 1px solid rgba(99,102,241,0.2) !important; }

/* RADIO BUTTONS */
[data-testid="stRadio"] > div[role="radiogroup"] { gap: 0.6rem !important; }
[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: #0D1120 !important; border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important; padding: 0.6rem 1rem !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    border-color: #6366F1 !important; background: rgba(99,102,241,0.1) !important;
}

/* BOTÓN */
div.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important; color: white !important;
    border: none !important; padding: 0.75rem 2rem !important; border-radius: 12px !important;
    font-family: 'Space Mono', monospace !important; font-weight: 700 !important; width: 100% !important;
}

/* PANEL DE RESULTADOS */
.result-card {
    background: linear-gradient(135deg, #131929, #0F172A);
    border: 1px solid rgba(99,102,241,0.25); border-radius: 16px; padding: 1.25rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. CARGA DE MODELOS
# ============================================================
@st.cache_resource
def load_vision_engines():
    try:
        m1 = tf.keras.models.load_model("mnist_cnn_model.keras")
        m2 = tf.keras.models.load_model("fashion_mnist_cnn_model.keras")
        return m1, m2
    except Exception as e:
        st.sidebar.error(f"⚠️ Error: {e}")
        return None, None

model_mnist, model_fashion = load_vision_engines()

# ============================================================
# 4. SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🔮 Vision Lab Pro")
    st.markdown("---")
    st.write("MOTOR IA:")
    engine_choice = st.radio(
        "Seleccionar Modelo:",
        options=["Números (MNIST)", "Moda (Fashion)"],
        index=0
    )
    
    st.write("MÉTODO DE ENTRADA:")
    input_mode = st.radio(
        "Fuente de imagen:",
        options=["Subir Archivo", "Pizarra (Solo MNIST)"] if engine_choice == "Números (MNIST)" else ["Subir Archivo"]
    )
    st.info("💡 La Pizarra es ideal para probar trazos manuales rápidos.")

# ============================================================
# 5. CUERPO PRINCIPAL
# ============================================================
st.title("🔮 Laboratorio de Clasificación Inteligente")
st.markdown(f"Motor activo: **{engine_choice}**")
st.markdown("---")

col_input, col_result = st.columns([1, 1.2], gap="large")

img_tensor = None

with col_input:
    st.subheader("📸 Entrada de Datos")
    
    if input_mode == "Subir Archivo":
        uploaded_file = st.file_uploader("Arrastra imagen", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if uploaded_file:
            img_raw = Image.open(uploaded_file)
            st.image(img_raw, caption="Imagen Subida", use_container_width=True)
            
            # Preprocesamiento
            img_gray = ImageOps.grayscale(img_raw)
            if np.mean(np.array(img_gray)) > 127:
                img_gray = ImageOps.invert(img_gray)
            img_final = img_gray.resize((28, 28))
            img_tensor = np.array(img_final).astype('float32') / 255.0
            img_tensor = img_tensor.reshape(1, 28, 28, 1)

    else:
        st.write("Dibuja aquí abajo:")
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas",
            update_streamlit=True,
            display_toolbar=True # Permite borrar (ícono de basura)
        )
        if canvas_result.image_data is not None:
            # Convertir el dibujo a formato MNIST
            raw_draw = canvas_result.image_data.astype('uint8')
            img_pil = Image.fromarray(raw_draw).convert('L')
            img_final = img_pil.resize((28, 28))
            img_tensor = np.array(img_final).astype('float32') / 255.0
            img_tensor = img_tensor.reshape(1, 28, 28, 1)

with col_result:
    st.subheader("⚡ Análisis Dinámico")
    
    if img_tensor is not None:
        if st.button("EJECUTAR INFERENCIA IA"):
            model = model_mnist if engine_choice == "Números (MNIST)" else model_fashion
            
            if model:
                preds = model.predict(img_tensor)
                
                if engine_choice == "Números (MNIST)":
                    labels = [str(i) for i in range(10)]
                else:
                    labels = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo', 
                              'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']
                
                top_idx = np.argmax(preds)
                confidence = np.max(preds)

                # UI de Resultados
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown(f"""<div class='result-card'>
                        <small>PREDICCIÓN</small><h3>{labels[top_idx]}</h3></div>""", unsafe_allow_html=True)
                with r2:
                    st.markdown(f"""<div class='result-card'>
                        <small>CONFIANZA</small><h3>{confidence:.2%}</h3></div>""", unsafe_allow_html=True)

                # Gráfico
                df = pd.DataFrame({'Clase': labels, 'Prob': preds[0]})
                fig = go.Figure(go.Bar(
                    x=df['Prob'], y=df['Clase'], orientation='h',
                    marker=dict(color='rgba(99,102,241,0.6)', line=dict(color='#6366F1', width=1))
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'), height=350, margin=dict(l=0, r=0, t=30, b=0),
                    xaxis=dict(showgrid=False, range=[0, 1]), yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Esperando entrada (archivo o dibujo)...")
