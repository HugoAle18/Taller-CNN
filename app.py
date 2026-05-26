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
# 2. CSS PROFESIONAL — DARK THEME (Mantenemos el fondo oscuro de la app)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

.stApp { background: #080B14 !important; font-family: 'Syne', sans-serif !important; }

/* SIDEBAR */
[data-testid="stSidebar"] { background: #0D1120 !important; border-right: 1px solid rgba(99,102,241,0.2) !important; }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }

/* RADIO BUTTONS */
[data-testid="stRadio"] > div[role="radiogroup"] { gap: 0.6rem !important; }
[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: #0D1120 !important; border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important; padding: 0.6rem 1rem !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:hover { border-color: rgba(99,102,241,0.6) !important; }
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    border-color: #6366F1 !important; background: rgba(99,102,241,0.1) !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"] p { color: #E2E8F0 !important; }

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
.result-card small { color: #94A3B8; font-family: 'Space Mono', monospace; }
.result-card h3 { margin: 0; color: white; }

/* Títulos principales */
h1, h2, h3, h4, p, span { color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. CARGA DE MODELOS
# ============================================================
@st.cache_resource
def load_vision_engines():
    try:
        # Reemplazar con tus rutas reales si son diferentes
        m1 = tf.keras.models.load_model("mnist_cnn_model.keras")
        m2 = tf.keras.models.load_model("fashion_mnist_cnn_model.keras")
        return m1, m2
    except Exception as e:
        st.sidebar.error(f"⚠️ Error cargando modelos: {e}")
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
    # Definir opciones disponibles
    if engine_choice == "Números (MNIST)":
        mode_options = ["Subir Archivo", "Pizarra Natural (Negro sobre Blanco)"]
    else:
        mode_options = ["Subir Archivo"]
        
    input_mode = st.radio(
        "Fuente de imagen:",
        options=mode_options,
        index=0
    )
    st.markdown("---")
    st.info("💡 **Inversión Automática:** Dibujas en negro sobre blanco, pero la IA recibe blanco sobre negro internamente para máxima precisión.")

# ============================================================
# 5. ENCABEZADO PRINCIPAL
# ============================================================
st.title("🔮 Laboratorio de Clasificación Inteligente")
st.markdown(f"Motor activo: **{engine_choice}**")
st.markdown("---")

col_input, col_result = st.columns([1, 1.2], gap="large")

# Tensor que alimentará al modelo
final_img_tensor = None

with col_input:
    st.subheader("📸 Entrada de Datos")
    
    if input_mode == "Subir Archivo":
        uploaded_file = st.file_uploader("Arrastra imagen", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if uploaded_file:
            img_raw = Image.open(uploaded_file)
            st.image(img_raw, caption="Imagen Subida", use_container_width=True)
            
            # Preprocesamiento inteligente para subidas (invierte si el fondo es claro)
            img_gray = ImageOps.grayscale(img_raw)
            if np.mean(np.array(img_gray)) > 127: # Si el fondo es mayormente blanco
                img_gray = ImageOps.invert(img_gray) # Invierte a número blanco/fondo negro
            img_final = img_gray.resize((28, 28))
            
            # Normalización
            final_img_tensor = np.array(img_final).astype('float32') / 255.0
            final_img_tensor = final_img_tensor.reshape(1, 28, 28, 1)

    elif input_mode == "Pizarra Natural (Negro sobre Blanco)":
        st.write("Dibuja tu número en **NEGRO** sobre el fondo **BLANCO**:")
        canvas_result = st_canvas(
            fill_color="white",       # Color de relleno de formas (no usado aquí)
            stroke_width=20,          # Grosor del trazo (ideal para MNIST)
            stroke_color="#000000",   # Trazo NEGRO
            background_color="#FFFFFF", # Fondo BLANCO
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas_natural",
            update_streamlit=True,
            display_toolbar=True      # Barra de herramientas para borrar (basura)
        )
        
        # Si hay dibujo, preprocesamos e INVERTIMOS
        if canvas_result.image_data is not None and np.any(canvas_result.image_data[:,:,:3] < 255):
            # 1. Obtener datos RGBA y convertir a Grayscale (L)
            raw_draw = canvas_result.image_data.astype('uint8')
            img_pil = Image.fromarray(raw_draw).convert('L') # Ahora es negro sobre blanco
            
            # 2. INVERSIÓN CRUCIAL: Convertir a blanco sobre negro para la IA
            img_inverted = ImageOps.invert(img_pil)
            
            # 3. Redimensionar a 28x28
            img_final = img_inverted.resize((28, 28))
            
            # 4. Normalización
            final_img_tensor = np.array(img_final).astype('float32') / 255.0
            final_img_tensor = final_img_tensor.reshape(1, 28, 28, 1)
            
            # (Opcional) Descomenta esto para ver qué recibe la IA realmente en la barra lateral
            # st.sidebar.image(img_final, caption="Input Real a la IA", width=50)

with col_result:
    st.subheader("⚡ Análisis Dinámico")
    
    if final_img_tensor is not None:
        if st.button("EJECUTAR INFERENCIA IA"):
            # Seleccionar modelo
            model = model_mnist if engine_choice == "Números (MNIST)" else model_fashion
            
            if model:
                preds = model.predict(final_img_tensor)
                
                # Definir etiquetas
                if engine_choice == "Números (MNIST)":
                    labels = [str(i) for i in range(10)]
                else:
                    labels = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo', 
                              'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']
                
                top_idx = np.argmax(preds)
                confidence = np.max(preds)

                # UI de Resultados Elegante
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown(f"""<div class='result-card'>
                        <small>PREDICCIÓN</small><h3>{labels[top_idx]}</h3></div>""", unsafe_allow_html=True)
                with r2:
                    st.markdown(f"""<div class='result-card'>
                        <small>CONFIANZA</small><h3>{confidence:.2%}</h3></div>""", unsafe_allow_html=True)

                st.markdown("---")
                
                # Gráfico de Probabilidades Plotly
                df = pd.DataFrame({'Clase': labels, 'Prob': preds[0]})
                df = df.sort_values('Prob', ascending=True) # Ordenar para el gráfico horizontal
                
                fig = go.Figure(go.Bar(
                    x=df['Prob'], 
                    y=df['Clase'], 
                    orientation='h',
                    marker=dict(
                        color='rgba(99,102,241,0.7)', # Color Indigo sutil
                        line=dict(color='#6366F1', width=1.5)
                    ),
                    hovertemplate='<b>%{y}</b><br>Probabilidad: %{x:.2%}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=dict(text='Distribución de Probabilidades', font=dict(color='white', size=14)),
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#CBD5E1', size=11),
                    height=380, 
                    margin=dict(l=0, r=10, t=40, b=0),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[0, 1], tickformat='.0%'),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Esperando entrada validada (archivo subido o dibujo detectado)...")
        st.markdown("""
        <div style="background: rgba(99,102,241,0.05); border: 1px dashed rgba(99,102,241,0.2); border-radius: 12px; padding: 2rem; text-align: center; color: #94A3B8;">
            <p style="font-size: 2rem; margin-bottom: 0.5rem;">📥</p>
            Proporciona una imagen en el panel izquierdo para iniciar el análisis de IA.
        </div>
        """, unsafe_allow_html=True)
