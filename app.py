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
    initial_sidebar_state="expanded" # Forza a que inicie abierta
)

# ============================================================
# 2. CSS PROFESIONAL — LIGHT THEME & SIDEBAR SEGURO
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

/* Fondo general de la app */
.stApp { background: #F8FAFC !important; font-family: 'Syne', sans-serif !important; }

/* Ocultar SOLO el Menú de opciones de Streamlit y el Footer, MANTENEMOS EL HEADER para no perder el botón */
#MainMenu { display: none !important; }
footer { display: none !important; }

/* Diseño del Sidebar */
[data-testid="stSidebar"] { 
    background: #FFFFFF !important; 
    border-right: 1px solid #E2E8F0 !important; 
    min-width: 320px !important;
    max-width: 320px !important;
}
[data-testid="stSidebar"] * { color: #334155 !important; }

/* RADIO BUTTONS */
[data-testid="stRadio"] > div[role="radiogroup"] { gap: 0.6rem !important; }
[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: #FFFFFF !important; border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important; padding: 0.6rem 1rem !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:hover { border-color: #6366F1 !important; background: #F8FAFC !important; }
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    border-color: #6366F1 !important; background: #EEF2FF !important;
    box-shadow: 0 2px 4px rgba(99,102,241,0.1) !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"] p { color: #1E293B !important; font-weight: 600 !important; }

/* BOTÓN PRINCIPAL */
div.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important; color: white !important;
    border: none !important; padding: 0.75rem 2rem !important; border-radius: 12px !important;
    font-family: 'Space Mono', monospace !important; font-weight: 700 !important; width: 100% !important;
    box-shadow: 0 4px 6px -1px rgba(99,102,241,0.2) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important; box-shadow: 0 10px 15px -3px rgba(99,102,241,0.3) !important;
}

/* Títulos principales y textos */
h1, h2, h3 { color: #0F172A !important; }
p, span, label { color: #475569 !important; }

/* PANEL DE RESULTADOS */
.result-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0; 
    border-radius: 16px; 
    padding: 1.25rem;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
}
.result-card small { color: #64748B; font-family: 'Space Mono', monospace; font-size: 0.75rem; letter-spacing: 0.05em; }
.result-card h3 { margin: 0.2rem 0 0 0 !important; color: #1E293B !important; font-weight: 800 !important; }

/* Alertas/Info */
[data-testid="stAlert"] { background: #EEF2FF !important; border: 1px solid #C7D2FE !important; color: #3730A3 !important; }
[data-testid="stAlert"] p { color: #3730A3 !important; }
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
    if engine_choice == "Números (MNIST)":
        mode_options = ["Subir Archivo", "Pizarra Natural (Negro sobre Blanco)"]
    else:
        mode_options = ["Subir Archivo", "Cámara en Vivo"]
        
    input_mode = st.radio(
        "Fuente de imagen:",
        options=mode_options,
        index=0
    )
    st.markdown("---")
    st.info("💡 **Inversión Automática:** La IA convierte los fondos blancos a negros internamente para máxima precisión.")

# ============================================================
# 5. ENCABEZADO PRINCIPAL
# ============================================================
st.title("🔮 Laboratorio de Clasificación Inteligente")
st.markdown(f"Motor activo: **<span style='color:#6366F1;'>{engine_choice}</span>**", unsafe_allow_html=True)
st.markdown("---")

col_input, col_result = st.columns([1, 1.2], gap="large")

final_img_tensor = None

with col_input:
    st.subheader("📸 Entrada de Datos")
    
    if input_mode == "Subir Archivo":
        uploaded_file = st.file_uploader("Arrastra imagen", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if uploaded_file:
            img_raw = Image.open(uploaded_file)
            st.image(img_raw, caption="Imagen Subida", use_container_width=True)
            
            img_gray = ImageOps.grayscale(img_raw)
            if np.mean(np.array(img_gray)) > 127: 
                img_gray = ImageOps.invert(img_gray)
            img_final = img_gray.resize((28, 28))
            
            final_img_tensor = np.array(img_final).astype('float32') / 255.0
            final_img_tensor = final_img_tensor.reshape(1, 28, 28, 1)

    elif input_mode == "Pizarra Natural (Negro sobre Blanco)":
        st.write("Dibuja tu número en **NEGRO** sobre el fondo **BLANCO**:")
        st.markdown('<div style="border: 2px solid #E2E8F0; border-radius: 4px; display: inline-block;">', unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=20,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas_natural",
            update_streamlit=True,
            display_toolbar=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if canvas_result.image_data is not None and np.any(canvas_result.image_data[:,:,:3] < 255):
            raw_draw = canvas_result.image_data.astype('uint8')
            img_pil = Image.fromarray(raw_draw).convert('L')
            img_inverted = ImageOps.invert(img_pil)
            img_final = img_inverted.resize((28, 28))
            
            final_img_tensor = np.array(img_final).astype('float32') / 255.0
            final_img_tensor = final_img_tensor.reshape(1, 28, 28, 1)

    elif input_mode == "Cámara en Vivo":
        st.markdown("<p style='font-size:0.85rem; color:#64748B;'>📸 <b>Tip:</b> Para mejores resultados, intenta que la prenda resalte sobre un fondo liso.</p>", unsafe_allow_html=True)
        
        camera_file = st.camera_input("Toma una foto de la prenda", label_visibility="collapsed")
        
        if camera_file:
            img_raw = Image.open(camera_file)
            
            img_gray = ImageOps.grayscale(img_raw)
            if np.mean(np.array(img_gray)) > 127: 
                img_gray = ImageOps.invert(img_gray)
            img_final = img_gray.resize((28, 28))
            
            final_img_tensor = np.array(img_final).astype('float32') / 255.0
            final_img_tensor = final_img_tensor.reshape(1, 28, 28, 1)

with col_result:
    st.subheader("⚡ Análisis Dinámico")
    
    if final_img_tensor is not None:
        if st.button("EJECUTAR INFERENCIA IA"):
            model = model_mnist if engine_choice == "Números (MNIST)" else model_fashion
            
            if model:
                preds = model.predict(final_img_tensor)
                
                if engine_choice == "Números (MNIST)":
                    labels = [str(i) for i in range(10)]
                else:
                    labels = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo', 
                              'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']
                
                top_idx = np.argmax(preds)
                confidence = np.max(preds)

                r1, r2 = st.columns(2)
                with r1:
                    st.markdown(f"""<div class='result-card'>
                        <small>PREDICCIÓN</small><h3>{labels[top_idx]}</h3></div>""", unsafe_allow_html=True)
                with r2:
                    st.markdown(f"""<div class='result-card'>
                        <small>CONFIANZA</small><h3>{confidence:.2%}</h3></div>""", unsafe_allow_html=True)

                st.markdown("---")
                
                df = pd.DataFrame({'Clase': labels, 'Prob': preds[0]})
                df = df.sort_values('Prob', ascending=True)
                
                fig = go.Figure(go.Bar(
                    x=df['Prob'], 
                    y=df['Clase'], 
                    orientation='h',
                    marker=dict(
                        color='rgba(99,102,241,0.8)', 
                        line=dict(color='#4F46E5', width=1)
                    ),
                    hovertemplate='<b>%{y}</b><br>Probabilidad: %{x:.2%}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=dict(text='Distribución de Probabilidades', font=dict(color='#0F172A', size=15)),
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#475569', size=12),
                    height=380, 
                    margin=dict(l=0, r=10, t=40, b=0),
                    xaxis=dict(showgrid=True, gridcolor='#E2E8F0', range=[0, 1], tickformat='.0%'),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Esperando entrada validada (archivo subido, dibujo detectado o foto capturada)...")
        st.markdown("""
        <div style="background: #F1F5F9; border: 1px dashed #CBD5E1; border-radius: 12px; padding: 2rem; text-align: center; color: #64748B;">
            <p style="font-size: 2.5rem; margin-bottom: 0.5rem;">📥</p>
            <strong>Proporciona una imagen</strong> en el panel izquierdo para iniciar el análisis de IA.
        </div>
        """, unsafe_allow_html=True)
