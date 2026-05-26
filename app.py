import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 1. CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Vision Lab Pro 2.0",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ocultar toolbar superior de Streamlit
st.markdown("""
<style>
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. CSS PROFESIONAL — DARK THEME CON CONTRASTE TOTAL
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── RESET GLOBAL ─────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

/* ── FONDO APP ────────────────────────────────────────── */
.stApp {
    background: #080B14 !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── SIDEBAR ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0D1120 !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #94A3B8 !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] h1 {
    color: #F8FAFC !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.4rem !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(99,102,241,0.25) !important;
}

/* ── OCULTAR TOOLTIP NATIVO STREAMLIT ─────────────────── */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
button[title],
[title="keyboard_double_arrow_right"],
div[class*="toolbar"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}
/* Eliminar cualquier tooltip flotante */
div[data-baseweb="tooltip"],
div[role="tooltip"] {
    display: none !important;
}

/* ── RADIO BUTTONS CORREGIDOS ─────────────────────────── */
[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 0.6rem !important;
}

[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: #0D1120 !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
}

[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    border-color: rgba(99,102,241,0.6) !important;
    background: #131929 !important;
}

[data-testid="stRadio"] label[data-baseweb="radio"] div,
[data-testid="stRadio"] label[data-baseweb="radio"] p {
    color: #E2E8F0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Resaltar la opción cuando está seleccionada */
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background: linear-gradient(135deg, rgba(79,70,229,0.2), rgba(124,58,237,0.2)) !important;
    border-color: #6366F1 !important;
    box-shadow: 0 0 15px rgba(99,102,241,0.15) !important;
}

[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* ── INFO BOX EN SIDEBAR ──────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(99,102,241,0.1) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
    color: #C7D2FE !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] p {
    color: #C7D2FE !important;
    font-size: 0.8rem !important;
}

/* ── TÍTULOS PRINCIPALES ──────────────────────────────── */
h1 {
    color: #F8FAFC !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(135deg, #F8FAFC 0%, #A5B4FC 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
h2, h3 {
    color: #E2E8F0 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

/* ── TEXTO GENERAL ────────────────────────────────────── */
p, span, label, div {
    color: #CBD5E1 !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── WRITE / CAPTION ──────────────────────────────────── */
[data-testid="stText"], .stMarkdown p {
    color: #94A3B8 !important;
}
.stCaption {
    color: #64748B !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
}

/* ── FILE UPLOADER ────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0D1120 !important;
    border: 1.5px dashed rgba(99,102,241,0.4) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.8) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span {
    color: #94A3B8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #A5B4FC !important;
}

/* ── BOTÓN PRINCIPAL ──────────────────────────────────── */
div.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    padding: 0.75rem 2.5rem !important;
    border-radius: 12px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.35) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.55) !important;
    background: linear-gradient(135deg, #7C3AED 0%, #6366F1 100%) !important;
}
div.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── MÉTRICAS ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #131929, #0F172A) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 16px !important;
    padding: 1.25rem 1.5rem !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stMetric"] label {
    color: #64748B !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
    background: linear-gradient(135deg, #A5B4FC, #C4B5FD) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

/* ── IMAGEN SUBIDA ────────────────────────────────────── */
[data-testid="stImage"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
}
[data-testid="stImage"] img {
    border-radius: 16px !important;
}
[data-testid="stImage"] + p {
    color: #64748B !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    text-align: center !important;
}

/* ── ERROR / WARNING ──────────────────────────────────── */
[data-testid="stAlert"][data-baseweb="notification"] {
    background: rgba(239,68,68,0.1) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 12px !important;
    color: #FCA5A5 !important;
}
[data-testid="stAlert"][data-baseweb="notification"] p {
    color: #FCA5A5 !important;
}

/* ── DIVISOR ──────────────────────────────────────────── */
hr {
    border-color: rgba(99,102,241,0.15) !important;
    margin: 2rem 0 !important;
}

/* ── SCROLLBAR CUSTOM ─────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080B14; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.7); }

/* ── PLOTLY CHART CONTAINER ───────────────────────────── */
[data-testid="stPlotlyChart"] {
    background: #0D1120 !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 20px !important;
    padding: 1rem !important;
    overflow: hidden !important;
}

/* ── COLUMNAS GAP ─────────────────────────────────────── */
[data-testid="stHorizontalBlock"] {
    gap: 2rem !important;
    align-items: flex-start !important;
}

/* ── SECTION LABEL ────────────────────────────────────── */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.25rem;
}
.section-label span.icon {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}
.section-label h3 {
    margin: 0 !important;
    font-size: 1rem !important;
    color: #E2E8F0 !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
}

/* ── STATUS BADGE ─────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    color: #6EE7B7 !important;
    padding: 4px 12px;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}
.status-dot {
    width: 6px;
    height: 6px;
    background: #10B981;
    border-radius: 50%;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── SIDEBAR LOGO AREA ────────────────────────────────── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.5rem;
}
.logo-mark {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. LÓGICA DE CARGA DE MODELOS
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
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-mark">🔮</div>
        <div>
            <div style="color:#F8FAFC;font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;line-height:1.2;">Vision Lab</div>
            <div style="color:#6366F1;font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;">PRO v2.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="border-top:1px solid rgba(99,102,241,0.2);margin-bottom:1.5rem;"></div>', unsafe_allow_html=True)

    st.markdown('<p style="color:#64748B!important;font-family:\'Space Mono\',monospace;font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.5rem;">CEREBRO ACTIVO</p>', unsafe_allow_html=True)

    engine_choice = st.radio(
        "Cerebro Activo:",
        options=["Números (MNIST)", "Moda (Fashion)"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)

    st.markdown('<div class="status-badge"><div class="status-dot"></div>SISTEMA ONLINE</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)

    st.info("💡 **Tip:** El sistema invierte automáticamente los colores si detecta un fondo blanco para máxima precisión.")

    st.markdown('<div style="border-top:1px solid rgba(99,102,241,0.2);margin-top:auto;padding-top:1rem;"></div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#334155!important;font-family:\'Space Mono\',monospace;font-size:0.65rem;">TensorFlow · Streamlit Cloud</p>', unsafe_allow_html=True)

# ============================================================
# 5. ENCABEZADO PRINCIPAL
# ============================================================
header_col, badge_col = st.columns([3, 1])
with header_col:
    st.title("🔮 Laboratorio de Clasificación Inteligente")
    st.markdown(f'<p style="color:#64748B;font-family:\'Space Mono\',monospace;font-size:0.8rem;margin-top:-0.5rem;">Motor activo → <span style="color:#A5B4FC;font-weight:700;">{engine_choice}</span></p>', unsafe_allow_html=True)

st.markdown('<div style="border-top:1px solid rgba(99,102,241,0.1);margin:1rem 0 2rem;"></div>', unsafe_allow_html=True)

# ============================================================
# 6. CUERPO PRINCIPAL
# ============================================================
col_input, col_result = st.columns([1, 1.3], gap="large")

with col_input:
    st.markdown("""
    <div class="section-label">
        <span class="icon">📸</span>
        <h3>Captura de Datos</h3>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Arrastra una imagen aquí",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        img_raw = Image.open(uploaded_file)
        st.image(img_raw, caption=f"📁 {uploaded_file.name}", use_container_width=True)
    else:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(99,102,241,0.05), rgba(139,92,246,0.05));
            border: 1.5px dashed rgba(99,102,241,0.25);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            margin-top: 0.5rem;
        ">
            <div style="font-size:2.5rem;margin-bottom:0.75rem;">🖼️</div>
            <p style="color:#475569!important;font-family:'Space Mono',monospace;font-size:0.78rem;line-height:1.6;">
                Arrastra una imagen PNG, JPG o JPEG<br>
                <span style="color:#334155!important;font-size:0.7rem;">Máx. 200MB</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

with col_result:
    st.markdown("""
    <div class="section-label">
        <span class="icon">⚡</span>
        <h3>Análisis Dinámico</h3>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file:
        if st.button("▶  Analizar Patrones"):
            if model_mnist and model_fashion:
                with st.spinner("Procesando imagen..."):
                    # --- PROCESAMIENTO ---
                    img_gray = ImageOps.grayscale(img_raw)

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
                        labels = ['T-shirt (Camiseta)', 'Trouser (Pantalón)', 'Pullover (Suéter)',
                                  'Dress (Vestido)', 'Coat (Abrigo)', 'Sandal (Sandalia)',
                                  'Shirt (Camisa)', 'Sneaker (Zapatilla)', 'Bag (Bolso)',
                                  'Ankle boot (Botín)']

                    # --- RESULTADOS ---
                    top_idx = np.argmax(preds)
                    confidence = np.max(preds)

                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#131929,#0F172A);border:1px solid rgba(99,102,241,0.25);border-radius:16px;padding:1.25rem 1.5rem;">
                            <div style="color:#64748B;font-family:'Space Mono',monospace;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">Predicción</div>
                            <div style="color:#A5B4FC;font-family:'Syne',sans-serif;font-weight:800;font-size:1.15rem;line-height:1.4;word-break:break-word;">{labels[top_idx]}</div>
                        </div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#131929,#0F172A);border:1px solid rgba(99,102,241,0.25);border-radius:16px;padding:1.25rem 1.5rem;">
                            <div style="color:#64748B;font-family:'Space Mono',monospace;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">Confianza</div>
                            <div style="color:#C4B5FD;font-family:'Syne',sans-serif;font-weight:800;font-size:1.15rem;">{confidence:.2%}</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

                    # --- GRÁFICO PLOTLY (DARK ELEGANTE) ---
                    df_chart = pd.DataFrame({'Clase': labels, 'Confianza': preds[0]})
                    df_chart = df_chart.sort_values('Confianza', ascending=True)

                    colors = ['rgba(99,102,241,0.25)'] * len(labels)
                    colors[df_chart['Clase'].tolist().index(labels[top_idx])] = 'rgba(139,92,246,1)'

                    fig = go.Figure(go.Bar(
                        x=df_chart['Confianza'],
                        y=df_chart['Clase'],
                        orientation='h',
                        marker=dict(
                            color=df_chart['Confianza'],
                            colorscale=[[0, 'rgba(99,102,241,0.15)'], [0.5, 'rgba(99,102,241,0.6)'], [1, 'rgba(139,92,246,1)']],
                            line=dict(color='rgba(139,92,246,0.3)', width=0.5),
                        ),
                        text=[f'{v:.1%}' for v in df_chart['Confianza']],
                        textposition='outside',
                        textfont=dict(color='#94A3B8', size=10, family='Space Mono'),
                        hovertemplate='<b>%{y}</b><br>Confianza: %{x:.2%}<extra></extra>',
                    ))

                    fig.update_layout(
                        title=dict(
                            text='Distribución de Probabilidades',
                            font=dict(color='#E2E8F0', size=13, family='Syne'),
                            x=0,
                        ),
                        paper_bgcolor='rgba(13,17,32,0)',
                        plot_bgcolor='rgba(13,17,32,0)',
                        height=400,
                        margin=dict(l=0, r=60, t=45, b=0),
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(99,102,241,0.08)',
                            tickformat='.0%',
                            tickfont=dict(color='#475569', size=9, family='Space Mono'),
                            zeroline=False,
                        ),
                        yaxis=dict(
                            tickfont=dict(color='#94A3B8', size=10, family='Syne'),
                            gridcolor='rgba(99,102,241,0.05)',
                        ),
                        hoverlabel=dict(
                            bgcolor='#0D1120',
                            bordercolor='rgba(99,102,241,0.4)',
                            font=dict(color='#E2E8F0', family='Syne'),
                        ),
                    )

                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("⚠️ No se detectaron los modelos en el repositorio.")
    else:
        st.markdown("""
        <div style="
            background: rgba(13,17,32,0.6);
            border: 1px solid rgba(99,102,241,0.15);
            border-radius: 20px;
            padding: 3rem 2rem;
            text-align: center;
            height: 280px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        ">
            <div style="font-size:2rem;margin-bottom:0.75rem;opacity:0.3;">⚡</div>
            <p style="color:#334155!important;font-family:'Space Mono',monospace;font-size:0.75rem;line-height:1.8;">
                Sube una imagen para<br>iniciar el análisis
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 7. FOOTER
# ============================================================
st.markdown('<div style="border-top:1px solid rgba(99,102,241,0.1);margin-top:3rem;padding-top:1.5rem;"></div>', unsafe_allow_html=True)

fc1, fc2, fc3 = st.columns(3)
with fc1:
    st.markdown('<p style="color:#334155!important;font-family:\'Space Mono\',monospace;font-size:0.65rem;">🔍 VISION LAB PRO © 2025</p>', unsafe_allow_html=True)
with fc2:
    st.markdown('<p style="color:#334155!important;font-family:\'Space Mono\',monospace;font-size:0.65rem;text-align:center;">POWERED BY TENSORFLOW</p>', unsafe_allow_html=True)
with fc3:
    st.markdown('<p style="color:#334155!important;font-family:\'Space Mono\',monospace;font-size:0.65rem;text-align:right;">STREAMLIT CLOUD</p>', unsafe_allow_html=True)
