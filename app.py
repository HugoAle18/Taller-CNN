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

st.markdown("""
<style>
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
.stApp { background: #080B14 !important; font-family: 'Syne', sans-serif !important; }

[data-testid="stSidebar"] {
    background: #0D1120 !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #94A3B8 !important; font-size: 0.85rem !important; }
[data-testid="stSidebar"] hr { border-color: rgba(99,102,241,0.25) !important; }

[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],
button[title],[title="keyboard_double_arrow_right"],div[class*="toolbar"] {
    display:none !important; visibility:hidden !important; opacity:0 !important;
}
div[data-baseweb="tooltip"],div[role="tooltip"] { display:none !important; }

[data-testid="stRadio"] > div { display:flex !important; flex-direction:column !important; gap:0.5rem !important; }
[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child { display:none !important; }
[data-testid="stRadio"] label {
    background:#0D1120 !important; border:1px solid rgba(99,102,241,0.2) !important;
    border-radius:10px !important; padding:0.6rem 1rem !important; cursor:pointer !important;
    font-family:'Space Mono',monospace !important; font-size:0.75rem !important;
    color:#475569 !important; width:100% !important; transition:all 0.25s ease !important;
}
[data-testid="stRadio"] label:hover { border-color:rgba(99,102,241,0.5) !important; color:#94A3B8 !important; background:#131929 !important; }
[data-testid="stRadio"] label:has(input:checked) {
    background:linear-gradient(135deg,#4F46E5,#7C3AED) !important;
    border-color:transparent !important; color:#FFFFFF !important;
    box-shadow:0 4px 20px rgba(99,102,241,0.45) !important; font-weight:700 !important;
}
[data-testid="stRadio"] label:has(input:checked) p,
[data-testid="stRadio"] label:has(input:checked) span { color:#FFFFFF !important; }

[data-testid="stSidebar"] [data-testid="stAlert"] {
    background:rgba(99,102,241,0.1) !important; border:1px solid rgba(99,102,241,0.3) !important;
    border-radius:12px !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] p { color:#C7D2FE !important; font-size:0.8rem !important; }

h1 {
    color:#F8FAFC !important; font-family:'Syne',sans-serif !important;
    font-weight:800 !important; font-size:2.2rem !important; letter-spacing:-0.03em !important;
    background:linear-gradient(135deg,#F8FAFC 0%,#A5B4FC 100%) !important;
    -webkit-background-clip:text !important; -webkit-text-fill-color:transparent !important;
    background-clip:text !important;
}
h2,h3 { color:#E2E8F0 !important; font-family:'Syne',sans-serif !important; font-weight:700 !important; }
p,span,label,div { color:#CBD5E1 !important; font-family:'Syne',sans-serif !important; }
[data-testid="stText"],.stMarkdown p { color:#94A3B8 !important; }

[data-testid="stFileUploader"] {
    background:#0D1120 !important; border:1.5px dashed rgba(99,102,241,0.4) !important;
    border-radius:16px !important; padding:1.5rem !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span { color:#94A3B8 !important; font-family:'Space Mono',monospace !important; font-size:0.8rem !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span { color:#A5B4FC !important; }

div.stButton > button {
    background:linear-gradient(135deg,#6366F1 0%,#8B5CF6 100%) !important;
    color:#FFFFFF !important; border:none !important; padding:0.75rem 2.5rem !important;
    border-radius:12px !important; font-family:'Space Mono',monospace !important;
    font-weight:700 !important; font-size:0.85rem !important; letter-spacing:0.05em !important;
    text-transform:uppercase !important; width:100% !important; margin-top:0.5rem !important;
    box-shadow:0 4px 24px rgba(99,102,241,0.35) !important; transition:all 0.3s ease !important;
}
div.stButton > button:hover {
    transform:translateY(-2px) !important; box-shadow:0 8px 32px rgba(99,102,241,0.55) !important;
    background:linear-gradient(135deg,#7C3AED 0%,#6366F1 100%) !important;
}

[data-testid="stPlotlyChart"] {
    background:#0D1120 !important; border:1px solid rgba(99,102,241,0.2) !important;
    border-radius:20px !important; padding:1rem !important; overflow:hidden !important;
}
[data-testid="stHorizontalBlock"] { gap:2rem !important; align-items:flex-start !important; }
[data-testid="stImage"] {
    border-radius:16px !important; overflow:hidden !important;
    border:1px solid rgba(99,102,241,0.2) !important; box-shadow:0 8px 32px rgba(0,0,0,0.4) !important;
}
[data-testid="stImage"] img { border-radius:16px !important; }

hr { border-color:rgba(99,102,241,0.15) !important; margin:2rem 0 !important; }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:#080B14; }
::-webkit-scrollbar-thumb { background:rgba(99,102,241,0.4); border-radius:3px; }

.section-label { display:flex; align-items:center; gap:0.6rem; margin-bottom:1.25rem; }
.section-label span.icon {
    width:32px; height:32px; background:linear-gradient(135deg,#6366F1,#8B5CF6);
    border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:1rem;
}
.section-label h3 { margin:0 !important; font-size:1rem !important; color:#E2E8F0 !important; font-weight:700 !important; }

.status-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3);
    color:#6EE7B7 !important; padding:4px 12px; border-radius:999px;
    font-family:'Space Mono',monospace; font-size:0.7rem; font-weight:700; letter-spacing:0.08em;
}
.status-dot { width:6px; height:6px; background:#10B981; border-radius:50%; animation:pulse-dot 2s infinite; }
@keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.3} }

.sidebar-logo { display:flex; align-items:center; gap:10px; margin-bottom:1.5rem; }
.logo-mark {
    width:40px; height:40px; background:linear-gradient(135deg,#6366F1,#8B5CF6);
    border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.2rem; flex-shrink:0;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background:#0D1120 !important; border-radius:12px !important;
    padding:4px !important; gap:4px !important; border:1px solid rgba(99,102,241,0.2) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background:transparent !important; color:#475569 !important;
    font-family:'Space Mono',monospace !important; font-size:0.75rem !important;
    border-radius:8px !important; padding:0.5rem 1rem !important; border:none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background:linear-gradient(135deg,#4F46E5,#7C3AED) !important; color:#FFFFFF !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none !important; }

/* Forzar fondo negro en el canvas drawable */
.canvas-container { border-radius: 12px !important; overflow: hidden !important; }
.canvas-container canvas {
    border-radius: 12px !important;
    border: 1.5px solid rgba(99,102,241,0.5) !important;
    cursor: crosshair !important;
}
/* Ocultar toolbar del canvas */
[data-testid="stCanvasToolbar"],
div[class*="canvasToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. MODELOS
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
# 4. FUNCIONES
# ============================================================
def run_inference(pil_image, engine):
    img_gray = ImageOps.grayscale(pil_image)
    if np.mean(np.array(img_gray)) > 120:
        img_gray = ImageOps.invert(img_gray)
    img_final  = img_gray.resize((28, 28))
    img_tensor = np.array(img_final).astype('float32') / 255.0
    img_tensor = img_tensor.reshape(1, 28, 28, 1)
    if engine == "Números (MNIST)":
        preds  = model_mnist.predict(img_tensor, verbose=0)
        labels = [str(i) for i in range(10)]
    else:
        preds  = model_fashion.predict(img_tensor, verbose=0)
        labels = ['T-shirt (Camiseta)','Trouser (Pantalón)','Pullover (Suéter)',
                  'Dress (Vestido)','Coat (Abrigo)','Sandal (Sandalia)',
                  'Shirt (Camisa)','Sneaker (Zapatilla)','Bag (Bolso)','Ankle boot (Botín)']
    return preds, labels

def show_results(preds, labels):
    top_idx    = np.argmax(preds)
    confidence = np.max(preds)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#131929,#0F172A);
                    border:1px solid rgba(99,102,241,0.25);border-radius:16px;padding:1.25rem 1.5rem;">
            <div style="color:#64748B;font-family:'Space Mono',monospace;font-size:0.68rem;
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">Predicción</div>
            <div style="color:#A5B4FC;font-family:'Syne',sans-serif;font-weight:800;
                        font-size:1.15rem;line-height:1.4;word-break:break-word;">{labels[top_idx]}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#131929,#0F172A);
                    border:1px solid rgba(99,102,241,0.25);border-radius:16px;padding:1.25rem 1.5rem;">
            <div style="color:#64748B;font-family:'Space Mono',monospace;font-size:0.68rem;
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">Confianza</div>
            <div style="color:#C4B5FD;font-family:'Syne',sans-serif;font-weight:800;font-size:1.15rem;">
                {confidence:.2%}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

    df_chart = pd.DataFrame({'Clase': labels, 'Confianza': preds[0]}).sort_values('Confianza', ascending=True)
    fig = go.Figure(go.Bar(
        x=df_chart['Confianza'], y=df_chart['Clase'], orientation='h',
        marker=dict(
            color=df_chart['Confianza'],
            colorscale=[[0,'rgba(99,102,241,0.15)'],[0.5,'rgba(99,102,241,0.6)'],[1,'rgba(139,92,246,1)']],
            line=dict(color='rgba(139,92,246,0.3)', width=0.5),
        ),
        text=[f'{v:.1%}' for v in df_chart['Confianza']],
        textposition='outside',
        textfont=dict(color='#94A3B8', size=10, family='Space Mono'),
        hovertemplate='<b>%{y}</b><br>Confianza: %{x:.2%}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='Distribución de Probabilidades',
                   font=dict(color='#E2E8F0', size=13, family='Syne'), x=0),
        paper_bgcolor='rgba(13,17,32,0)', plot_bgcolor='rgba(13,17,32,0)',
        height=400, margin=dict(l=0, r=60, t=45, b=0),
        xaxis=dict(showgrid=True, gridcolor='rgba(99,102,241,0.08)', tickformat='.0%',
                   tickfont=dict(color='#475569', size=9, family='Space Mono'), zeroline=False),
        yaxis=dict(tickfont=dict(color='#94A3B8', size=10, family='Syne'),
                   gridcolor='rgba(99,102,241,0.05)'),
        hoverlabel=dict(bgcolor='#0D1120', bordercolor='rgba(99,102,241,0.4)',
                        font=dict(color='#E2E8F0', family='Syne')),
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 5. SESSION STATE
# ============================================================
if "active_image"   not in st.session_state: st.session_state.active_image   = None
if "show_results"   not in st.session_state: st.session_state.show_results   = False
if "last_preds"     not in st.session_state: st.session_state.last_preds     = None
if "last_labels"    not in st.session_state: st.session_state.last_labels    = None
if "canvas_key"     not in st.session_state: st.session_state.canvas_key     = 0

# ============================================================
# 6. SIDEBAR
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
        index=0, label_visibility="collapsed"
    )
    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="status-badge"><div class="status-dot"></div>SISTEMA ONLINE</div>', unsafe_allow_html=True)
    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
    st.info("💡 **Tip:** El sistema invierte automáticamente los colores si detecta un fondo blanco para máxima precisión.")
    st.markdown('<div style="border-top:1px solid rgba(99,102,241,0.2);margin-top:auto;padding-top:1rem;"></div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#334155!important;font-family:\'Space Mono\',monospace;font-size:0.65rem;">TensorFlow · Streamlit Cloud</p>', unsafe_allow_html=True)

# ============================================================
# 7. ENCABEZADO
# ============================================================
header_col, _ = st.columns([3, 1])
with header_col:
    st.title("🔮 Laboratorio de Clasificación Inteligente")
    st.markdown(
        f'<p style="color:#64748B;font-family:\'Space Mono\',monospace;font-size:0.8rem;margin-top:-0.5rem;">'
        f'Motor activo → <span style="color:#A5B4FC;font-weight:700;">{engine_choice}</span></p>',
        unsafe_allow_html=True
    )
st.markdown('<div style="border-top:1px solid rgba(99,102,241,0.1);margin:1rem 0 2rem;"></div>', unsafe_allow_html=True)

# ============================================================
# 8. LAYOUT PRINCIPAL
# ============================================================
col_input, col_result = st.columns([1, 1.3], gap="large")

with col_input:
    st.markdown("""
    <div class="section-label">
        <span class="icon">📸</span>
        <h3>Captura de Datos</h3>
    </div>
    """, unsafe_allow_html=True)

    # ── MNIST ────────────────────────────────────────────────
    if engine_choice == "Números (MNIST)":
        tab_upload, tab_draw = st.tabs(["📁 Subir imagen", "✏️ Dibujar número"])

        # ---------- TAB SUBIR ----------
        with tab_upload:
            uploaded_file = st.file_uploader(
                "Sube tu imagen", type=["png","jpg","jpeg"],
                key="uploader_mnist", label_visibility="collapsed"
            )
            if uploaded_file:
                img = Image.open(uploaded_file).convert("RGB")
                st.session_state.active_image = img
                st.session_state.show_results = False
                st.image(img, caption=f"📁 {uploaded_file.name}", use_container_width=True)
            else:
                if st.session_state.active_image is None:
                    st.markdown("""
                    <div style="background:linear-gradient(135deg,rgba(99,102,241,0.05),rgba(139,92,246,0.05));
                        border:1.5px dashed rgba(99,102,241,0.25);border-radius:16px;
                        padding:3rem 2rem;text-align:center;margin-top:0.5rem;">
                        <div style="font-size:2.5rem;margin-bottom:0.75rem;">🖼️</div>
                        <p style="color:#475569!important;font-family:'Space Mono',monospace;font-size:0.78rem;line-height:1.6;">
                            Arrastra una imagen PNG, JPG o JPEG<br>
                            <span style="color:#334155!important;font-size:0.7rem;">Máx. 200MB</span>
                        </p>
                    </div>""", unsafe_allow_html=True)

        # ---------- TAB DIBUJAR ----------
        with tab_draw:
            # Instrucción
            st.markdown("""
            <p style="color:#64748B!important;font-family:'Space Mono',monospace;
               font-size:0.72rem;margin-bottom:0.75rem;">
               ✏️ Dibuja un dígito del 0 al 9 — trazo blanco sobre fondo negro.
            </p>""", unsafe_allow_html=True)

            # ── CANVAS — clave: update_streamlit=True hace que cada trazo
            #    actualice Python en tiempo real sin botón extra ──────────
            canvas_result = st_canvas(
                fill_color        = "rgba(0,0,0,0)",   # relleno transparente
                stroke_width      = 18,
                stroke_color      = "#FFFFFF",
                background_color  = "#000000",         # fondo negro explícito
                background_image  = None,
                update_streamlit  = True,              # actualiza en cada trazo
                height            = 350,
                width             = 350,
                drawing_mode      = "freedraw",
                key               = f"digit_canvas_{st.session_state.canvas_key}",
                display_toolbar   = False,             # oculta toolbar nativa
            )

            st.markdown('<div style="margin-top:0.5rem;"></div>', unsafe_allow_html=True)

            col_b1, col_b2 = st.columns(2)

            with col_b1:
                if st.button("🗑 Limpiar", key="btn_clear_canvas"):
                    # Incrementar la key fuerza un canvas nuevo (vacío)
                    st.session_state.canvas_key   += 1
                    st.session_state.active_image  = None
                    st.session_state.show_results  = False
                    st.session_state.last_preds    = None
                    st.session_state.last_labels   = None
                    st.rerun()

            with col_b2:
                # El botón solo aparece si hay algo dibujado
                hay_trazo = (
                    canvas_result is not None
                    and canvas_result.image_data is not None
                    and canvas_result.image_data.max() > 10
                )
                if hay_trazo:
                    if st.button("▶ Analizar dibujo", key="btn_analyze_draw"):
                        arr = canvas_result.image_data.astype(np.uint8)
                        pil = Image.fromarray(arr, mode="RGBA").convert("RGB")
                        st.session_state.active_image = pil
                        # Inferencia inmediata
                        if model_mnist and model_fashion:
                            preds, labels = run_inference(pil, engine_choice)
                            st.session_state.last_preds  = preds
                            st.session_state.last_labels = labels
                            st.session_state.show_results = True
                        st.rerun()

            # Badge estado
            if hay_trazo if 'hay_trazo' in dir() else False:
                st.markdown("""
                <div style="display:inline-flex;align-items:center;gap:6px;margin-top:8px;
                    background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);
                    padding:4px 12px;border-radius:999px;">
                    <div style="width:6px;height:6px;background:#10B981;border-radius:50%;"></div>
                    <span style="color:#6EE7B7!important;font-family:'Space Mono',monospace;
                          font-size:0.68rem;font-weight:700;">LISTO PARA ANALIZAR</span>
                </div>""", unsafe_allow_html=True)

    # ── FASHION ─────────────────────────────────────────────
    else:
        uploaded_file = st.file_uploader(
            "Sube tu imagen", type=["png","jpg","jpeg"],
            key="uploader_fashion", label_visibility="collapsed"
        )
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.session_state.active_image = img
            st.session_state.show_results = False
            st.image(img, caption=f"📁 {uploaded_file.name}", use_container_width=True)
        else:
            st.session_state.active_image = None
            st.markdown("""
            <div style="background:linear-gradient(135deg,rgba(99,102,241,0.05),rgba(139,92,246,0.05));
                border:1.5px dashed rgba(99,102,241,0.25);border-radius:16px;
                padding:3rem 2rem;text-align:center;margin-top:0.5rem;">
                <div style="font-size:2.5rem;margin-bottom:0.75rem;">🖼️</div>
                <p style="color:#475569!important;font-family:'Space Mono',monospace;font-size:0.78rem;line-height:1.6;">
                    Arrastra una imagen PNG, JPG o JPEG<br>
                    <span style="color:#334155!important;font-size:0.7rem;">Máx. 200MB</span>
                </p>
            </div>""", unsafe_allow_html=True)

# ============================================================
# 9. COLUMNA RESULTADOS
# ============================================================
with col_result:
    st.markdown("""
    <div class="section-label">
        <span class="icon">⚡</span>
        <h3>Análisis Dinámico</h3>
    </div>
    """, unsafe_allow_html=True)

    # Mostrar resultados guardados del canvas (persisten tras rerun)
    if st.session_state.show_results and st.session_state.last_preds is not None:
        show_results(st.session_state.last_preds, st.session_state.last_labels)

    elif st.session_state.active_image is not None:
        if st.button("▶  Analizar Patrones", key="btn_main_analyze"):
            if model_mnist and model_fashion:
                with st.spinner("Procesando imagen..."):
                    preds, labels = run_inference(st.session_state.active_image, engine_choice)
                    st.session_state.last_preds  = preds
                    st.session_state.last_labels = labels
                    st.session_state.show_results = True
                show_results(preds, labels)
            else:
                st.error("⚠️ No se detectaron los modelos.")

    else:
        st.markdown("""
        <div style="background:rgba(13,17,32,0.6);border:1px solid rgba(99,102,241,0.15);
             border-radius:20px;padding:3rem 2rem;text-align:center;height:280px;
             display:flex;flex-direction:column;align-items:center;justify-content:center;">
            <div style="font-size:2rem;margin-bottom:0.75rem;opacity:0.3;">⚡</div>
            <p style="color:#334155!important;font-family:'Space Mono',monospace;font-size:0.75rem;line-height:1.8;">
                Sube una imagen o dibuja un número<br>para iniciar el análisis
            </p>
        </div>""", unsafe_allow_html=True)

# ============================================================
# 10. FOOTER
# ============================================================
st.markdown('<div style="border-top:1px solid rgba(99,102,241,0.1);margin-top:3rem;padding-top:1.5rem;"></div>', unsafe_allow_html=True)
fc1, fc2, fc3 = st.columns(3)
with fc1:
    st.markdown('<p style="color:#334155!important;font-family:\'Space Mono\',monospace;font-size:0.65rem;">🔍 VISION LAB PRO © 2025</p>', unsafe_allow_html=True)
with fc2:
    st.markdown('<p style="color:#334155!important;font-family:\'Space Mono\',monospace;font-size:0.65rem;text-align:center;">POWERED BY TENSORFLOW</p>', unsafe_allow_html=True)
with fc3:
    st.markdown('<p style="color:#334155!important;font-family:\'Space Mono\',monospace;font-size:0.65rem;text-align:right;">STREAMLIT CLOUD</p>', unsafe_allow_html=True)
