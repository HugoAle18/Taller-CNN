import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import plotly.graph_objects as go
import base64
import io

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
# 2. CSS PROFESIONAL
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
[data-testid="stSidebar"] h1 {
    color: #F8FAFC !important; font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important; font-size: 1.4rem !important; letter-spacing: -0.02em !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(99,102,241,0.25) !important; }

[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
button[title], [title="keyboard_double_arrow_right"], div[class*="toolbar"] {
    display: none !important; visibility: hidden !important; opacity: 0 !important;
}
div[data-baseweb="tooltip"], div[role="tooltip"] { display: none !important; }

[data-testid="stRadio"] > div { display: flex !important; flex-direction: column !important; gap: 0.5rem !important; }
[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child { display: none !important; }
[data-testid="stRadio"] label {
    background: #0D1120 !important; border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important; padding: 0.6rem 1rem !important; cursor: pointer !important;
    transition: all 0.25s ease !important; font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important; color: #475569 !important; width: 100% !important;
}
[data-testid="stRadio"] label:hover {
    border-color: rgba(99,102,241,0.5) !important; color: #94A3B8 !important; background: #131929 !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    border-color: transparent !important; color: #FFFFFF !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.45) !important; font-weight: 700 !important;
}
[data-testid="stRadio"] label:has(input:checked) p,
[data-testid="stRadio"] label:has(input:checked) span { color: #FFFFFF !important; }

[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(99,102,241,0.1) !important; border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 12px !important; color: #C7D2FE !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] p { color: #C7D2FE !important; font-size: 0.8rem !important; }

h1 {
    color: #F8FAFC !important; font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important; font-size: 2.2rem !important; letter-spacing: -0.03em !important;
    background: linear-gradient(135deg, #F8FAFC 0%, #A5B4FC 100%) !important;
    -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
h2, h3 { color: #E2E8F0 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
p, span, label, div { color: #CBD5E1 !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stText"], .stMarkdown p { color: #94A3B8 !important; }
.stCaption { color: #64748B !important; font-family: 'Space Mono', monospace !important; font-size: 0.72rem !important; }

[data-testid="stFileUploader"] {
    background: #0D1120 !important; border: 1.5px dashed rgba(99,102,241,0.4) !important;
    border-radius: 16px !important; padding: 1.5rem !important; transition: border-color 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(99,102,241,0.8) !important; }
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span { color: #94A3B8 !important; font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span { color: #A5B4FC !important; }

div.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    color: #FFFFFF !important; border: none !important; padding: 0.75rem 2.5rem !important;
    border-radius: 12px !important; font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important; font-size: 0.85rem !important; letter-spacing: 0.05em !important;
    text-transform: uppercase !important; transition: all 0.3s ease !important;
    width: 100% !important; margin-top: 0.5rem !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.35) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important; box-shadow: 0 8px 32px rgba(99,102,241,0.55) !important;
    background: linear-gradient(135deg, #7C3AED 0%, #6366F1 100%) !important;
}
div.stButton > button:active { transform: translateY(0px) !important; }

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #131929, #0F172A) !important;
    border: 1px solid rgba(99,102,241,0.25) !important; border-radius: 16px !important;
    padding: 1.25rem 1.5rem !important; backdrop-filter: blur(10px) !important;
}
[data-testid="stMetric"] label {
    color: #64748B !important; font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 0.1em !important;
}
[data-testid="stMetricValue"] {
    color: #F8FAFC !important; font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important; font-size: 2rem !important;
    background: linear-gradient(135deg, #A5B4FC, #C4B5FD) !important;
    -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

[data-testid="stImage"] {
    border-radius: 16px !important; overflow: hidden !important;
    border: 1px solid rgba(99,102,241,0.2) !important; box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
}
[data-testid="stImage"] img { border-radius: 16px !important; }
[data-testid="stImage"] + p {
    color: #64748B !important; font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important; text-align: center !important;
}

[data-testid="stAlert"][data-baseweb="notification"] {
    background: rgba(239,68,68,0.1) !important; border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 12px !important; color: #FCA5A5 !important;
}
[data-testid="stAlert"][data-baseweb="notification"] p { color: #FCA5A5 !important; }

hr { border-color: rgba(99,102,241,0.15) !important; margin: 2rem 0 !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080B14; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.7); }

[data-testid="stPlotlyChart"] {
    background: #0D1120 !important; border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 20px !important; padding: 1rem !important; overflow: hidden !important;
}
[data-testid="stHorizontalBlock"] { gap: 2rem !important; align-items: flex-start !important; }

.section-label { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1.25rem; }
.section-label span.icon {
    width: 32px; height: 32px; background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1rem;
}
.section-label h3 { margin: 0 !important; font-size: 1rem !important; color: #E2E8F0 !important; font-weight: 700 !important; letter-spacing: 0.02em !important; }

.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3);
    color: #6EE7B7 !important; padding: 4px 12px; border-radius: 999px;
    font-family: 'Space Mono', monospace; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
}
.status-dot { width: 6px; height: 6px; background: #10B981; border-radius: 50%; animation: pulse-dot 2s infinite; }
@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

.sidebar-logo { display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem; }
.logo-mark {
    width: 40px; height: 40px; background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; flex-shrink: 0;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #0D1120 !important; border-radius: 12px !important;
    padding: 4px !important; gap: 4px !important; border: 1px solid rgba(99,102,241,0.2) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important; color: #475569 !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important;
    border-radius: 8px !important; padding: 0.5rem 1rem !important; border: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important; color: #FFFFFF !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }

/* Ocultar el textarea puente */
[data-testid="stTextArea"] { display: none !important; }
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
# 4. FUNCIONES AUXILIARES
# ============================================================
def run_inference(pil_image, engine):
    img_gray = ImageOps.grayscale(pil_image)
    if np.mean(np.array(img_gray)) > 120:
        img_gray = ImageOps.invert(img_gray)
    img_final = img_gray.resize((28, 28))
    img_tensor = np.array(img_final).astype('float32') / 255.0
    img_tensor = img_tensor.reshape(1, 28, 28, 1)
    if engine == "Números (MNIST)":
        preds = model_mnist.predict(img_tensor)
        labels = [str(i) for i in range(10)]
    else:
        preds = model_fashion.predict(img_tensor)
        labels = ['T-shirt (Camiseta)', 'Trouser (Pantalón)', 'Pullover (Suéter)',
                  'Dress (Vestido)', 'Coat (Abrigo)', 'Sandal (Sandalia)',
                  'Shirt (Camisa)', 'Sneaker (Zapatilla)', 'Bag (Bolso)', 'Ankle boot (Botín)']
    return preds, labels

def show_results(preds, labels):
    top_idx   = np.argmax(preds)
    confidence = np.max(preds)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#131929,#0F172A);border:1px solid rgba(99,102,241,0.25);
                    border-radius:16px;padding:1.25rem 1.5rem;">
            <div style="color:#64748B;font-family:'Space Mono',monospace;font-size:0.68rem;
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">Predicción</div>
            <div style="color:#A5B4FC;font-family:'Syne',sans-serif;font-weight:800;
                        font-size:1.15rem;line-height:1.4;word-break:break-word;">{labels[top_idx]}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#131929,#0F172A);border:1px solid rgba(99,102,241,0.25);
                    border-radius:16px;padding:1.25rem 1.5rem;">
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
        title=dict(text='Distribución de Probabilidades', font=dict(color='#E2E8F0', size=13, family='Syne'), x=0),
        paper_bgcolor='rgba(13,17,32,0)', plot_bgcolor='rgba(13,17,32,0)',
        height=400, margin=dict(l=0, r=60, t=45, b=0),
        xaxis=dict(showgrid=True, gridcolor='rgba(99,102,241,0.08)', tickformat='.0%',
                   tickfont=dict(color='#475569', size=9, family='Space Mono'), zeroline=False),
        yaxis=dict(tickfont=dict(color='#94A3B8', size=10, family='Syne'), gridcolor='rgba(99,102,241,0.05)'),
        hoverlabel=dict(bgcolor='#0D1120', bordercolor='rgba(99,102,241,0.4)', font=dict(color='#E2E8F0', family='Syne')),
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 5. SESSION STATE
# ============================================================
for key, val in [("canvas_image", None), ("canvas_b64", ""), ("analyze_canvas", False)]:
    if key not in st.session_state:
        st.session_state[key] = val

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
    engine_choice = st.radio("Cerebro Activo:", options=["Números (MNIST)", "Moda (Fashion)"], index=0, label_visibility="collapsed")
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
    st.markdown(f'<p style="color:#64748B;font-family:\'Space Mono\',monospace;font-size:0.8rem;margin-top:-0.5rem;">Motor activo → <span style="color:#A5B4FC;font-weight:700;">{engine_choice}</span></p>', unsafe_allow_html=True)
st.markdown('<div style="border-top:1px solid rgba(99,102,241,0.1);margin:1rem 0 2rem;"></div>', unsafe_allow_html=True)

# ============================================================
# 8. CUERPO PRINCIPAL
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
                "Arrastra una imagen aquí", type=["png","jpg","jpeg"],
                key="uploader_mnist", label_visibility="collapsed"
            )
            if uploaded_file:
                img_raw = Image.open(uploaded_file)
                st.session_state.canvas_image  = None
                st.session_state.canvas_b64    = ""
                st.session_state.analyze_canvas = False
                st.image(img_raw, caption=f"📁 {uploaded_file.name}", use_container_width=True)
            else:
                st.markdown("""
                <div style="background:linear-gradient(135deg,rgba(99,102,241,0.05),rgba(139,92,246,0.05));
                    border:1.5px dashed rgba(99,102,241,0.25);border-radius:16px;padding:3rem 2rem;
                    text-align:center;margin-top:0.5rem;">
                    <div style="font-size:2.5rem;margin-bottom:0.75rem;">🖼️</div>
                    <p style="color:#475569!important;font-family:'Space Mono',monospace;font-size:0.78rem;line-height:1.6;">
                        Arrastra una imagen PNG, JPG o JPEG<br>
                        <span style="color:#334155!important;font-size:0.7rem;">Máx. 200MB</span>
                    </p>
                </div>""", unsafe_allow_html=True)

        # ---------- TAB DIBUJAR ----------
        with tab_draw:
            from streamlit.components.v1 import html as st_html

            # Canvas con botón "Analizar" integrado que envía la imagen via form oculto
            canvas_component = """
            <style>
              body { margin:0; background:transparent; }
              #wrap { display:flex; flex-direction:column; gap:10px; }
              canvas {
                background:#000; border-radius:12px; cursor:crosshair;
                border:1.5px solid rgba(99,102,241,0.5);
                touch-action:none; display:block;
              }
              .row { display:flex; gap:8px; width:280px; }
              button {
                padding:10px 0; border-radius:8px; font-family:'Space Mono',monospace;
                font-size:0.72rem; font-weight:700; cursor:pointer;
                transition:all 0.2s; border:none;
              }
              #btnClear {
                flex:1; background:rgba(99,102,241,0.08);
                color:#94A3B8; border:1px solid rgba(99,102,241,0.3) !important;
              }
              #btnClear:hover { background:rgba(99,102,241,0.18); color:#C7D2FE; }
              #btnAnalyze {
                flex:2.5;
                background: linear-gradient(135deg,#6366F1,#8B5CF6);
                color:#fff;
                box-shadow: 0 4px 16px rgba(99,102,241,0.4);
                display:none;
              }
              #btnAnalyze.visible { display:block; }
              #btnAnalyze:hover { background:linear-gradient(135deg,#7C3AED,#6366F1); }
              #hint {
                color:#475569; font-family:'Space Mono',monospace;
                font-size:0.68rem; height:14px; transition:opacity 0.3s;
              }
              textarea { display:none; }
            </style>

            <div id="wrap">
              <canvas id="c" width="280" height="280"></canvas>
              <div class="row">
                <button id="btnClear">🗑 Limpiar</button>
                <button id="btnAnalyze">▶ Analizar dibujo</button>
              </div>
              <div id="hint">Dibuja un dígito del 0 al 9</div>
              <textarea id="out"></textarea>
            </div>

            <script>
            (function(){
              const canvas  = document.getElementById('c');
              const ctx     = canvas.getContext('2d');
              const btnA    = document.getElementById('btnAnalyze');
              const btnC    = document.getElementById('btnClear');
              const hint    = document.getElementById('hint');
              const out     = document.getElementById('out');

              ctx.fillStyle = '#000';
              ctx.fillRect(0, 0, 280, 280);
              ctx.strokeStyle = '#fff';
              ctx.lineWidth   = 20;
              ctx.lineCap     = 'round';
              ctx.lineJoin    = 'round';

              let drawing = false, lx = 0, ly = 0, hasStrokes = false;

              function getPos(e) {
                const r   = canvas.getBoundingClientRect();
                const src = e.touches ? e.touches[0] : e;
                return [
                  (src.clientX - r.left) * (280 / r.width),
                  (src.clientY - r.top)  * (280 / r.height)
                ];
              }

              function onStroke() {
                if (!hasStrokes) {
                  hasStrokes = true;
                  btnA.classList.add('visible');
                  hint.style.opacity = '0';
                }
              }

              canvas.addEventListener('mousedown',  e => { drawing=true; [lx,ly]=getPos(e); });
              canvas.addEventListener('mousemove',  e => {
                if (!drawing) return;
                const [x,y] = getPos(e);
                ctx.beginPath(); ctx.moveTo(lx,ly); ctx.lineTo(x,y); ctx.stroke();
                [lx,ly] = [x,y];
                onStroke();
              });
              canvas.addEventListener('mouseup',    () => drawing=false);
              canvas.addEventListener('mouseleave', () => drawing=false);

              canvas.addEventListener('touchstart', e => {
                e.preventDefault(); drawing=true; [lx,ly]=getPos(e);
              }, {passive:false});
              canvas.addEventListener('touchmove', e => {
                e.preventDefault();
                if (!drawing) return;
                const [x,y]=getPos(e);
                ctx.beginPath(); ctx.moveTo(lx,ly); ctx.lineTo(x,y); ctx.stroke();
                [lx,ly]=[x,y]; onStroke();
              }, {passive:false});
              canvas.addEventListener('touchend', () => drawing=false);

              btnC.addEventListener('click', () => {
                ctx.fillStyle='#000'; ctx.fillRect(0,0,280,280);
                hasStrokes = false;
                btnA.classList.remove('visible');
                hint.style.opacity = '1';
                out.value = '';
                // Notifica vaciado
                window.parent.postMessage({type:'canvas_clear'}, '*');
              });

              btnA.addEventListener('click', () => {
                // Exporta y envía al padre
                const data = canvas.toDataURL('image/png');
                out.value  = data;
                window.parent.postMessage({type:'canvas_analyze', data: data}, '*');
                btnA.textContent = '⏳ Analizando...';
                btnA.style.opacity = '0.7';
                setTimeout(() => {
                  btnA.textContent = '▶ Analizar dibujo';
                  btnA.style.opacity = '1';
                }, 2000);
              });
            })();
            </script>
            """
            st_html(canvas_component, height=370)

            # ── Receptor Streamlit ──────────────────────────
            # Usamos st.query_params como puente de comunicación
            # El canvas envía postMessage → JS en receiver lo captura
            # → escribe en un text_input oculto → Streamlit lo lee

            receiver = """
            <script>
            (function(){
              window.addEventListener('message', function(e){
                if (!e.data) return;

                if (e.data.type === 'canvas_analyze') {
                  // Busca el textarea oculto de Streamlit con key canvas_bridge
                  const doc = window.parent.document;
                  const areas = doc.querySelectorAll('textarea');
                  areas.forEach(function(el){
                    if (el.getAttribute('aria-label') === 'canvas_bridge' ||
                        el.closest('[data-testid="stTextArea"]')) {
                      const setter = Object.getOwnPropertyDescriptor(
                        window.parent.HTMLTextAreaElement.prototype, 'value').set;
                      setter.call(el, e.data.data);
                      el.dispatchEvent(new Event('input', {bubbles:true}));
                    }
                  });
                  // Guarda también en localStorage como respaldo
                  try { localStorage.setItem('vlab_canvas', e.data.data); } catch(err){}
                }

                if (e.data.type === 'canvas_clear') {
                  try { localStorage.removeItem('vlab_canvas'); } catch(err){}
                }
              });
            })();
            </script>
            """
            st_html(receiver, height=0)

            # ── Text area puente (oculta via CSS) ──────────
            canvas_b64 = st.text_area("canvas_bridge", key="canvas_bridge", label_visibility="hidden", height=68)

            # ── Botón nativo Streamlit que procesa el base64 ──
            if st.button("▶ Confirmar y analizar dibujo", key="btn_confirm_canvas"):
                raw = st.session_state.get("canvas_bridge", "").strip()
                if raw and raw.startswith("data:image"):
                    _, b64data = raw.split(",", 1)
                    img_bytes  = base64.b64decode(b64data)
                    pil_img    = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    st.session_state.canvas_image    = pil_img
                    st.session_state.analyze_canvas  = True
                else:
                    st.warning("Dibuja algo y pulsa ▶ Analizar dibujo en el canvas primero.")

            if st.session_state.canvas_image is not None:
                st.image(st.session_state.canvas_image, caption="✏️ Dibujo listo para análisis", use_container_width=True)

        # Imagen activa MNIST
        if uploaded_file:
            active_image = Image.open(uploaded_file)
        elif st.session_state.canvas_image is not None:
            active_image = st.session_state.canvas_image
        else:
            active_image = None

    # ── FASHION ─────────────────────────────────────────────
    else:
        uploaded_file = st.file_uploader(
            "Arrastra una imagen aquí", type=["png","jpg","jpeg"],
            key="uploader_fashion", label_visibility="collapsed"
        )
        if uploaded_file:
            active_image = Image.open(uploaded_file)
            st.image(active_image, caption=f"📁 {uploaded_file.name}", use_container_width=True)
        else:
            active_image = None
            st.markdown("""
            <div style="background:linear-gradient(135deg,rgba(99,102,241,0.05),rgba(139,92,246,0.05));
                border:1.5px dashed rgba(99,102,241,0.25);border-radius:16px;padding:3rem 2rem;
                text-align:center;margin-top:0.5rem;">
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

    if active_image is not None:
        # Auto-analizar si viene del canvas con el botón interno
        if st.session_state.get("analyze_canvas") and st.session_state.canvas_image is not None:
            if model_mnist and model_fashion:
                with st.spinner("Procesando imagen..."):
                    preds, labels = run_inference(active_image, engine_choice)
                    show_results(preds, labels)
                st.session_state.analyze_canvas = False
            else:
                st.error("⚠️ No se detectaron los modelos en el repositorio.")
        else:
            if st.button("▶  Analizar Patrones"):
                if model_mnist and model_fashion:
                    with st.spinner("Procesando imagen..."):
                        preds, labels = run_inference(active_image, engine_choice)
                        show_results(preds, labels)
                else:
                    st.error("⚠️ No se detectaron los modelos en el repositorio.")
    else:
        st.markdown("""
        <div style="background:rgba(13,17,32,0.6);border:1px solid rgba(99,102,241,0.15);
             border-radius:20px;padding:3rem 2rem;text-align:center;height:280px;
             display:flex;flex-direction:column;align-items:center;justify-content:center;">
            <div style="font-size:2rem;margin-bottom:0.75rem;opacity:0.3;">⚡</div>
            <p style="color:#334155!important;font-family:'Space Mono',monospace;font-size:0.75rem;line-height:1.8;">
                Sube una imagen o dibuja un número<br>para iniciar el análisis
            </p>
        </div>
        """, unsafe_allow_html=True)

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
