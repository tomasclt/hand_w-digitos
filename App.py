# -*- coding: utf-8 -*-
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title='Reconocimiento de Dígitos escritos a mano',
    page_icon='✍️',
    layout='centered'
)

# =========================
# ESTILOS (tema oscuro + sidebar)
# =========================
st.markdown("""
<style>
:root{
  --bg:#0b1120; --bg2:#0f172a; --panel:#111827; --border:#1f2937;
  --text:#ffffff; --muted:#cbd5e1; --accent:#22d3ee; --accent2:#6366f1;
}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(900px 500px at 85% 0%, rgba(99,102,241,.14), transparent 60%),
    radial-gradient(900px 500px at 10% 0%, rgba(34,211,238,.12), transparent 60%),
    linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%) !important;
  color: var(--text) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial;
}
main .block-container{ padding-top:1.8rem; padding-bottom:2.2rem; max-width:860px; }

h1,h2,h3{ color:#fff; letter-spacing:-.02em; }
h1 span.grad{
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}

/* Tarjetas */
.card{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1.2rem 1.4rem;
  box-shadow: 0 22px 60px rgba(0,0,0,.55);
  animation: fadeIn .55s ease;
}
@keyframes fadeIn{ from{opacity:0; transform: translateY(10px);} to{opacity:1; transform:none;} }

/* Botones */
.stButton > button{
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  border:0; color:#fff; font-weight:700; letter-spacing:.2px;
  border-radius:999px; padding:.8rem 1.25rem;
  box-shadow: 0 12px 38px rgba(99,102,241,.35);
  transition: transform .16s ease, box-shadow .16s ease, filter .16s ease;
}
.stButton > button:hover{ transform:translateY(-1px) scale(1.015); box-shadow: 0 18px 50px rgba(99,102,241,.5); filter: brightness(1.06); }

/* Slider */
.stSlider [data-baseweb="slider"] > div > div{ background:#1e2b49 !important; }
.stSlider [role="slider"]{ background: var(--accent) !important; }

/* Canvas con glow */
.canvas-wrap{
  border:1px solid #263349; border-radius:16px; padding:10px; background:#0f172a;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.02), 0 14px 40px rgba(0,0,0,.35);
  animation: glow 2.2s ease-in-out infinite;
}
@keyframes glow{
  0%   { box-shadow: inset 0 0 0 1px rgba(255,255,255,.02), 0 0 0 rgba(34,211,238,0.0); }
  50%  { box-shadow: inset 0 0 0 1px rgba(255,255,255,.05), 0 0 30px rgba(34,211,238,.25); }
  100% { box-shadow: inset 0 0 0 1px rgba(255,255,255,.02), 0 0 0 rgba(34,211,238,0.0); }
}

/* Bloque resultados */
.result-box{ background:#0f172a; border:1px solid #24324a; border-radius:16px; padding:12px 14px; }

/* === Sidebar oscuro y legible === */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0b1224 0%, #0f172a 100%) !important;
  color: #ffffff !important;
  border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div{ color:#ffffff !important; }

/* Card dentro del sidebar */
.sidecard{
  background:#111827; border:1px solid #1e293b; border-radius:16px;
  padding:16px 18px; box-shadow:0 16px 44px rgba(0,0,0,.45);
}
.sidecard ul{ margin:0; padding-left:1.2rem; }
.sidecard li{ color:#e5e7eb !important; margin:.55rem 0; }
.sidecard li strong{ color:#ffffff !important; }
.sidecard hr{ border:0; height:1px; background:#1e293b; margin:12px 0; }

footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

def card_start(): st.markdown('<div class="card">', unsafe_allow_html=True)
def card_end():   st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MODELO (cacheado)
# =========================
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model("model/handwritten.h5")

def predictDigit(image):
    model = load_model()
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32') / 255.0
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    result = int(np.argmax(pred[0]))
    probs = pred[0].astype(float)  # 10 probabilidades
    return result, probs

# =========================
# UI
# =========================
st.markdown("## ✍️ <span class='grad'>Reconocimiento de Dígitos escritos a mano</span>", unsafe_allow_html=True)
st.caption("Dibuja un dígito (0–9) y presiona **Predecir**.")

# Lienzo + preview
card_start()
colL, colR = st.columns([1.05, 1])

with colL:
    st.markdown("#### 🎨 Lienzo")
    stroke_width = st.slider('Grosor del trazo', 4, 40, 18)
    st.markdown('<div class="canvas-wrap">', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0)",   # sin relleno
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        key="canvas",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with colR:
    st.markdown("#### 👁️ Vista previa")
    if canvas_result.image_data is not None:
        arr = np.array(canvas_result.image_data, dtype=np.uint8)
        prev = Image.fromarray(arr, mode='RGBA').convert('L').resize((150,150))
        st.image(prev, caption="Escala de grises", use_container_width=False)
    else:
        st.info("Dibuja un dígito a la izquierda para previsualizar.")

card_end()

# Botón de predicción
card_start()
if st.button('🚀 Predecir', use_container_width=True):
    if canvas_result.image_data is not None:
        arr = np.array(canvas_result.image_data)
        input_image = Image.fromarray(arr.astype('uint8'), 'RGBA')
        digit, probs = predictDigit(input_image)
        st.balloons()

        st.markdown("#### ✅ Resultado")
        st.markdown(
            f"<div class='result-box'><strong>El dígito es:</strong> "
            f"<span style='font-size:1.6rem;color:#a5f3fc'>{digit}</span></div>",
            unsafe_allow_html=True
        )

        # Barras de probabilidades
        import pandas as pd
        df = pd.DataFrame({'dígito': list(range(10)), 'probabilidad': probs})
        st.markdown("#### 📊 Probabilidades por clase")
        st.bar_chart(df.set_index("dígito"))
    else:
        st.warning('Por favor, dibuja en el lienzo antes de predecir.')
card_end()

# =========================
# SIDEBAR (ahora sí, en el lugar correcto)
# =========================
with st.sidebar:
    st.markdown("""
    <div class="sidecard">
      <h3>ℹ️ Acerca de</h3>
      <ul>
        <li><strong>Modelo:</strong> CNN entrenada en dígitos escritos a mano (MNIST-like).</li>
        <li><strong>Entrada:</strong> imagen 28×28 en escala de grises (normalizada a [0,1]).</li>
        <li><strong>Salida:</strong> vector de 10 probabilidades (0–9).</li>
      </ul>
      <hr/>
      <h4>Consejos ✨</h4>
      <ul>
        <li>Trazos <strong>blancos</strong> sobre fondo <strong>negro</strong>.</li>
        <li>Dibuja <strong>grande</strong> y centrado.</li>
        <li>Si duda, aumenta el <strong>grosor</strong> del trazo.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("✍️ Reconocimiento de dígitos • Streamlit + TensorFlow — tema oscuro y animaciones visibles.")


