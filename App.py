# -*- coding: utf-8 -*-
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ============================================================
# ESTADO Y CONFIG
# ============================================================
st.set_page_config(
    page_title='Reconocimiento de Dígitos escritos a mano',
    page_icon='✍️',
    layout='centered'
)

# ============================================================
# ESTILO (tema oscuro + animaciones notorias, coherente con tus otras apps)
# ============================================================
st.markdown("""
<style>
:root{
  --bg:#0b1120; --bg2:#0f172a; --panel:#111827; --border:#1f2937;
  --text:#ffffff; --muted:#cbd5e1; --accent:#22d3ee; --accent2:#6366f1; --ok:#10b981;
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

/* Inputs (sliders) */
.stSlider [data-baseweb="slider"] > div > div{ background:#1e2b49 !important; }
.stSlider [role="slider"]{ background: var(--accent) !important; }

/* Canvas: borde con pulso */
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

/* Bloque de resultados */
.result-box{
  background:#0f172a; border:1px solid #24324a; border-radius:16px; padding:12px 14px;
}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

def card_start(): st.markdown('<div class="card">', unsafe_allow_html=True)
def card_end():   st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# MODELO: función (misma lógica, + retornamos probabilidades para el gráfico)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model("model/handwritten.h5")

def predictDigit(image):
    model = load_model()
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32') / 255.0
    # plt.imshow(img); plt.show()  # (evitamos abrir figura blocking)
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    result = int(np.argmax(pred[0]))
    probs = pred[0].astype(float)   # array de 10 probabilidades
    return result, probs

# ============================================================
# UI
# ============================================================
st.markdown("## ✍️ <span class='grad'>Reconocimiento de Dígitos escritos a mano</span>", unsafe_allow_html=True)
st.caption("Dibuja un dígito (0–9) en el panel, ajusta el grosor si quieres y presiona **Predecir**.")

# Controles + Canvas
card_start()
colL, colR = st.columns([1.05, 1])

with colL:
    st.markdown("#### 🎨 Lienzo")
    stroke_width = st.slider('Grosor del trazo', 4, 40, 18)
    st.markdown('<div class="canvas-wrap">', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0)",   # sin relleno de fondo
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",              # blanco
        background_color="#000000",          # negro
        height=300,
        width=300,
        key="canvas",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with colR:
    st.markdown("#### 👁️ Vista previa")
    if canvas_result.image_data is not None:
        # preview RGBA -> L
        arr = np.array(canvas_result.image_data, dtype=np.uint8)
        prev = Image.fromarray(arr, mode='RGBA').convert('L').resize((150,150))
        st.image(prev, caption="Previsualización (escala de grises)", use_container_width=False)
    else:
        st.info("Dibuja un dígito a la izquierda para ver la previsualización.")

card_end()

# Botón de predicción
card_start()
btn = st.button('🚀 Predecir', use_container_width=True)
predicted = False

if btn:
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        # No guardamos a disco; trabajamos en memoria
        res, probs = predictDigit(input_image)
        st.balloons()  # animación notoria 🎉

        st.markdown("#### ✅ Resultado")
        st.markdown(f"<div class='result-box'><strong>El dígito es:</strong> <span style='font-size:1.6rem;color:#a5f3fc'>{res}</span></div>", unsafe_allow_html=True)

        # Gráfico de probabilidades (0–9)
        st.markdown("#### 📊 Probabilidades por clase")
        import pandas as pd
        df = pd.DataFrame({
            "dígito": list(range(10)),
            "probabilidad": probs
        })
        st.bar_chart(df.set_index("dígito"))
        predicted = True
    else:
        st.warning('Por favor, dibuja en el lienzo antes de predecir.')
card_end()

# Sidebar informativa (imágenes más “notorias” a través de emojis y formato)
with st.sidebar:
    st.markdown("## ℹ️ Acerca de")
    st.markdown("- **Modelo**: CNN entrenada en dígitos escritos a mano (MNIST-like).")
    st.markdown("- **Entrada**: imagen 28×28 en escala de grises (normalizada a [0,1]).")
    st.markdown("- **Salida**: vector de 10 probabilidades (0–9).")
    st.markdown("---")
    st.markdown("### Consejos ✨")
    st.markdown("- Usa trazos **blancos** sobre fondo **negro**.")
    st.markdown("- Dibuja **grande** y centrado.")
    st.markdown("- Si el resultado es incierto, prueba a **engrosar** el trazo.")

# Footer
st.markdown("---")
st.caption("✍️ Reconocimiento de dígitos • Streamlit + TensorFlow — tema oscuro y animaciones visibles.")

