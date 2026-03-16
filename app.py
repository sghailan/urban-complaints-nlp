# app.py -  app MUY simple para la demo ->  Agente Urbano Ghailan 
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))  # ← añade src/ al path

import streamlit as st
from PIL import Image

from topic_modeling.modelos import load_topic_models
from agent.agente_urbano import agente_urbano, formatear_para_chat, VisionConfig # AGENTE URBANO OJO

st.set_page_config(page_title="Urban Agent Ghailan", page_icon="🏙️")

# cargamos los  modelos 
@st.cache_resource
def get_models():
    return load_topic_models(str(BASE_DIR / "artefactos"))

topic_models = get_models()
vision_cfg = VisionConfig(enabled=True)

st.title(" Urban Agent Ghailan")
st.write("Escribe una incidencia y adjunta una imagen opcional.")

if "history" not in st.session_state:
    st.session_state.history = []

texto = st.text_area("Mensaje", height=100)
img_file = st.file_uploader("Imagen (opcional)", type=["png","jpg","jpeg","webp","avif"])

if st.button("Enviar"):
    image = None
    if img_file is not None:
        image = Image.open(img_file).convert("RGB")

    out = agente_urbano(
        mensaje=texto,
        topic_models=topic_models,
        image=image,
        vision_cfg=vision_cfg,
        top_k_temas=3
    )

    st.session_state.history.append(("Ciudadano", texto))
    st.session_state.history.append(("Agente", formatear_para_chat(out)))

    with st.expander("Ver JSON completo"):
        st.json(out)

st.divider()
for who, msg in st.session_state.history:
    st.markdown(f"**{who}:** {msg}")

# en la demo adjunto la imagen que esta en imagenes/basura.avif