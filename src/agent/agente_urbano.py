# agente_urbano.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import json
import warnings
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# from openai import OpenAI
from groq import Groq
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration

from topic_modeling.modelos import TopicModels, load_topic_models
from topic_modeling.topic_predict import predecir_tema

warnings.filterwarnings("ignore")

client = Groq()
BETO_MODEL = "finiteautomata/beto-sentiment-analysis"

# Cache global para no reinstanciar
_SENTIMENT_PIPELINE = None
_BLIP_PROCESSOR = None
_BLIP_MODEL = None


# ── Módulo visual (BLIP) ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class VisionConfig:
    """Configura el módulo visual (opcional)."""
    enabled: bool = False


def get_image_caption_pipeline(model_name: str = "Salesforce/blip-image-captioning-base"):
    """Carga (solo una vez) el procesador y modelo BLIP."""
    global _BLIP_PROCESSOR, _BLIP_MODEL
    if _BLIP_MODEL is None:
        _BLIP_PROCESSOR = BlipProcessor.from_pretrained(model_name)
        _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _BLIP_MODEL = _BLIP_MODEL.to(device)
    return _BLIP_PROCESSOR, _BLIP_MODEL


# ── Módulo de sentimiento (BETO) ──────────────────────────────────────────────

def get_beto_pipeline():
    """Carga (solo una vez) el pipeline de BETO sentimiento."""
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None:
        _SENTIMENT_PIPELINE = pipeline(
            "sentiment-analysis",
            model=BETO_MODEL,
            tokenizer=BETO_MODEL,
            top_k=None,        # devuelve los 3 scores (NEG, NEU, POS)
            truncation=True,
            framework="pt",
            dtype=None,
        )
    return _SENTIMENT_PIPELINE


# ── 1) Features: temática ─────────────────────────────────────────────────────

def compute_topic_features(
    texto: str,
    models: TopicModels,
    top_k: int = 3,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Ejecuta la predicción temática con NMF + LDA."""
    res = predecir_tema(
        texto,
        mode="both",
        tfidf=models.tfidf,
        nmf=models.nmf,
        nombres_temas_auto=models.nombres_temas_auto,
        bow=models.bow,
        lda=models.lda,
        mapa_temas=models.mapa_temas,
        top_k=top_k,
        verbose=verbose,
    )

    nmf_res = res.get("nmf", {})
    lda_res = res.get("lda", {})

    out: Dict[str, Any] = {
        "top_k": top_k,
        "nmf": {
            "tema_id": nmf_res.get("tema"),
            "nombre":  nmf_res.get("nombre"),
            "conf":    nmf_res.get("conf"),
            "topk":    nmf_res.get("topk", []),
        },
        "lda": {
            "tema_id": lda_res.get("tema"),
            "nombre":  lda_res.get("nombre"),
            "conf":    lda_res.get("conf"),
            "topk":    lda_res.get("topk", []),
        },
    }

    # priorizar LDA si existe, si no NMF
    out["tema_principal"] = out["lda"]["nombre"] or out["nmf"]["nombre"]
    return out


# ── 2) Features: sentimiento (BETO) ──────────────────────────────────────────

def sentiment_features(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convierte la salida del modelo de sentimiento en variables interpretables.
    scores: [{'label':'NEG','score':...}, {'label':'NEU','score':...}, {'label':'POS','score':...}]
    """
    scores_dict = {d["label"]: float(d["score"]) for d in scores}
    best = max(scores, key=lambda d: d["score"])
    neg_score = scores_dict.get("NEG", 0.0)

    if neg_score >= 0.80:
        emotion = "muy_negativo"
    elif neg_score >= 0.50:
        emotion = "negativo"
    elif neg_score >= 0.30:
        emotion = "ligeramente_negativo"
    else:
        emotion = "neutro"

    return {
        "sentiment_label":     best["label"],
        "sentiment_score":     float(best["score"]),
        "sentiment_neg_score": neg_score,
        "sentiment_neu_score": scores_dict.get("NEU", 0.0),
        "sentiment_pos_score": scores_dict.get("POS", 0.0),
        "emotion_bucket":      emotion,
    }


def compute_beto_features(texto: str) -> Dict[str, Any]:
    """Ejecuta BETO y devuelve un dict listo para integrar."""
    sent = get_beto_pipeline()
    raw = sent(texto)[0]
    feats = sentiment_features(raw)
    return {
        "modelo":     BETO_MODEL,
        "raw_scores": raw,
        "features":   feats,
    }


# ── 3) Módulo visual (BLIP) ───────────────────────────────────────────────────

def compute_image_caption_optional(
    image: Optional[Any],
    vision_cfg: VisionConfig,
) -> Optional[Dict[str, Any]]:
    """
    Si hay imagen y el módulo está habilitado, genera una descripción
    automática usando BLIP (image captioning).
    """
    if not vision_cfg.enabled or image is None:
        return None

    processor, blip_model = get_image_caption_pipeline()
    device = next(blip_model.parameters()).device
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return {
        "caption": caption,
        "modelo":  "Salesforce/blip-image-captioning-base",
    }


# ── 4) Contexto para el LLM ───────────────────────────────────────────────────

def build_llm_context(
    texto_usuario: str,
    tematica: Dict[str, Any],
    sentimiento: Dict[str, Any],
    vision: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Estructura estable que el LLM recibe siempre."""
    ctx = {
        "texto_usuario": texto_usuario,
        "tematica":      tematica,
        "sentimiento":   sentimiento,
    }
    if vision is not None:
        ctx["vision"] = vision
    return ctx


# ── 5) Agente LLM (orquestador principal) ────────────────────────────────────

def agente_urbano(
    mensaje: str,
    topic_models: TopicModels,
    image: Optional[Any] = None,
    vision_cfg: VisionConfig = VisionConfig(enabled=False),
    top_k_temas: int = 3,
    llm_model: str = "llama-3.3-70b-versatile",
) -> Dict[str, Any]:
    """
    Orquestador principal del agente.
    Devuelve JSON completo: entrada + contexto + salida del LLM.
    """
    tematica   = compute_topic_features(mensaje, topic_models, top_k=top_k_temas, verbose=False)
    sentimiento = compute_beto_features(mensaje)
    vision     = compute_image_caption_optional(image=image, vision_cfg=vision_cfg)

    ctx = build_llm_context(
        texto_usuario=mensaje,
        tematica=tematica,
        sentimiento=sentimiento,
        vision=vision,
    )

    system_prompt = (
        "Eres un agente municipal para gestión de incidencias urbanas. "
        "Recibirás un JSON con: texto del usuario, clasificación temática (top-k) "
        "y análisis de sentimiento (y opcionalmente info visual). Tu tarea es: "
        "1) redactar una respuesta empática y útil al ciudadano, "
        "2) indicar si faltan datos (ubicación, referencia, foto, etc.), "
        "3) proponer prioridad (baja/media/alta) basada en el contexto disponible, "
        "4) generar un resumen breve para supervisión si es necesario. "
        "Devuelve SIEMPRE un JSON con las claves: "
        "estado, respuesta_ciudadano, prioridad, datos_faltantes, resumen_para_supervision."
    )

    completion = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": json.dumps(ctx, ensure_ascii=False)},
        ],
        temperature=0.3,
    )

    raw_text = completion.choices[0].message.content or ""
    try:
        llm_out = json.loads(raw_text)
    except json.JSONDecodeError:
        llm_out = {
            "estado": "escalar",
            "respuesta_ciudadano": (
                "Gracias por tu aviso. ¿Podrías indicar la ubicación exacta "
                "para gestionarlo correctamente?"
            ),
            "prioridad":               "media",
            "datos_faltantes":         ["ubicacion_exacta"],
            "resumen_para_supervision": "Salida no-JSON del LLM; se requiere revisión.",
            "raw_llm":                 raw_text,
        }

    return {
        "input_usuario": mensaje,
        "contexto":      ctx,
        "salida_llm":    llm_out,
    }


def formatear_para_chat(out: Dict[str, Any]) -> str:
    """Extrae el texto para el ciudadano desde la salida completa del agente."""
    return out.get("salida_llm", {}).get("respuesta_ciudadano", "").strip()


# ── 6) Ejecución de prueba ────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent

    topic_models = load_topic_models(str(BASE_DIR.parent.parent / "artefactos"))

    mensaje = "Está todo lleno de mierda, nadie limpia nada"

    image_path = BASE_DIR.parent.parent / "data" / "imagenes" / "basura.avif"
    image = Image.open(image_path).convert("RGB")

    vision_cfg = VisionConfig(enabled=True)

    out = agente_urbano(
        mensaje=mensaje,
        topic_models=topic_models,
        image=image,
        vision_cfg=vision_cfg,
        top_k_temas=3,
    )

    print("=== JSON COMPLETO (para sistema/supervisión) ===")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    print("\n=== TEXTO PARA EL CIUDADANO ===")
    print(formatear_para_chat(out))

    if out["salida_llm"].get("estado") == "escalar":
        print("\n=== PARA SUPERVISIÓN / EQUIPO ===")
        print(out["salida_llm"].get("resumen_para_supervision"))
        print("Datos faltantes:", out["salida_llm"].get("datos_faltantes"))

    # guardar salida en outputs/
    outputs_dir = BASE_DIR.parent.parent / "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = outputs_dir / "agente_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSalida guardada en: {output_path}")