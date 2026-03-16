# modelos.py
import joblib
import json
from dataclasses import dataclass
from typing import Any, Dict

def cargar_mapa_int(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {int(k): v for k, v in d.items()}

@dataclass(frozen=True)
class TopicModels:
    tfidf: Any
    nmf: Any
    nombres_temas_auto: Dict[int, str]
    bow: Any
    lda: Any
    mapa_temas: Dict[int, str]

def load_topic_models(artefact_dir: str = "artefactos") -> TopicModels:
    tfidf = joblib.load(f"{artefact_dir}/tfidf.joblib")
    nmf   = joblib.load(f"{artefact_dir}/nmf.joblib")
    bow   = joblib.load(f"{artefact_dir}/bow.joblib")
    lda   = joblib.load(f"{artefact_dir}/lda.joblib")
    nombres_temas_auto = cargar_mapa_int(f"{artefact_dir}/nombres_temas_auto.json")
    mapa_temas         = cargar_mapa_int(f"{artefact_dir}/nombres_temas_lda.json")  # ← corregido
    return TopicModels(
        tfidf=tfidf,
        nmf=nmf,
        nombres_temas_auto=nombres_temas_auto,
        bow=bow,
        lda=lda,
        mapa_temas=mapa_temas,
    )