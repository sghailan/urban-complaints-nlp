# text_cleaner.py
import re
import unicodedata
import pandas as pd
from wordfreq import zipf_frequency

# Expresiones regulares-> usar apuntes extresiones_regulares.md
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticonos
    "\U0001F300-\U0001F5FF"  # símbolos y pictogramas
    "\U0001F680-\U0001F6FF"  # transporte y símbolos
    "\U0001F1E0-\U0001F1FF"  # banderas
    "\u2600-\u26FF"          # símbolos misceláneos
    "\u2700-\u27BF"          # dingbats
    "]+", flags=re.UNICODE
)
ELONGATED_RE = re.compile(r"(.)\1{2,}")  

# tokens típicos de inglés / pruebas / seguridad-> limpieza
BLACKLIST = {
    "ssrf","xss","sqli","csrf","payload","exploit","vulnerability", "link"
    "test","testing","this","is","for","www", "life", "earth",  "from", "above", "water", "bertrand"
}

# préstamos útiles que NO se limpian -> no limpieza
WHITELIST = {"wifi","parking","email","web","internet","app","online"}

TOKEN_RE = re.compile(r"[a-zñ]+", flags=re.IGNORECASE)

def filtrar_palabras_no_es(texto: str, umbral=2.5) -> str:
    tokens = TOKEN_RE.findall(texto.lower())
    out = []
    for t in tokens:
        if t in BLACKLIST:
            continue
        if t in WHITELIST:
            out.append(t)
            continue
        if zipf_frequency(t, "es") >= umbral:
            out.append(t)
    return " ".join(out)

def ratio_espanol(texto: str, umbral=3) -> float:
    toks = texto.split()
    if not toks:
        return 0.0
    ok = sum((t in WHITELIST) or (zipf_frequency(t, "es") >= umbral) for t in toks)
    return ok / len(toks)

def quitar_tildes_sin_tocar_enye(texto: str) -> str:
    texto = texto.replace("ñ", "__ENYE__").replace("Ñ", "__ENYE_MAY__")
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = texto.replace("__ENYE__", "ñ").replace("__ENYE_MAY__", "Ñ")
    return texto

def limpiar_normalizar_texto(texto: str) -> str:
    if pd.isna(texto):
        return ""
    texto = str(texto)

    # Limpieza inicial
    texto = EMOJI_RE.sub(" ", texto)
    texto = HTML_TAG_RE.sub(" ", texto)
    texto = URL_RE.sub(" ", texto)
    texto = EMAIL_RE.sub(" ", texto)
    texto = MENTION_RE.sub(" ", texto)
    texto = HASHTAG_RE.sub(" ", texto)
    texto = ELONGATED_RE.sub(r"\1", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    # Normalización y filtrado limpio
    texto = texto.lower()
    texto = re.sub(r"\bal\b", "a el", texto)
    texto = re.sub(r"\bdel\b", "de el", texto)
    texto = quitar_tildes_sin_tocar_enye(texto)
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    texto = filtrar_palabras_no_es(texto, umbral=2.5)
    if ratio_espanol(texto, umbral=2.5) < 0.7:
        return ""
    
    return texto

def preprocesar_dataframe(df: pd.DataFrame, columna: str = "documento", min_palabras: int = 5) -> pd.DataFrame:
    df["texto_proc"] = df[columna].apply(limpiar_normalizar_texto)
    # Filtrado por longitud
    df = df[df["texto_proc"].str.split().apply(len) >= min_palabras].reset_index(drop=True)
    return df
