import spacy
import re

from .text_cleaner import quitar_tildes_sin_tocar_enye # desde el mismo paq

nlp = spacy.load("es_core_news_sm")

stopwords_spacy = set(nlp.Defaults.stop_words)

stopwords_sociales = {
    "hola","buenos","buenas","dias","tardes","noches",
    "gracias","muchas","muchisimas","favor","porfavor",
    "saludos","saludo","cordial","cordiales","aprecio","agradecer",
    "estimado","estimada","señor","señora","sr","sra","atentamente",
    "pues","bueno","vale","ok","okay",
}

stopwords_dominio = {
    "madrid","madrileño","madrileña","ciudad",
    "ayuntamiento","municipal","ciudadano","ciudadanos",
    "persona","personas",
    "propuesta","propuestas",
    "vecino","vecinos","barrio","barrios"
}

stopwords_metadatos = {
    "enlace","enlazar","externo","listado","relativo",
    "repetido","complementario","fichero","merito",
    "originario","dejo","apoyo",  "forma", "medida"
}

# tokens basura concretos que se han colado en temas en prueba y error(no recomendable añadir así las cosas-> mejora temática)
stopwords_basura = {"vehicu", "enlacir", "line"}

stopwords_articulos = {"el","la","los","las","un","una","unos","unas"} 
# por claridad

stopwords_totales = (
    stopwords_spacy
    | stopwords_sociales
    | stopwords_dominio
    | stopwords_metadatos
    | stopwords_articulos
    | stopwords_basura
)

for w in stopwords_totales:
    nlp.vocab[w].is_stop = True

VOCAL_RE = re.compile(r"[aeiouáéíóúü]", re.IGNORECASE)

def token_valido(w: str) -> bool:
    # mínimo tamaño
    if len(w) <= 3:
        return False
    # descarta tokens sin vocal (basura)
    if not VOCAL_RE.search(w):
        return False
    return True

def procesar_spacy_lema(texto):
    doc = nlp(texto)
    out = []
    for tok in doc:
        if not tok.is_alpha or tok.is_stop:
            continue
        w = quitar_tildes_sin_tocar_enye(tok.lemma_.lower())
        # vuelve a mirar stopword tras normalizar (por si cambian tildes)
        if w in stopwords_totales:
            continue
        if not token_valido(w):
            continue
        out.append(w)
    return " ".join(out)

# eficiencia:
def procesar_lista_textos(textos, batch_size=200, n_process=1):
    resultados = []

    for doc in nlp.pipe(textos, batch_size=batch_size, n_process=n_process):
        out = []
        for tok in doc:
            if not tok.is_alpha or tok.is_stop:
                continue
            w = quitar_tildes_sin_tocar_enye(tok.lemma_.lower())
            if w in stopwords_totales:
                continue
            if not token_valido(w):
                continue
            out.append(w)
        resultados.append(" ".join(out))

    return resultados

