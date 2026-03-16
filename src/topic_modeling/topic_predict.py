# topic_predict.py
import numpy as np
from preprocessing.text_cleaner import limpiar_normalizar_texto
from preprocessing.spacy_processor import procesar_spacy_lema

def preprocesar_texto(texto: str) -> str:
    """Normaliza + lematiza (usa tus funciones)."""
    texto_proc = limpiar_normalizar_texto(texto)
    texto_proc_lema = procesar_spacy_lema(texto_proc)
    return texto_proc_lema


def predecir_tema(
    texto: str,
    mode: str,
    tfidf=None,
    nmf=None,
    nombres_temas_auto=None,
    bow=None,
    lda=None,
    mapa_temas=None,
    top_k: int = 5,
    verbose: bool = True,
):
    """
    mode: 'nmf', 'lda' o 'both'
    Requiere pasar los objetos entrenados según el modo.
    Devuelve dict con resultados.
    """
    mode = mode.lower()
    if mode not in {"nmf", "lda", "both"}:
        raise ValueError("mode debe ser 'nmf', 'lda' o 'both'")

    texto_proc_lema = preprocesar_texto(texto)

    if verbose:
        print("Texto procesado:")
        print(texto_proc_lema)

    resultados = {"texto": texto, "texto_proc": texto_proc_lema}

    def _topk(dist, names_map, prefix):
        idx = np.argsort(dist)[::-1][:min(top_k, dist.shape[0])]
        out = []
        for i in idx:
            nombre = None
            if names_map is not None:
                nombre = names_map.get(int(i), f"Tema {prefix} {int(i)}")
            else:
                nombre = f"Tema {prefix} {int(i)}"
            out.append((int(i), float(dist[i]), nombre))
        return out

    # -------- NMF --------
    if mode in {"nmf", "both"}:
        if tfidf is None or nmf is None:
            raise ValueError("Para mode='nmf' necesitas pasar tfidf y nmf entrenados.")
        if nombres_temas_auto is None:
            nombres_temas_auto = {}

        X_new_tfidf = tfidf.transform([texto_proc_lema])
        W_new_nmf = nmf.transform(X_new_tfidf)[0]  # (n_topics,)

        tema_pred_nmf = int(W_new_nmf.argmax())
        conf_nmf = float(W_new_nmf.max())
        nombre_nmf = nombres_temas_auto.get(tema_pred_nmf, f"Tema NMF {tema_pred_nmf}")

        resultados["nmf"] = {
            "tema": tema_pred_nmf,
            "nombre": nombre_nmf,
            "conf": conf_nmf,
            "dist": W_new_nmf,
            "topk": _topk(W_new_nmf, nombres_temas_auto, "NMF"),
        }

        if verbose:
            print("\n[NMF]")
            print("Tema predicho:", tema_pred_nmf)
            print("Nombre del tema:", nombre_nmf)
            print("Confianza (peso máx):", round(conf_nmf, 3))

    # -------- LDA --------
    if mode in {"lda", "both"}:
        if bow is None or lda is None:
            raise ValueError("Para mode='lda' necesitas pasar bow y lda entrenados.")
        if mapa_temas is None:
            mapa_temas = {}

        X_new_bow = bow.transform([texto_proc_lema])
        W_new_lda = lda.transform(X_new_bow)[0]  # (n_topics,)

        tema_pred_lda = int(W_new_lda.argmax())
        conf_lda = float(W_new_lda.max())
        nombre_lda = mapa_temas.get(tema_pred_lda, f"Tema LDA {tema_pred_lda}")

        resultados["lda"] = {
            "tema": tema_pred_lda,
            "nombre": nombre_lda,
            "conf": conf_lda,
            "dist": W_new_lda,
            "topk": _topk(W_new_lda, mapa_temas, "LDA"),
        }

        if verbose:
            print("\n[LDA]")
            print("Tema predicho:", tema_pred_lda)
            print("Nombre del tema:", nombre_lda)
            print("Confianza (prob máx):", round(conf_lda, 3))

    return resultados
