# similitud_semantica_iu.py
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

import ipywidgets as widgets
from IPython.display import display, clear_output

from topic_modeling.topic_predict import predecir_tema  # ← corregido

def doc_vector(tokens, model):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def build_ui(
    df_path="data/df_sugerencias_controlada_temas.csv",
    topk=10,
    # modelos ya entrenados (obligatorios para texto nuevo)
    tfidf=None, nmf=None, nombres_temas_auto=None,
    bow=None, lda=None, mapa_temas=None,
):
    # -------- cargar df --------
    df = pd.read_csv(df_path)

    # columnas reales
    COL_TITLE = "title"
    COL_SHOW = "title"
    COL_NMF_ID = "tema_nmf"
    COL_NMF_NAME = "tema_nombre_nmf"
    COL_LDA_ID = "tema_lda"
    COL_LDA_NAME = "tema_lda_nombre"
    COL_LDA_CONF = "tema_lda_conf"

    # -------- preparar corpus w2v --------
    X_text = df["texto_proc_lema"].astype(str).values
    sentences = [doc.split() for doc in X_text]

    w2v = Word2Vec(
        sentences=sentences,
        vector_size=200,
        window=5,
        min_count=2,
        workers=4,
        sg=0,
        negative=10,
        epochs=15,
        seed=42
    )

    X_w2v = np.vstack([doc_vector(s, w2v) for s in sentences])
    X_w2v_norm = normalize(X_w2v)

    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(X_w2v_norm)

    def buscar_similares_por_vec(vec_norm, top_k=topk, excluir_idx=None):
        n_neighbors = top_k + 1 if excluir_idx is not None else top_k
        distances, indices = nn.kneighbors(vec_norm.reshape(1, -1), n_neighbors=n_neighbors)

        indices = indices[0]
        sims = 1 - distances[0]

        if excluir_idx is not None:
            mask = indices != excluir_idx
            indices = indices[mask][:top_k]
            sims = sims[mask][:top_k]
        else:
            indices = indices[:top_k]
            sims = sims[:top_k]
        return indices, sims

    def resultados_df(indices, sims):
        res = df.iloc[indices].copy()
        res = res.reset_index().rename(columns={"index": "doc_index"})
        res["cosine_sim"] = sims

        cols = ["doc_index", "cosine_sim", COL_NMF_NAME, COL_LDA_NAME, COL_SHOW]
        cols = [c for c in cols if c in res.columns]
        return res[cols].sort_values("cosine_sim", ascending=False)

    def print_existente(row):
        print("DOCUMENTO CONSULTA (existente):")
        print(row.get(COL_TITLE, ""))

        if COL_NMF_ID in row.index and COL_NMF_NAME in row.index:
            print("\n[NMF] (df)")
            print("Tema:", int(row[COL_NMF_ID]), "|", row[COL_NMF_NAME])

        if COL_LDA_ID in row.index and COL_LDA_NAME in row.index:
            print("\n[LDA] (df)")
            conf_txt = ""
            if COL_LDA_CONF in row.index and pd.notna(row[COL_LDA_CONF]):
                conf_txt = f" | conf: {round(float(row[COL_LDA_CONF]), 3)}"
            print("Tema:", int(row[COL_LDA_ID]), "|", row[COL_LDA_NAME] + conf_txt)

    def print_nuevo(res_pred):
        print("\n[NMF] (pred)")
        print("Tema:", res_pred["nmf"]["tema"], "|", res_pred["nmf"]["nombre"], "| conf:", round(res_pred["nmf"]["conf"], 3))
        print("Top-3:", res_pred["nmf"]["topk"])

        print("\n[LDA] (pred)")
        print("Tema:", res_pred["lda"]["tema"], "|", res_pred["lda"]["nombre"], "| conf:", round(res_pred["lda"]["conf"], 3))
        print("Top-3:", res_pred["lda"]["topk"])

    # -------- Widgets --------
    df_aux = df.copy()
    df_aux["__label__"] = df_aux.index.astype(str) + " | " + df_aux[COL_TITLE].astype(str).str.slice(0, 90)

    selector = widgets.Dropdown(
        options=[("— Selecciona un documento existente —", None)] +
                list(zip(df_aux["__label__"].tolist(), df_aux.index.tolist())),
        description="Doc:",
        layout=widgets.Layout(width="95%")
    )

    txt_new = widgets.Textarea(
        value="",
        placeholder="O pega aquí un texto nuevo…",
        description="Nuevo:",
        layout=widgets.Layout(width="95%", height="110px")
    )

    btn_doc = widgets.Button(description=f"Buscar similares (doc existente) TOP-{topk}", button_style="primary")
    btn_new = widgets.Button(description=f"Buscar similares (texto nuevo) TOP-{topk}", button_style="success")

    out = widgets.Output()

    btn_doc._click_handlers.callbacks.clear()
    btn_new._click_handlers.callbacks.clear()

    def on_doc_click(_):
        with out:
            out.clear_output(wait=True)

            if selector.value is None:
                print("Selecciona un documento existente.")
                return

            doc_idx = int(selector.value)
            row = df.loc[doc_idx]

            print_existente(row)

            vec_norm = X_w2v_norm[doc_idx]
            idx, sims = buscar_similares_por_vec(vec_norm, top_k=topk, excluir_idx=doc_idx)

            print(f"\nTOP-{topk} SIMILARES (Word2Vec + coseno):\n")
            display(resultados_df(idx, sims))

    def on_new_click(_):
        with out:
            out.clear_output(wait=True)

            texto_nuevo = txt_new.value.strip()
            if not texto_nuevo:
                print("Pega un texto nuevo.")
                return

            # chequeo de modelos
            missing = []
            for name, obj in [("tfidf", tfidf), ("nmf", nmf), ("nombres_temas_auto", nombres_temas_auto),
                              ("bow", bow), ("lda", lda), ("mapa_temas", mapa_temas)]:
                if obj is None:
                    missing.append(name)
            if missing:
                print("Faltan objetos para predecir temas en texto nuevo:", ", ".join(missing))
                print("Pásalos a build_ui(tfidf=..., nmf=..., bow=..., lda=..., ...)")
                return

            res_pred = predecir_tema(
                texto_nuevo,
                mode="both",
                tfidf=tfidf, nmf=nmf, nombres_temas_auto=nombres_temas_auto,
                bow=bow, lda=lda, mapa_temas=mapa_temas,
                top_k=3,
                verbose=True
            )
            print_nuevo(res_pred)

            texto_proc_lema = res_pred["texto_proc"]
            tokens_new = texto_proc_lema.split()

            vec_new = doc_vector(tokens_new, w2v)
            if np.allclose(vec_new, 0):
                print("\nVector cero: el texto nuevo no comparte vocabulario con Word2Vec.")
                return

            vec_new_norm = vec_new / (np.linalg.norm(vec_new) + 1e-12)
            idx, sims = buscar_similares_por_vec(vec_new_norm, top_k=topk, excluir_idx=None)

            print(f"\nTOP-{topk} SIMILARES (Word2Vec + coseno):\n")
            display(resultados_df(idx, sims))

    btn_doc.on_click(on_doc_click)
    btn_new.on_click(on_new_click)

    ui = widgets.VBox([
        widgets.HTML(f"<h3>Similitud semántica (Word2Vec) + Clasificación temática (NMF+LDA)"),
        selector,
        btn_doc,
        widgets.HTML("<hr><b>O pega un texto nuevo:</b>"),
        txt_new,
        btn_new,
        out
    ])

    clear_output(wait=True)
    display(ui)

    # opcional: devolver cosas útiles
    return {"df": df, "w2v": w2v, "X_w2v": X_w2v}
