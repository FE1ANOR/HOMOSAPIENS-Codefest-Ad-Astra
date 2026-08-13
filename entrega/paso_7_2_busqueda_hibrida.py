#!/usr/bin/env python3
"""
CODEFEST AD ASTRA 2026
Paso 7.2: búsqueda semántica + refuerzo léxico controlado.

Coloca en la MISMA carpeta:
    paso_7_2_busqueda_hibrida.py
    index.faiss
    metadata.jsonl
    consultas.jsonl

Uso:
    python paso_7_2_busqueda_hibrida.py

Luego escribe:
    q001
    q017
    q033
    ...

No modifica index.faiss ni metadata.jsonl.
"""

import json
import re
import unicodedata
from pathlib import Path

import faiss
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel


MODELO = "BAAI/bge-m3"

CARPETA = Path(__file__).resolve().parent
RUTA_INDEX = CARPETA / "index.faiss"
RUTA_METADATA = CARPETA / "metadata.jsonl"
RUTA_CONSULTAS = CARPETA / "consultas.jsonl"

TOP_FAISS = 5000
TOP_CANDIDATOS_FENOMENO = 500
TOP_CHUNKS = 10
TOP_DOCUMENTOS = 3

# Importante:
# ya NO hacemos min-max del score denso.
# Conservamos directamente la similitud BGE-M3
# y añadimos un refuerzo léxico pequeño.
PESO_LEXICO = 0.18
BONUS_CONCEPTO_CRITICO = 0.10

MAX_LENGTH_QUERY = 512


STOPWORDS = {
    "como", "cual", "cuales", "que", "de", "del", "la", "las", "el", "los",
    "un", "una", "unos", "unas", "y", "o", "en", "para", "por", "con", "sin",
    "se", "su", "sus", "al", "a", "es", "son", "esta", "estan", "han", "ha",
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on",
    "is", "are", "as", "by", "from", "how", "what", "which",
}


# Grupos de equivalencias de dominio.
# Si la consulta contiene un término del grupo,
# cualquiera de sus equivalentes cuenta como coincidencia.
GRUPOS_ALIAS = [
    {
        "nbqr", "cbrn", "cbrne",
        "chemical", "biological", "radiological", "nuclear",
        "quimico", "biologico", "radiologico", "nuclear"
    },
    {
        "ia", "ai",
        "inteligencia artificial",
        "artificial intelligence"
    },
    {
        "dron", "drones", "uav", "uas",
        "unmanned", "unmanned systems",
        "no tripulado", "no tripulados"
    },
    {
        "antisatelite", "anti satellite",
        "anti-satellite", "asat"
    },
    {
        "spoofing", "gnss spoofing",
        "gps spoofing"
    },
    {
        "rpo", "rendezvous",
        "proximity operations",
        "rendezvous and proximity operations"
    },
    {
        "guerra electronica",
        "electronic warfare",
        "jamming", "interference"
    },
    {
        "energia dirigida",
        "directed energy",
        "laser", "lasers"
    },
    {
        "ciber", "cyber",
        "cybersecurity",
        "cibernetica", "ciberneticas"
    },
    {
        "mineria ilegal",
        "illegal mining",
        "gold mining", "oro", "gold"
    },
    {
        "narcotrafico",
        "drug trafficking",
        "cocaine", "cocaina"
    },
    {
        "reclutamiento",
        "recruitment",
        "children",
        "ninos", "ninas", "adolescentes"
    },
    {
        "control territorial",
        "territorial control"
    },
]


# Conceptos cuya presencia explícita es especialmente informativa.
CONCEPTOS_CRITICOS = [
    {"nbqr", "cbrn", "cbrne"},
    {"spoofing"},
    {"rpo", "rendezvous", "proximity operations"},
    {"asat", "anti-satellite", "anti satellite", "antisatelite"},
    {"directed energy", "energia dirigida"},
    {"illegal mining", "mineria ilegal"},
    {"drug trafficking", "narcotrafico"},
    {"recruitment", "reclutamiento"},
]


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(
        c for c in texto
        if not unicodedata.combining(c)
    )
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9\s\-]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokens_utiles(texto):
    return {
        t
        for t in normalizar(texto).split()
        if len(t) >= 3 and t not in STOPWORDS
    }


def grupos_activos(consulta):
    q = normalizar(consulta)
    activos = []

    for grupo in GRUPOS_ALIAS:
        gn = {normalizar(x) for x in grupo}

        if any(alias in q for alias in gn):
            activos.append(gn)

    return activos


def score_lexico(consulta, texto):
    q_tokens = tokens_utiles(consulta)
    t_norm = normalizar(texto)
    t_tokens = set(t_norm.split())

    if q_tokens:
        base = len(q_tokens & t_tokens) / len(q_tokens)
    else:
        base = 0.0

    activos = grupos_activos(consulta)

    if activos:
        encontrados = 0

        for grupo in activos:
            if any(alias in t_norm for alias in grupo):
                encontrados += 1

        alias_score = encontrados / len(activos)
    else:
        alias_score = 0.0

    return min(
        1.0,
        0.45 * base + 0.55 * alias_score
    )


def bonus_critico(consulta, texto):
    q = normalizar(consulta)
    t = normalizar(texto)

    activos = 0
    presentes = 0

    for grupo in CONCEPTOS_CRITICOS:
        gn = {normalizar(x) for x in grupo}

        if any(x in q for x in gn):
            activos += 1

            if any(x in t for x in gn):
                presentes += 1

    if activos == 0:
        return 0.0

    return presentes / activos


def fenomeno_de_query_id(query_id):
    n = int(query_id[1:])

    if 1 <= n <= 16:
        return 1
    if 17 <= n <= 32:
        return 2
    if 33 <= n <= 50:
        return 3

    raise ValueError(query_id)


def cargar_jsonl(ruta):
    out = []

    with ruta.open("r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()

            if linea:
                out.append(json.loads(linea))

    return out


def cargar_sistema():
    for ruta in (
        RUTA_INDEX,
        RUTA_METADATA,
        RUTA_CONSULTAS
    ):
        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe: {ruta}"
            )

    print("Cargando index.faiss...")
    index = faiss.read_index(str(RUTA_INDEX))

    print("Cargando metadata.jsonl...")
    metadata = cargar_jsonl(RUTA_METADATA)

    print("Cargando consultas.jsonl...")
    consultas_lista = cargar_jsonl(RUTA_CONSULTAS)

    consultas = {
        x["query_id"]: x["query"]
        for x in consultas_lista
    }

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"FAISS={index.ntotal}, metadata={len(metadata)}"
        )

    if len(consultas) != 50:
        raise RuntimeError(
            f"Consultas esperadas=50, encontradas={len(consultas)}"
        )

    usar_fp16 = torch.cuda.is_available()

    if usar_fp16:
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )
    else:
        print("Usando CPU.")

    print("Cargando BAAI/bge-m3...")

    modelo = BGEM3FlagModel(
        MODELO,
        use_fp16=usar_fp16
    )

    print()
    print("BUSCADOR 7.2 LISTO")
    print(f"Vectores: {index.ntotal:,}")
    print(f"Metadata: {len(metadata):,}")
    print(f"Consultas: {len(consultas)}")

    return (
        index,
        metadata,
        consultas,
        modelo
    )


def embedding_query(modelo, consulta):
    salida = modelo.encode(
        [consulta],
        batch_size=1,
        max_length=MAX_LENGTH_QUERY,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    vector = np.asarray(
        salida["dense_vecs"],
        dtype=np.float32
    )

    faiss.normalize_L2(vector)
    return vector


def buscar(
    index,
    metadata,
    modelo,
    consulta,
    fenomeno
):
    vector = embedding_query(
        modelo,
        consulta
    )

    k = min(
        TOP_FAISS,
        index.ntotal
    )

    scores, ids = index.search(
        vector,
        k
    )

    candidatos = []
    dense_rank = 0

    for dense, idx in zip(
        scores[0],
        ids[0]
    ):
        idx = int(idx)

        if idx < 0:
            continue

        r = metadata[idx]

        if int(r["fenomeno"]) != fenomeno:
            continue

        dense_rank += 1

        lex = score_lexico(
            consulta,
            r["texto"]
        )

        crit = bonus_critico(
            consulta,
            r["texto"]
        )

        final = (
            float(dense)
            + PESO_LEXICO * lex
            + BONUS_CONCEPTO_CRITICO * crit
        )

        candidatos.append({
            **r,
            "faiss_id": idx,
            "dense_rank": dense_rank,
            "score_dense": float(dense),
            "score_lexico": lex,
            "score_critico": crit,
            "score_final": final,
        })

        if len(candidatos) >= TOP_CANDIDATOS_FENOMENO:
            break

    candidatos.sort(
        key=lambda x: (
            -x["score_final"],
            -x["score_dense"],
            x["chunk_id"]
        )
    )

    top_chunks = candidatos[:TOP_CHUNKS]

    # Ranking de documentos:
    # mejor chunk + apoyo pequeño del segundo mejor chunk.
    por_doc = {}

    for c in candidatos:
        doc_id = c["doc_id"]

        por_doc.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "fuente": c["fuente"],
                "scores": []
            }
        )

        if len(por_doc[doc_id]["scores"]) < 3:
            por_doc[doc_id]["scores"].append(
                c["score_final"]
            )

    docs = []

    for d in por_doc.values():
        ss = sorted(
            d["scores"],
            reverse=True
        )

        score_doc = ss[0]

        if len(ss) >= 2:
            score_doc += 0.05 * ss[1]

        docs.append({
            "doc_id": d["doc_id"],
            "fuente": d["fuente"],
            "score": score_doc,
        })

    docs.sort(
        key=lambda x: (
            -x["score"],
            x["doc_id"]
        )
    )

    return (
        top_chunks,
        docs[:TOP_DOCUMENTOS]
    )


def imprimir(
    query_id,
    consulta,
    fenomeno,
    chunks,
    docs
):
    print()
    print("=" * 96)
    print(
        f"{query_id} | F{fenomeno}"
    )
    print("=" * 96)
    print(consulta)

    print()
    print("TOP 3 DOCUMENTOS")
    print("-" * 96)

    for rank, d in enumerate(
        docs,
        1
    ):
        print(
            f"{rank}. {d['doc_id']} | "
            f"score={d['score']:.6f}"
        )
        print(
            f"   {d['fuente']}"
        )

    print()
    print("TOP 10 CHUNKS")
    print("-" * 96)

    for rank, c in enumerate(
        chunks,
        1
    ):
        texto = " ".join(
            str(c["texto"]).split()
        )

        if len(texto) > 700:
            texto = texto[:700] + "..."

        print()
        print(
            f"{rank}. final={c['score_final']:.6f} | "
            f"dense={c['score_dense']:.6f} | "
            f"lex={c['score_lexico']:.4f} | "
            f"crit={c['score_critico']:.2f} | "
            f"dense_rank={c['dense_rank']}"
        )
        print(
            f"   doc_id: {c['doc_id']}"
        )
        print(
            f"   chunk_id: {c['chunk_id']}"
        )
        print(
            f"   formato: {c['formato']}"
        )
        print(
            f"   texto: {texto}"
        )


def main():
    index, metadata, consultas, modelo = cargar_sistema()

    print()
    print(
        "Escribe q001 ... q050."
    )
    print(
        "Para terminar: salir"
    )

    while True:
        query_id = input(
            "\nquery_id: "
        ).strip().lower()

        if query_id in {
            "salir",
            "exit",
            "quit"
        }:
            break

        if query_id not in consultas:
            print(
                "query_id inválido."
            )
            continue

        consulta = consultas[query_id]
        fenomeno = fenomeno_de_query_id(
            query_id
        )

        chunks, docs = buscar(
            index,
            metadata,
            modelo,
            consulta,
            fenomeno
        )

        imprimir(
            query_id,
            consulta,
            fenomeno,
            chunks,
            docs
        )


if __name__ == "__main__":
    main()
