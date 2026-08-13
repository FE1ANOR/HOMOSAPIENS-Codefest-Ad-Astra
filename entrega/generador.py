#!/usr/bin/env python3
"""
CODEFEST AD ASTRA 2026
Generador final de resultados.

Encoder:
    BAAI/bge-m3

Base esperada:
    ./base_vectorial/encoder_bge-m3/index.faiss
    ./base_vectorial/encoder_bge-m3/metadata.jsonl

Entrada por defecto:
    ./consultas.jsonl

Salida por defecto:
    ./resultados.jsonl

Contrato oficial:
    python generador.py

Equivalente a:
    python generador.py \
        --consultas consultas.jsonl \
        --base-vectorial ./base_vectorial \
        --salida resultados.jsonl

Dependencias:
    pip install -U FlagEmbedding faiss-cpu numpy torch

Estrategia:
    1. Búsqueda semántica densa con BGE-M3 + FAISS IndexFlatIP.
    2. Filtro por fenómeno mediante metadata:
           q001-q016 -> F1
           q017-q032 -> F2
           q033-q050 -> F3
    3. Reranking determinista:
           score_final =
               score_dense
               + 0.18 * score_lexico
               + 0.10 * score_concepto_critico
               + 0.12 * score_pareja_conceptual
               - 0.05 * ruido_bibliografico
    4. Devuelve exactamente:
           - 3 documentos
           - 10 fragmentos
           - máximo 250 palabras por fragmento

No utiliza modelos generativos.
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

try:
    import faiss
    import numpy as np
    import torch
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    print(
        "Faltan dependencias.\n"
        "Instala con:\n"
        "pip install -U FlagEmbedding faiss-cpu numpy torch",
        file=sys.stderr,
    )
    raise


# ============================================================
# CONFIGURACIÓN
# ============================================================

MODELO = "BAAI/bge-m3"
CARPETA_ENCODER = "encoder_bge-m3"

DIMENSION_ESPERADA = 1024
MAX_LENGTH_QUERY = 512

TOP_FAISS = 10000
TOP_CANDIDATOS_FENOMENO = 1000

TOP_DOCUMENTOS = 3
TOP_FRAGMENTOS = 10
MAX_PALABRAS_FRAGMENTO = 250

PESO_LEXICO = 0.18
BONUS_CONCEPTO_CRITICO = 0.10
BONUS_PAREJA_CONCEPTUAL = 0.12
PENALIZACION_RUIDO = 0.05


# ============================================================
# STOPWORDS Y EQUIVALENCIAS
# ============================================================

STOPWORDS = {
    # Español
    "a", "al", "como", "cual", "cuales", "que", "de", "del",
    "la", "las", "el", "los", "un", "una", "unos", "unas",
    "y", "o", "en", "para", "por", "con", "sin", "se", "su",
    "sus", "es", "son", "esta", "estan", "ha", "han", "como",
    "manera", "principales", "actualmente", "recientes",
    "recientemente",

    # Inglés
    "a", "an", "and", "or", "of", "to", "in", "for", "with",
    "on", "is", "are", "as", "by", "from", "how", "what",
    "which", "the", "that", "their",

    # Portugués
    "ao", "aos", "as", "da", "das", "do", "dos", "e", "em",
    "entre", "os", "ou", "sem", "sobre", "um", "uma",
}


# Equivalencias terminológicas controladas.
# Se usan únicamente para reranking léxico.
GRUPOS_ALIAS = [
    {
        "nbqr", "cbrn", "cbrne",
        "chemical", "biological", "radiological", "nuclear",
        "quimico", "biologico", "radiologico",
    },
    {
        "ia", "ai",
        "inteligencia artificial",
        "artificial intelligence",
    },
    {
        "dron", "drones", "uav", "uas",
        "unmanned", "unmanned systems",
        "no tripulado", "no tripulados",
    },
    {
        "antisatelite", "anti satellite",
        "anti-satellite", "asat",
    },
    {
        "spoofing", "gnss spoofing",
        "gps spoofing",
    },
    {
        "rpo", "rendezvous",
        "proximity operations",
        "rendezvous and proximity operations",
    },
    {
        "guerra electronica",
        "electronic warfare",
        "jamming", "interference",
    },
    {
        "energia dirigida",
        "directed energy",
        "laser", "lasers",
    },
    {
        "ciber", "cyber",
        "cybersecurity",
        "cibernetica", "ciberneticas",
    },
    {
        "mineria ilegal",
        "illegal mining",
        "gold mining", "oro", "gold",
    },
    {
        "narcotrafico",
        "drug trafficking",
        "cocaine", "cocaina",
    },
    {
        "reclutamiento",
        "recruitment",
        "children",
        "ninos", "ninas", "adolescentes",
    },
    {
        "control territorial",
        "territorial control",
    },
    {
        "operaciones espaciales",
        "space operations",
        "space mission", "space missions",
        "satellite operations",
        "space activities",
    },
    {
        "crimen organizado",
        "organized crime",
        "criminal organization", "criminal organizations",
        "organized criminal groups",
        "criminal networks",
        "cartels", "mafias",
    },
    {
        "instituciones del estado",
        "state institutions",
        "government institutions",
        "public institutions",
        "state capture",
        "institutional capture",
        "corruption",
        "institutional infiltration",
        "infiltration",
        "cooptation", "co-option",
    },
    {
        "grupos armados",
        "grupos armados ilegales",
        "armed groups",
        "illegal armed groups",
        "armed organizations",
        "non state armed groups",
        "criminal groups",
        "gao", "gaor", "gdo",
    },
    {
        "innovaciones tacticas",
        "tactical innovation", "tactical innovations",
        "tactical adaptation", "tactical adaptations",
        "operational innovation", "operational adaptations",
        "new tactics",
        "fpv", "ied", "ieds",
        "improvised explosive device",
        "improvised explosive devices",
        "weaponized drones",
        "explosive drones",
    },
    {
        "rutas aereas",
        "air routes",
        "air route",
        "air corridor", "air corridors",
        "airstrip", "airstrips",
        "clandestine airstrip", "clandestine airstrips",
        "clandestine flights",
        "aircraft", "aviation",
    },
    {
        "trafico de narcoticos",
        "trafico de armas",
        "contrabando",
        "drug trafficking",
        "narcotics trafficking",
        "arms trafficking",
        "weapons trafficking",
        "smuggling",
        "contraband",
    },
    {
        "petrolera", "petroleo",
        "oil", "petroleum",
        "hydrocarbon", "hydrocarbons",
        "oil exploration",
        "oil extraction",
        "oil exploitation",
        "oil production",
        "oil field", "oil fields", "oilfield",
        "pipeline", "pipelines",
    },
    {
        "rentas derivadas",
        "resource rents",
        "extractive rents",
        "royalties",
        "oil rents",
    },
]


# Conceptos especialmente discriminativos.
CONCEPTOS_CRITICOS = [
    {"nbqr", "cbrn", "cbrne"},
    {"spoofing"},
    {"rpo", "rendezvous", "proximity operations"},
    {"asat", "anti-satellite", "anti satellite", "antisatelite"},
    {"directed energy", "energia dirigida"},
    {"illegal mining", "mineria ilegal"},
    {"drug trafficking", "narcotrafico"},
    {"recruitment", "reclutamiento"},
    {
        "rutas aereas", "air routes", "air route",
        "air corridor", "air corridors",
        "airstrip", "airstrips",
        "clandestine flights",
    },
    {
        "petrolera", "petroleo", "oil", "petroleum",
        "hydrocarbon", "hydrocarbons",
        "oil exploration", "oil extraction",
    },
    {
        "crimen organizado", "organized crime",
        "criminal organizations", "criminal networks",
    },
    {
        "innovaciones tacticas", "tactical innovation",
        "tactical adaptations", "operational innovation",
        "weaponized drones", "explosive drones", "ied", "ieds",
    },
]


# Pares de ideas que deben aparecer juntas para preguntas compuestas.
# El bonus solo se activa cuando la propia consulta contiene ambos conceptos.
PARES_CONCEPTUALES = [
    (
        {
            "ia", "ai",
            "inteligencia artificial",
            "artificial intelligence",
        },
        {
            "operaciones espaciales",
            "space operations",
            "space mission", "space missions",
            "satellite operations",
            "space activities",
        },
    ),
    (
        {
            "crimen organizado",
            "organized crime",
            "criminal organizations",
            "criminal networks",
        },
        {
            "instituciones del estado",
            "state institutions",
            "government institutions",
            "public institutions",
            "state capture",
            "institutional capture",
            "corruption",
            "infiltration",
            "cooptation", "co-option",
        },
    ),
    (
        {
            "grupos armados",
            "grupos armados ilegales",
            "armed groups",
            "illegal armed groups",
            "armed organizations",
            "criminal groups",
            "gao", "gaor", "gdo",
        },
        {
            "innovaciones tacticas",
            "tactical innovation",
            "tactical adaptations",
            "operational innovation",
            "new tactics",
            "weaponized drones",
            "explosive drones",
            "fpv", "ied", "ieds",
            "improvised explosive devices",
        },
    ),
    (
        {
            "rutas aereas",
            "air routes", "air route",
            "air corridor", "air corridors",
            "airstrip", "airstrips",
            "clandestine flights",
        },
        {
            "trafico de narcoticos",
            "trafico de armas",
            "contrabando",
            "drug trafficking",
            "narcotics trafficking",
            "arms trafficking",
            "weapons trafficking",
            "smuggling",
            "contraband",
        },
    ),
    (
        {
            "petrolera", "petroleo",
            "oil", "petroleum",
            "hydrocarbon", "hydrocarbons",
            "oil exploration", "oil extraction",
            "oil exploitation",
        },
        {
            "grupos armados",
            "armed groups",
            "illegal armed groups",
            "criminal groups",
            "rentas derivadas",
            "resource rents",
            "extractive rents",
            "royalties",
        },
    ),
]


# ============================================================
# UTILIDADES JSONL
# ============================================================

def leer_jsonl(ruta):
    ruta = Path(ruta)
    registros = []

    with ruta.open("r", encoding="utf-8") as archivo:
        for numero_linea, linea in enumerate(archivo, start=1):
            linea = linea.strip()

            if not linea:
                continue

            try:
                registro = json.loads(linea)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{ruta}: JSON inválido en línea {numero_linea}."
                ) from error

            registros.append(registro)

    return registros


def obtener_texto_consulta(registro):
    for campo in (
        "query",
        "consulta",
        "question",
        "pregunta",
        "text",
    ):
        valor = registro.get(campo)

        if isinstance(valor, str) and valor.strip():
            return valor.strip()

    raise ValueError(
        f"No se encontró el texto de "
        f"{registro.get('query_id', 'consulta desconocida')}."
    )


def cargar_consultas(ruta):
    registros = leer_jsonl(ruta)
    consultas = []
    vistos = set()

    for registro in registros:
        query_id = str(
            registro.get("query_id", "")
        ).strip().lower()

        if not re.fullmatch(r"q\d{3}", query_id):
            raise ValueError(
                f"query_id inválido: {query_id!r}"
            )

        if query_id in vistos:
            raise ValueError(
                f"query_id repetido: {query_id}"
            )

        vistos.add(query_id)

        consultas.append({
            "query_id": query_id,
            "query": obtener_texto_consulta(registro),
        })

    consultas.sort(
        key=lambda item: int(item["query_id"][1:])
    )

    esperados = [
        f"q{i:03d}"
        for i in range(1, 51)
    ]

    encontrados = [
        item["query_id"]
        for item in consultas
    ]

    if encontrados != esperados:
        raise ValueError(
            "consultas.jsonl debe contener exactamente "
            "q001-q050, una vez cada una."
        )

    return consultas


# ============================================================
# FENÓMENOS
# ============================================================

def fenomeno_de_query_id(query_id):
    numero = int(query_id[1:])

    if 1 <= numero <= 16:
        return 1

    if 17 <= numero <= 32:
        return 2

    if 33 <= numero <= 50:
        return 3

    raise ValueError(
        f"query_id fuera del rango oficial: {query_id}"
    )


# ============================================================
# NORMALIZACIÓN LÉXICA
# ============================================================

def normalizar(texto):
    texto = unicodedata.normalize(
        "NFKD",
        str(texto)
    )

    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(caracter)
    )

    texto = texto.lower()

    # Guiones y demás signos se convierten en espacios
    # para que "anti-satellite" y "anti satellite" coincidan.
    texto = re.sub(
        r"[^a-z0-9]+",
        " ",
        texto
    )

    return re.sub(
        r"\s+",
        " ",
        texto
    ).strip()


def contiene_termino(texto_normalizado, termino):
    termino = normalizar(termino)

    if not termino:
        return False

    return (
        f" {termino} "
        in f" {texto_normalizado} "
    )


def tokens_utiles(texto):
    return {
        token
        for token in normalizar(texto).split()
        if (
            len(token) >= 3
            and token not in STOPWORDS
        )
    }


def grupos_alias_activos(consulta):
    consulta_norm = normalizar(consulta)
    activos = []

    for grupo in GRUPOS_ALIAS:
        grupo_norm = {
            normalizar(termino)
            for termino in grupo
        }

        if any(
            contiene_termino(
                consulta_norm,
                termino
            )
            for termino in grupo_norm
        ):
            activos.append(
                grupo_norm
            )

    return activos


def calcular_score_lexico(consulta, texto):
    consulta_tokens = tokens_utiles(
        consulta
    )

    texto_norm = normalizar(
        texto
    )

    texto_tokens = set(
        texto_norm.split()
    )

    if consulta_tokens:
        cobertura_literal = (
            len(
                consulta_tokens
                & texto_tokens
            )
            / len(
                consulta_tokens
            )
        )
    else:
        cobertura_literal = 0.0

    activos = grupos_alias_activos(
        consulta
    )

    if activos:
        encontrados = 0

        for grupo in activos:
            if any(
                contiene_termino(
                    texto_norm,
                    termino
                )
                for termino in grupo
            ):
                encontrados += 1

        cobertura_alias = (
            encontrados
            / len(activos)
        )
    else:
        cobertura_alias = 0.0

    score = (
        0.45 * cobertura_literal
        + 0.55 * cobertura_alias
    )

    return min(
        1.0,
        max(0.0, score)
    )


def calcular_bonus_critico(
    consulta,
    texto
):
    consulta_norm = normalizar(
        consulta
    )

    texto_norm = normalizar(
        texto
    )

    activos = 0
    presentes = 0

    for grupo in CONCEPTOS_CRITICOS:
        grupo_norm = {
            normalizar(termino)
            for termino in grupo
        }

        consulta_contiene = any(
            contiene_termino(
                consulta_norm,
                termino
            )
            for termino in grupo_norm
        )

        if not consulta_contiene:
            continue

        activos += 1

        texto_contiene = any(
            contiene_termino(
                texto_norm,
                termino
            )
            for termino in grupo_norm
        )

        if texto_contiene:
            presentes += 1

    if activos == 0:
        return 0.0

    return (
        presentes
        / activos
    )


def _grupo_presente(texto_norm, grupo):
    return any(
        contiene_termino(
            texto_norm,
            termino
        )
        for termino in {
            normalizar(x)
            for x in grupo
        }
    )


def calcular_bonus_parejas(
    consulta,
    texto
):
    consulta_norm = normalizar(
        consulta
    )

    texto_norm = normalizar(
        texto
    )

    activos = 0
    completos = 0

    for grupo_a, grupo_b in PARES_CONCEPTUALES:
        consulta_a = _grupo_presente(
            consulta_norm,
            grupo_a
        )

        consulta_b = _grupo_presente(
            consulta_norm,
            grupo_b
        )

        if not (
            consulta_a
            and consulta_b
        ):
            continue

        activos += 1

        texto_a = _grupo_presente(
            texto_norm,
            grupo_a
        )

        texto_b = _grupo_presente(
            texto_norm,
            grupo_b
        )

        if (
            texto_a
            and texto_b
        ):
            completos += 1

    if activos == 0:
        return 0.0

    return (
        completos
        / activos
    )


def calcular_ruido_referencias(
    texto
):
    texto_lower = str(
        texto
    ).lower()

    urls = len(
        re.findall(
            r"https?://|www\.",
            texto_lower
        )
    )

    indicadores = sum(
        marcador in texto_lower
        for marcador in (
            "last accessed",
            "accessed on",
            "bibliography",
            "references",
            "doi.org",
        )
    )

    if urls >= 2:
        return 1.0

    if (
        urls >= 1
        and indicadores >= 1
    ):
        return 0.75

    if indicadores >= 2:
        return 0.50

    return 0.0


# ============================================================
# BASE VECTORIAL
# ============================================================

def localizar_encoder(base_vectorial):
    base = Path(base_vectorial)

    preferida = (
        base
        / CARPETA_ENCODER
    )

    if (
        (preferida / "index.faiss").is_file()
        and (preferida / "metadata.jsonl").is_file()
    ):
        return preferida

    candidatas = []

    if base.is_dir():
        for carpeta in sorted(
            base.iterdir()
        ):
            if not carpeta.is_dir():
                continue

            if (
                (carpeta / "index.faiss").is_file()
                and (carpeta / "metadata.jsonl").is_file()
            ):
                candidatas.append(
                    carpeta
                )

    if len(candidatas) == 1:
        return candidatas[0]

    if not base.exists():
        raise FileNotFoundError(
            f"No existe el directorio de base vectorial: {base}"
        )

    if not candidatas:
        raise FileNotFoundError(
            f"No se encontró index.faiss + metadata.jsonl dentro de {base}."
        )

    raise RuntimeError(
        "Se encontraron varios encoders y no existe "
        f"{CARPETA_ENCODER}. No es posible elegir uno automáticamente."
    )


def cargar_base(base_vectorial):
    carpeta_encoder = localizar_encoder(
        base_vectorial
    )

    ruta_index = (
        carpeta_encoder
        / "index.faiss"
    )

    ruta_metadata = (
        carpeta_encoder
        / "metadata.jsonl"
    )

    print(
        f"     Encoder: {carpeta_encoder.name}"
    )

    indice = faiss.read_index(
        str(ruta_index)
    )

    metadata = leer_jsonl(
        ruta_metadata
    )

    if indice.ntotal != len(metadata):
        raise ValueError(
            f"FAISS contiene {indice.ntotal} vectores "
            f"pero metadata contiene {len(metadata)} registros."
        )

    if int(indice.d) != DIMENSION_ESPERADA:
        raise ValueError(
            f"FAISS tiene dimensión {indice.d}; "
            f"se esperaba {DIMENSION_ESPERADA}."
        )

    return (
        indice,
        metadata
    )


# ============================================================
# ENCODER DE CONSULTAS
# ============================================================

def cargar_modelo():
    usar_fp16 = bool(
        torch.cuda.is_available()
    )

    if usar_fp16:
        print(
            "     Dispositivo: CUDA - "
            + torch.cuda.get_device_name(0)
        )
    else:
        print(
            "     Dispositivo: CPU"
        )

    modelo = BGEM3FlagModel(
        MODELO,
        use_fp16=usar_fp16
    )

    return modelo


def codificar_consultas(
    modelo,
    consultas
):
    textos = [
        consulta["query"]
        for consulta in consultas
    ]

    batch_size = (
        8
        if torch.cuda.is_available()
        else 2
    )

    salida = modelo.encode(
        textos,
        batch_size=batch_size,
        max_length=MAX_LENGTH_QUERY,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    vectores = np.asarray(
        salida["dense_vecs"],
        dtype=np.float32
    )

    if vectores.shape != (
        len(consultas),
        DIMENSION_ESPERADA
    ):
        raise ValueError(
            "Dimensión inesperada en los embeddings de consulta: "
            f"{vectores.shape}"
        )

    if not np.isfinite(
        vectores
    ).all():
        raise ValueError(
            "Los embeddings contienen NaN o infinitos."
        )

    faiss.normalize_L2(
        vectores
    )

    return vectores


# ============================================================
# CANDIDATOS Y RERANKING
# ============================================================

def contar_palabras(texto):
    return len(
        re.findall(
            r"\S+",
            str(texto)
        )
    )


def construir_candidatos(
    ids,
    scores,
    metadata,
    consulta,
    fenomeno
):
    candidatos = []
    dense_rank = 0

    for score_dense, faiss_id in zip(
        scores,
        ids
    ):
        faiss_id = int(
            faiss_id
        )

        if faiss_id < 0:
            continue

        registro = metadata[
            faiss_id
        ]

        try:
            fenomeno_registro = int(
                registro["fenomeno"]
            )
        except Exception as error:
            raise ValueError(
                f"fenomeno inválido en metadata, FAISS id {faiss_id}."
            ) from error

        if fenomeno_registro != fenomeno:
            continue

        texto = str(
            registro.get(
                "texto",
                ""
            )
        ).strip()

        if not texto:
            continue

        # La base final fue construida con chunks <=250 palabras.
        # Si aparece un registro fuera de contrato, no lo usamos.
        if contar_palabras(
            texto
        ) > MAX_PALABRAS_FRAGMENTO:
            continue

        doc_id = str(
            registro.get(
                "doc_id",
                ""
            )
        ).strip()

        chunk_id = str(
            registro.get(
                "chunk_id",
                ""
            )
        ).strip()

        if not doc_id or not chunk_id:
            continue

        dense_rank += 1

        score_lexico = calcular_score_lexico(
            consulta,
            texto
        )

        score_critico = calcular_bonus_critico(
            consulta,
            texto
        )

        score_pareja = calcular_bonus_parejas(
            consulta,
            texto
        )

        score_ruido = calcular_ruido_referencias(
            texto
        )

        score_final = (
            float(score_dense)
            + PESO_LEXICO * score_lexico
            + BONUS_CONCEPTO_CRITICO * score_critico
            + BONUS_PAREJA_CONCEPTUAL * score_pareja
            - PENALIZACION_RUIDO * score_ruido
        )

        candidatos.append({
            "faiss_id": faiss_id,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "texto": texto,
            "fenomeno": fenomeno_registro,
            "score_dense": float(
                score_dense
            ),
            "score_lexico": float(
                score_lexico
            ),
            "score_critico": float(
                score_critico
            ),
            "score_pareja": float(
                score_pareja
            ),
            "score_ruido": float(
                score_ruido
            ),
            "score_final": float(
                score_final
            ),
            "dense_rank": dense_rank,
        })

        if len(
            candidatos
        ) >= TOP_CANDIDATOS_FENOMENO:
            break

    candidatos.sort(
        key=lambda item: (
            -item["score_final"],
            -item["score_dense"],
            item["chunk_id"],
        )
    )

    return candidatos


def seleccionar_fragmentos(
    candidatos
):
    if len(
        candidatos
    ) < TOP_FRAGMENTOS:
        raise RuntimeError(
            "No fue posible recuperar 10 fragmentos válidos."
        )

    return candidatos[
        :TOP_FRAGMENTOS
    ]


def seleccionar_documentos(
    candidatos
):
    por_documento = {}

    for candidato in candidatos:
        doc_id = candidato[
            "doc_id"
        ]

        if doc_id not in por_documento:
            por_documento[
                doc_id
            ] = []

        # Solo necesitamos unos pocos mejores chunks por documento.
        if len(
            por_documento[
                doc_id
            ]
        ) < 3:
            por_documento[
                doc_id
            ].append(
                candidato[
                    "score_final"
                ]
            )

    ranking = []

    for doc_id, scores in por_documento.items():
        scores = sorted(
            scores,
            reverse=True
        )

        score_documento = scores[
            0
        ]

        # Igual que en la versión 7.2 validada:
        # mejor chunk + pequeño apoyo del segundo.
        if len(
            scores
        ) >= 2:
            score_documento += (
                0.05
                * scores[1]
            )

        ranking.append({
            "doc_id": doc_id,
            "score": float(
                score_documento
            ),
        })

    ranking.sort(
        key=lambda item: (
            -item["score"],
            item["doc_id"],
        )
    )

    if len(
        ranking
    ) < TOP_DOCUMENTOS:
        raise RuntimeError(
            "No fue posible recuperar 3 documentos."
        )

    return ranking[
        :TOP_DOCUMENTOS
    ]


# ============================================================
# FORMATO OFICIAL DE RESULTADOS
# ============================================================

def construir_resultado(
    query_id,
    documentos,
    fragmentos
):
    return {
        "query_id": query_id,

        "documents": [
            {
                "rank": rank,
                "doc_id": documento[
                    "doc_id"
                ],
            }
            for rank, documento in enumerate(
                documentos,
                start=1
            )
        ],

        "fragments": [
            {
                "rank": rank,
                "chunk_id": fragmento[
                    "chunk_id"
                ],
                "doc_id": fragmento[
                    "doc_id"
                ],
                "text": fragmento[
                    "texto"
                ],
            }
            for rank, fragmento in enumerate(
                fragmentos,
                start=1
            )
        ],
    }


def validar_resultado(
    resultado
):
    if set(
        resultado.keys()
    ) != {
        "query_id",
        "documents",
        "fragments",
    }:
        raise ValueError(
            f"{resultado.get('query_id')}: "
            "campos superiores incorrectos."
        )

    if len(
        resultado["documents"]
    ) != TOP_DOCUMENTOS:
        raise ValueError(
            f"{resultado['query_id']}: "
            "debe contener exactamente 3 documentos."
        )

    if len(
        resultado["fragments"]
    ) != TOP_FRAGMENTOS:
        raise ValueError(
            f"{resultado['query_id']}: "
            "debe contener exactamente 10 fragmentos."
        )

    for rank, documento in enumerate(
        resultado["documents"],
        start=1
    ):
        if set(
            documento.keys()
        ) != {
            "rank",
            "doc_id",
        }:
            raise ValueError(
                f"{resultado['query_id']}: "
                f"campos incorrectos en documento rank {rank}."
            )

        if documento[
            "rank"
        ] != rank:
            raise ValueError(
                f"{resultado['query_id']}: "
                "ranking documental incorrecto."
            )

        if not documento[
            "doc_id"
        ]:
            raise ValueError(
                f"{resultado['query_id']}: "
                "doc_id vacío."
            )

    for rank, fragmento in enumerate(
        resultado["fragments"],
        start=1
    ):
        if set(
            fragmento.keys()
        ) != {
            "rank",
            "chunk_id",
            "doc_id",
            "text",
        }:
            raise ValueError(
                f"{resultado['query_id']}: "
                f"campos incorrectos en fragmento rank {rank}."
            )

        if fragmento[
            "rank"
        ] != rank:
            raise ValueError(
                f"{resultado['query_id']}: "
                "ranking de fragmentos incorrecto."
            )

        if not fragmento[
            "chunk_id"
        ]:
            raise ValueError(
                f"{resultado['query_id']}: "
                "chunk_id vacío."
            )

        if not fragmento[
            "doc_id"
        ]:
            raise ValueError(
                f"{resultado['query_id']}: "
                "doc_id vacío."
            )

        palabras = contar_palabras(
            fragmento[
                "text"
            ]
        )

        if not (
            1
            <= palabras
            <= MAX_PALABRAS_FRAGMENTO
        ):
            raise ValueError(
                f"{resultado['query_id']}: "
                f"fragmento rank {rank} tiene {palabras} palabras."
            )


def validar_salida_completa(
    resultados
):
    if len(
        resultados
    ) != 50:
        raise ValueError(
            "La salida debe tener exactamente 50 resultados."
        )

    esperados = [
        f"q{i:03d}"
        for i in range(
            1,
            51
        )
    ]

    encontrados = [
        resultado[
            "query_id"
        ]
        for resultado in resultados
    ]

    if encontrados != esperados:
        raise ValueError(
            "La salida debe estar ordenada exactamente de q001 a q050."
        )

    for resultado in resultados:
        validar_resultado(
            resultado
        )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generador CODEFEST AD ASTRA 2026 "
            "con BGE-M3 + FAISS + reranking léxico."
        )
    )

    parser.add_argument(
        "--consultas",
        default="consultas.jsonl",
        help="Ruta al archivo consultas.jsonl",
    )

    parser.add_argument(
        "--base-vectorial",
        default="./base_vectorial",
        help="Ruta al directorio base_vectorial",
    )

    parser.add_argument(
        "--salida",
        default="./resultados.jsonl",
        help="Ruta del archivo resultados.jsonl",
    )

    args = parser.parse_args()

    print("1/6 Leyendo consultas...")
    consultas = cargar_consultas(
        args.consultas
    )

    print(
        f"     Consultas: {len(consultas)}"
    )

    print("2/6 Cargando base vectorial...")
    indice, metadata = cargar_base(
        args.base_vectorial
    )

    print(
        f"     Vectores: {indice.ntotal:,}"
    )
    print(
        f"     Dimensión: {indice.d}"
    )
    print(
        f"     Metadata: {len(metadata):,}"
    )

    print("3/6 Cargando BGE-M3...")
    modelo = cargar_modelo()

    print("4/6 Codificando 50 consultas...")
    vectores = codificar_consultas(
        modelo,
        consultas
    )

    print("5/6 Buscando y construyendo rankings...")

    top_k = min(
        TOP_FAISS,
        indice.ntotal
    )

    scores_todos, ids_todos = indice.search(
        vectores,
        top_k
    )

    resultados = []

    for i, consulta in enumerate(
        consultas
    ):
        query_id = consulta[
            "query_id"
        ]

        fenomeno = fenomeno_de_query_id(
            query_id
        )

        candidatos = construir_candidatos(
            ids_todos[i],
            scores_todos[i],
            metadata,
            consulta[
                "query"
            ],
            fenomeno,
        )

        documentos = seleccionar_documentos(
            candidatos
        )

        fragmentos = seleccionar_fragmentos(
            candidatos
        )

        resultado = construir_resultado(
            query_id,
            documentos,
            fragmentos
        )

        validar_resultado(
            resultado
        )

        resultados.append(
            resultado
        )

        print(
            f"     {query_id}: OK"
        )

    print("6/6 Validando y guardando resultados.jsonl...")

    validar_salida_completa(
        resultados
    )

    ruta_salida = Path(
        args.salida
    )

    ruta_salida.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with ruta_salida.open(
        "w",
        encoding="utf-8"
    ) as archivo:
        for resultado in resultados:
            archivo.write(
                json.dumps(
                    resultado,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":"
                    ),
                )
                + "\n"
            )

    # Validación física del archivo final.
    comprobacion = leer_jsonl(
        ruta_salida
    )

    validar_salida_completa(
        comprobacion
    )

    print()
    print("=" * 72)
    print("GENERACIÓN FINALIZADA")
    print("=" * 72)
    print(
        f"Salida: {ruta_salida}"
    )
    print(
        f"Líneas: {len(comprobacion)}"
    )
    print(
        "Documentos por consulta: 3"
    )
    print(
        "Fragmentos por consulta: 10"
    )
    print(
        "Máximo por fragmento: 250 palabras"
    )
    print(
        "VALIDACIÓN FINAL: OK"
    )


if __name__ == "__main__":
    main()
