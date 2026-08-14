# CODEFEST AD ASTRA 2026 — Sistema de Recuperación Semántica

Proyecto desarrollado para el reto clasificatorio **CODEFEST AD ASTRA 2026**.

El objetivo es recuperar, para cada consulta, los **3 documentos** y **10 fragmentos** más relevantes dentro del corpus oficial usando búsqueda semántica con **BAAI/bge-m3**, una base vectorial **FAISS** y un reranking determinista.

---

## ¿Cómo funciona?

```text
Documentos
   ↓
Extracción de texto
   ↓
Chunking
   ↓
BAAI/bge-m3
   ↓
Embeddings de 1024 dimensiones
   ↓
FAISS IndexFlatIP
   ↓
index.faiss + metadata.jsonl
```

Cuando se ejecuta una consulta:

```text
consultas.jsonl
      ↓
generador.py
      ↓
BGE-M3 convierte la consulta en embedding
      ↓
FAISS recupera candidatos similares
      ↓
Filtro por fenómeno
      ↓
Reranking semántico + léxico
      ↓
TOP 3 documentos + TOP 10 fragmentos
      ↓
resultados.jsonl
```

---

## Componentes principales

### `generador.py`

Script principal del sistema.

Lee las consultas, carga BGE-M3, abre la base FAISS, recupera los candidatos y genera el archivo final `resultados.jsonl`.

### `index.faiss`

Contiene los embeddings de todos los chunks del corpus.

El índice utilizado es:

```text
FAISS IndexFlatIP
```

Los embeddings fueron normalizados con norma L2, por lo que el producto interno se usa como medida de similitud entre los vectores normalizados.

### `metadata.jsonl`

Mantiene la relación entre cada posición del índice FAISS y su fragmento original.

Cada registro contiene:

```text
doc_id
chunk_id
fuente
formato
fenomeno
posicion
num_tokens
texto
```

### `consultas.jsonl`

Contiene las 50 consultas oficiales:

```text
q001 ... q050
```

### `resultados.jsonl`

Salida producida por `generador.py`.

Cada consulta contiene exactamente:

```text
3 documentos
10 fragmentos
```

---

## Encoder

Se utiliza:

```text
BAAI/bge-m3
```

El encoder transforma cada texto en un vector numérico de **1024 dimensiones**.

Durante la construcción de la base se generaron aproximadamente:

```text
185,376 chunks
185,376 embeddings
```

Los chunks fueron construidos respetando la integridad lingüística de las oraciones y un máximo de 250 palabras por fragmento final.

---

## Estrategia de recuperación

El sistema combina:

- búsqueda semántica con BGE-M3;
- FAISS `IndexFlatIP`;
- filtro por fenómeno;
- coincidencia léxica;
- equivalencias terminológicas;
- conceptos críticos;
- pares conceptuales;
- penalización ligera de ruido bibliográfico.

Ejemplos de equivalencias:

```text
NBQR ↔ CBRN ↔ CBRNE
IA ↔ AI ↔ Artificial Intelligence
ASAT ↔ anti-satellite
minería ilegal ↔ illegal mining
narcotráfico ↔ drug trafficking
```

El reranking no modifica los embeddings ni la base FAISS. Solo reorganiza los candidatos recuperados.

---

## Fenómenos

```text
q001 - q016 → Fenómeno 1
q017 - q032 → Fenómeno 2
q033 - q050 → Fenómeno 3
```

---

## Instalación

Se recomienda utilizar un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Estructura esperada

```text
entrega/
├── generador.py
├── requirements.txt
├── consultas.jsonl
├── resultados.jsonl
├── informe_tecnico.pdf
└── base_vectorial/
    └── encoder_bge-m3/
        ├── index.faiss
        └── metadata.jsonl
```

---

## Ejecución

Desde la carpeta raíz:

```bash
python generador.py
```

También puede ejecutarse explícitamente:

```bash
python generador.py \
  --consultas consultas.jsonl \
  --base-vectorial ./base_vectorial \
  --salida resultados.jsonl
```

Al finalizar debe aparecer una validación similar a:

```text
GENERACIÓN FINALIZADA
Líneas: 50
Documentos por consulta: 3
Fragmentos por consulta: 10
VALIDACIÓN FINAL: OK
```

---

## Archivos grandes

`index.faiss` y `metadata.jsonl` son archivos grandes.

Si se almacenan dentro del repositorio, puede usarse **Git LFS**:

```bash
git lfs install
git lfs track "*.faiss"
git lfs track "metadata.jsonl"
git add .gitattributes
```

Después:

```bash
git add .
git commit -m "Initial CODEFEST semantic retrieval system"
```

Como alternativa, la base vectorial puede distribuirse mediante una Release o almacenamiento externo y mantener el código en GitHub.

---

## Tecnologías

```text
Python
PyTorch
FlagEmbedding
BAAI/bge-m3
FAISS
NumPy
JSONL
```

---

## Grafo de conocimiento

En la versión final no se implementó un grafo de conocimiento.

Se priorizó una arquitectura de recuperación vectorial porque la combinación BGE-M3 + FAISS + reranking ofreció una solución más simple, reproducible y suficiente para el objetivo del reto.

---

## Reproducibilidad

La correspondencia entre:

```text
posición del vector en index.faiss
```

y:

```text
línea correspondiente en metadata.jsonl
```

debe mantenerse exactamente.

Modificar el orden de `metadata.jsonl` sin reconstruir `index.faiss` rompe la correspondencia entre vectores y chunks.

---

## Licencia

El código fuente de este repositorio se publica bajo licencia **MIT**. Consulta el archivo [`LICENSE`](LICENSE).

La licencia MIT de este repositorio aplica únicamente al código desarrollado para este proyecto.

No implica derechos sobre:

- el corpus oficial de CODEFEST;
- documentos originales de terceros;
- datasets externos;
- pesos de modelos externos;
- contenido con licencias propias.

`BAAI/bge-m3` y las demás dependencias conservan sus respectivas licencias.

---

## Nota

Este repositorio corresponde a una solución de recuperación de información. No utiliza un modelo generativo para producir respuestas: recupera y ordena documentos y fragmentos existentes del corpus.
