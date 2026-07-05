<div align="center">

# PDDAI

**Modelo transformer construido desde cero — 4,69M parámetros**

[🇬🇧 English](README-en.md) • [🇸🇦 العربية](README-ar.md) • [🇨🇳 中文](README-zh.md) • [🇯🇵 日本語](README-ja.md) • [🇷🇺 Русский](README-ru.md) • [🇫🇷 Français](README-fr.md) • [🇩🇪 Deutsch](README-de.md) • [🇵🇹 Português](README-pt.md) • [🇮🇳 हिन्दी](README-hi.md)

</div>

Un modelo transformer decoder-only de **4,69M parámetros** construido desde cero con PyTorch — con GQA, RoPE, SwiGLU, Refresh Gates, inferencia con caché KV, búsqueda multi-fuente, herramientas de agente, RAG e integración con 9Router.

## Características

| Categoría | Detalles |
|---|---|
| **Arquitectura** | Atención de consulta agrupada (6 cabezas de consulta, 3 cabezas KV), incrustaciones posicionales rotativas, SwiGLU FFN, Refresh Gates, caché KV |
| **Entrenamiento** | Precisión mixta (AMP), acumulación de gradientes, optimizador Muon, pérdida Unlikelihood + suavizado de etiquetas, guardado de checkpoints |
| **Chat** | Conversación simple con salida en flujo (token por token) y exportación de datos de entrenamiento al salir |
| **Agente** | Razonamiento con herramientas y búsqueda multi-fuente (Wikipedia, Wikidata, arXiv, PubMed, StackExchange, GitHub), calculadora, lector de archivos |
| **Enrutador de búsqueda** | Enrutamiento inteligente de consultas para seleccionar las mejores fuentes y agregar resultados (límite 3000 caracteres) |
| **Pipeline 9Router** | API de IA externa que comprime resultados de búsqueda en contexto limpio para el modelo local |
| **Limpiador de salida** | Eliminación de caracteres de control, reducción de repeticiones, deduplicación de oraciones, corrección de espaciado |
| **RAG** | Generación aumentada por recuperación — versiones simple y mejorada |
| **AI2AI** | Conversación autónoma entre dos modelos con simulación de turnos |
| **Exportación de entrenamiento** | Conversaciones formateadas vía 9Router y guardadas como datos de entrenamiento de alta calidad |

## Inicio rápido

```bash
# Entrenar desde cero
python main.py

# Chatear con el modelo
python chat.py

# Agente con herramientas
python -m agents.agent

# Conversación IA-IA
python ai2ai.py
```

## Estructura del proyecto

```
PDDAI/
├── config.py               # Hiperparámetros + configuración 9Router
├── main.py                 # Pipeline de entrenamiento
├── chat.py                 # Interfaz de chat
├── ai2ai.py                # Conversación IA-IA
├── models/
│   └── transformer.py      # TransformerModel, GQA, RoPE, SwiGLU
├── training/
│   └── train.py            # Bucle de entrenamiento
├── agents/
│   └── agent.py            # Agente con razonamiento + búsqueda
├── tools/
│   └── registry.py         # Sistema de herramientas
├── data/
│   ├── tokenizer.py        # Tokenizador BPE
│   └── dataset.py          # Carga de datos
├── rag/
│   ├── simple_rag.py       # RAG básico
│   └── better_rag.py       # RAG mejorado
├── memory/
│   └── light_memory.py     # Sistema de memoria ligero
└── utils/
    ├── auto_config.py      # Detección de hardware
    └── muon.py             # Optimizador Muon
```

## Requisitos

- Python 3.10+
- PyTorch 2.0+
- GPU con CUDA recomendado (funciona en CPU)
- Biblioteca `datasets` de HuggingFace

## Arquitectura

- **GQA** — 6 cabezas de consulta, 3 cabezas KV para memoria eficiente
- **RoPE** — Codificación posicional relativa sin parámetros aprendidos
- **SwiGLU** — FFN con compuerta (Swish × GLU)
- **Refresh Gates** — Actualizaciones selectivas por capa
- **Caché KV** — Generación O(1) por token

## Licencia

Solo uso no comercial. Ver [LICENSE](LICENSE).
