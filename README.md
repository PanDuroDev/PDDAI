<div align="center">

# PDDAI

**From-scratch transformer language model — 4.69M parameters**

</div>

---

<p align="center">
  <a href="#en">🇬🇧 English</a> •
  <a href="#ar">🇸🇦 العربية</a> •
  <a href="#fr">🇫🇷 Français</a> •
  <a href="#es">🇪🇸 Español</a> •
  <a href="#de">🇩🇪 Deutsch</a>
</p>

---

<a name="en"></a>

## 🇬🇧 English

[🇸🇦 العربية](#ar) • [🇫🇷 Français](#fr) • [🇪🇸 Español](#es) • [🇩🇪 Deutsch](#de)

A **4.69M parameter** decoder-only transformer built from scratch with PyTorch — featuring GQA, RoPE, SwiGLU, Refresh Gates, KV-cached inference, multi-source search, agent tools, RAG, and 9Router integration.

### Features

| Category | Details |
|---|---|
| **Architecture** | Grouped Query Attention (6 heads, 3 KV heads), Rotary Positional Embeddings, SwiGLU FFN, Refresh Gates, KV cache |
| **Training** | Mixed precision (AMP), gradient accumulation, Muon optimizer, Unlikelihood loss + label smoothing, last/best/final checkpointing |
| **Chat** | Plain conversation with streamed token-by-token output and on-exit training data export |
| **Agent** | Tool-assisted reasoning with multi-source search (Wikipedia, Wikidata, arXiv, PubMed, StackExchange, GitHub), calculator, file reader |
| **Search Router** | Query-intent routing that selects the best API sources and aggregates results (3000 char limit) |
| **9Router Pipeline** | External AI API compresses multi-source search results into clean context for the local model |
| **Output Cleaner** | Lightweight post-processing: control char removal, repetition reduction, sentence deduplication, spacing fixes |
| **RAG** | Simple and improved retrieval-augmented generation implementations |
| **AI2AI** | Two-model autonomous conversation with simulated turn-taking |
| **Training Export** | Conversations formatted via 9Router into high-quality training data, saved on clean exit |

### Quick Start

```bash
# Train from scratch
python main.py

# Chat with the trained model
python chat.py

# Agent with tool access
python -m agents.agent

# AI-to-AI conversation
python ai2ai.py
```

### Project Structure

```
PDDAI/
├── config.py               # Hyperparameters + 9Router settings
├── main.py                 # Training pipeline
├── chat.py                 # Chat interface with streaming
├── ai2ai.py                # AI-to-AI conversation
├── models/
│   └── transformer.py      # TransformerModel, GQA, RoPE, SwiGLU
├── training/
│   └── train.py            # Training loop, checkpointing, logging
├── agents/
│   └── agent.py            # Agent with tool reasoning + multi-source search
├── tools/
│   └── registry.py         # Tool system (web_search, calculator, read_file)
├── data/
│   ├── tokenizer.py        # BPE tokenizer training
│   └── dataset.py          # Dataset loading
├── rag/
│   ├── simple_rag.py       # Basic RAG
│   └── better_rag.py       # Improved RAG with enhanced retrieval
├── memory/
│   └── light_memory.py     # Lightweight memory system
└── utils/
    ├── auto_config.py      # Hardware auto-detection
    └── muon.py             # Muon optimizer implementation
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU recommended (falls back to CPU)
- HuggingFace `datasets` library

### Architecture

- **Grouped Query Attention** — 6 query heads, 3 KV heads for efficient memory
- **Rotary Positional Embeddings** — relative position encoding, no learned params
- **SwiGLU** — gated FFN (Swish × GLU)
- **Refresh Gates** — per-layer selective state updates
- **KV Cache** — O(1) per-token generation

### License

Non-Commercial Use Only. See [LICENSE](LICENSE).

---

<a name="ar"></a>

## 🇸🇦 العربية

[🇬🇧 English](#en) • [🇫🇷 Français](#fr) • [🇪🇸 Español](#es) • [🇩🇪 Deutsch](#de)

نموذج محول **4.69M معامل** مبني من الصفر باستخدام PyTorch — مع GQA، RoPE، SwiGLU، Refresh Gates، استدلال مخبئ، بحث متعدد المصادر، أدوات وكيل، RAG، وتكامل 9Router.

### المميزات

| التصنيف | التفاصيل |
|---|---|
| **الهندسة** | انتباه استعلامي مجمع (6 رؤوس استعلام، 3 رؤوس KV)، تضمينات موضعية دورانية، SwiGLU، بوابات التحديث، ذاكرة تخزين KV |
| **التدريب** | دقة مختلطة، تراكم التدرج، محسن Muon، خسارة عدم الاحتمال + تنعيم التسميات، حفظ النماذج |
| **المحادثة** | محادثة مع إخراج متدفق رمزًا برمز، حفظ بيانات التدريب عند الخروج |
| **الوكيل** | استدلال بأدوات وبحث متعدد المصادر (ويكيبيديا، ويكي بيانات، arXiv، PubMed، StackExchange، GitHub)، آلة حاسبة، قارئ ملفات |
| **موجه البحث** | توجيه الاستعلام لاختيار أفضل المصادر وتجميع النتائج (حد 3000 حرف) |
| **خط 9Router** | واجهة AI خارجية تضغط نتائج البحث إلى سياق نظيف للنموذج المحلي |
| **منظف المخرجات** | إزالة رموز التحكم، تقليل التكرار، إزالة ازدواج الجمل، تنظيف المسافات |
| **RAG** | توليد معزز بالاسترجاع — إصدارات بسيطة ومحسّنة |
| **AI2AI** | محادثة مستقلة بين نموذجين مع محاكاة تبادل الأدوار |
| **تصدير التدريب** | المحادثات تُنسق عبر 9Router وتُحفظ كبيانات تدريب عالية الجودة |

### بداية سريعة

```bash
# تدريب من الصفر
python main.py

# محادثة مع النموذج
python chat.py

# وكيل مع أدوات
python -m agents.agent

# محادثة ذكاءين
python ai2ai.py
```

### هيكل المشروع

```
PDDAI/
├── config.py               # الإعدادات
├── main.py                 # خط التدريب
├── chat.py                 # واجهة المحادثة
├── ai2ai.py                # محادثة ذكاءين
├── models/
│   └── transformer.py      # النموذج
├── training/
│   └── train.py            # حلقة التدريب
├── agents/
│   └── agent.py            # الوكيل
├── tools/
│   └── registry.py         # نظام الأدوات
├── data/
│   ├── tokenizer.py        # المدوّن
│   └── dataset.py          # تحميل البيانات
├── rag/
│   ├── simple_rag.py       # RAG أساسي
│   └── better_rag.py       # RAG محسّن
├── memory/
│   └── light_memory.py     # نظام الذاكرة
└── utils/
    ├── auto_config.py      # كشف العتاد
    └── muon.py             # محسن Muon
```

### المتطلبات

- Python 3.10+
- PyTorch 2.0+
- GPU مع CUDA موصى به (يدعم CPU)
- مكتبة `datasets` من HuggingFace

### الهندسة

- **GQA** — 6 رؤوس استعلام، 3 رؤوس KV لكفاءة الذاكرة
- **RoPE** — ترميز موضعي نسبي بدون معلمات مكتسبة
- **SwiGLU** — FFN ببوابة (Swish × GLU)
- **Refresh Gates** — تحديث انتقائي لكل طبقة
- **KV Cache** — توليد O(1) لكل رمز

### الترخيص

للاستخدام غير التجاري فقط. انظر [LICENSE](LICENSE).

---

<a name="fr"></a>

## 🇫🇷 Français

[🇬🇧 English](#en) • [🇸🇦 العربية](#ar) • [🇪🇸 Español](#es) • [🇩🇪 Deutsch](#de)

Un modèle transformer **4,69M paramètres** construit de zéro avec PyTorch — avec GQA, RoPE, SwiGLU, Refresh Gates, inférence avec cache KV, recherche multi-source, outils agent, RAG, et intégration 9Router.

### Fonctionnalités

| Catégorie | Détails |
|---|---|
| **Architecture** | Attention par requêtes groupées (6 têtes requêtes, 3 têtes KV), embeddings positionnels rotatifs, SwiGLU FFN, Refresh Gates, cache KV |
| **Entraînement** | Précision mixte (AMP), accumulation de gradient, optimiseur Muon, perte Unlikelihood + lissage d'étiquettes, sauvegarde de checkpoints |
| **Chat** | Conversation simple avec sortie en continu (token par token) et export de données d'entraînement à la sortie |
| **Agent** | Raisonnement avec outils et recherche multi-source (Wikipedia, Wikidata, arXiv, PubMed, StackExchange, GitHub), calculatrice, lecteur de fichiers |
| **Routeur de recherche** | Routage intelligent de requêtes pour sélectionner les meilleures sources et agréger les résultats (limite 3000 caractères) |
| **Pipeline 9Router** | API AI externe qui compresse les résultats de recherche en contexte propre pour le modèle local |
| **Nettoyeur de sortie** | Suppression des caractères de contrôle, réduction des répétitions, déduplication de phrases, correction d'espacement |
| **RAG** | Génération augmentée de récupération — versions simple et améliorée |
| **AI2AI** | Conversation autonome entre deux modèles avec simulation de tours de parole |
| **Export d'entraînement** | Conversations formatées via 9Router et sauvegardées comme données d'entraînement de haute qualité |

### Démarrage rapide

```bash
# Entraînement de zéro
python main.py

# Conversation avec le modèle
python chat.py

# Agent avec outils
python -m agents.agent

# Conversation IA-IA
python ai2ai.py
```

### Structure du projet

```
PDDAI/
├── config.py               # Hyperparamètres + réglages 9Router
├── main.py                 # Pipeline d'entraînement
├── chat.py                 # Interface de chat
├── ai2ai.py                # Conversation IA-IA
├── models/
│   └── transformer.py      # TransformerModel, GQA, RoPE, SwiGLU
├── training/
│   └── train.py            # Boucle d'entraînement
├── agents/
│   └── agent.py            # Agent avec raisonnement + recherche
├── tools/
│   └── registry.py         # Système d'outils
├── data/
│   ├── tokenizer.py        # Tokenizer BPE
│   └── dataset.py          # Chargement de données
├── rag/
│   ├── simple_rag.py       # RAG basique
│   └── better_rag.py       # RAG amélioré
├── memory/
│   └── light_memory.py     # Système mémoire léger
└── utils/
    ├── auto_config.py      # Détection matérielle
    └── muon.py             # Optimiseur Muon
```

### Prérequis

- Python 3.10+
- PyTorch 2.0+
- GPU CUDA recommandé (fonctionne sur CPU)
- Bibliothèque HuggingFace `datasets`

### Architecture

- **GQA** — 6 têtes requêtes, 3 têtes KV pour une mémoire efficace
- **RoPE** — Encodage positionnel relatif sans paramètres appris
- **SwiGLU** — FFN à porte (Swish × GLU)
- **Refresh Gates** — Mises à jour sélectives par couche
- **Cache KV** — Génération en O(1) par token

### Licence

Usage non commercial uniquement. Voir [LICENSE](LICENSE).

---

<a name="es"></a>

## 🇪🇸 Español

[🇬🇧 English](#en) • [🇸🇦 العربية](#ar) • [🇫🇷 Français](#fr) • [🇩🇪 Deutsch](#de)

Un modelo transformer de **4,69M parámetros** construido desde cero con PyTorch — con GQA, RoPE, SwiGLU, Refresh Gates, inferencia con caché KV, búsqueda multi-fuente, herramientas de agente, RAG e integración con 9Router.

### Características

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

### Inicio rápido

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

### Estructura del proyecto

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

### Requisitos

- Python 3.10+
- PyTorch 2.0+
- GPU con CUDA recomendado (funciona en CPU)
- Biblioteca `datasets` de HuggingFace

### Arquitectura

- **GQA** — 6 cabezas de consulta, 3 cabezas KV para memoria eficiente
- **RoPE** — Codificación posicional relativa sin parámetros aprendidos
- **SwiGLU** — FFN con compuerta (Swish × GLU)
- **Refresh Gates** — Actualizaciones selectivas por capa
- **Caché KV** — Generación O(1) por token

### Licencia

Solo uso no comercial. Ver [LICENSE](LICENSE).

---

<a name="de"></a>

## 🇩🇪 Deutsch

[🇬🇧 English](#en) • [🇸🇦 العربية](#ar) • [🇫🇷 Français](#fr) • [🇪🇸 Español](#es)

Ein **4,69M Parameter** Transformer-Modell von Grund auf mit PyTorch entwickelt — mit GQA, RoPE, SwiGLU, Refresh Gates, KV-Cache-Inferenz, Multi-Quellen-Suche, Agentenwerkzeugen, RAG und 9Router-Integration.

### Funktionen

| Kategorie | Details |
|---|---|
| **Architektur** | Gruppierte Query-Aufmerksamkeit (6 Query-Köpfe, 3 KV-Köpfe), rotatorische Positionscodierungen, SwiGLU FFN, Refresh Gates, KV-Cache |
| **Training** | Gemischte Präzision (AMP), Gradientenakkumulation, Muon-Optimierer, Unlikelihood-Verlust + Label-Smoothing, Checkpoint-Speicherung |
| **Chat** | Einfache Konversation mit tokenweiser Ausgabe und Trainingsexport beim Beenden |
| **Agent** | Werkzeugunterstütztes Denken mit Multi-Quellen-Suche (Wikipedia, Wikidata, arXiv, PubMed, StackExchange, GitHub), Taschenrechner, Dateileser |
| **Such-Router** | Anfrageintent-basierte Quellenauswahl und Ergebnisaggregation (3000 Zeichen Limit) |
| **9Router-Pipeline** | Externe AI-API komprimiert Suchergebnisse zu sauberem Kontext für das lokale Modell |
| **Ausgabe-Reiniger** | Entfernung von Steuerzeichen, Reduzierung von Wiederholungen, Satzdeduplizierung, Abstandsbereinigung |
| **RAG** | Retrieval-augmentierte Generierung — einfache und verbesserte Versionen |
| **AI2AI** | Autonome Zwei-Modell-Konversation mit simuliertem Sprecherwechsel |
| **Trainingsexport** | Gespräche werden via 9Router formatiert und als hochwertige Trainingsdaten gespeichert |

### Schnellstart

```bash
# Von Grund auf trainieren
python main.py

# Mit dem Modell chatten
python chat.py

# Agent mit Werkzeugen
python -m agents.agent

# KI-KI-Konversation
python ai2ai.py
```

### Projektstruktur

```
PDDAI/
├── config.py               # Hyperparameter + 9Router-Einstellungen
├── main.py                 # Trainingspipeline
├── chat.py                 # Chat-Schnittstelle
├── ai2ai.py                # KI-KI-Konversation
├── models/
│   └── transformer.py      # TransformerModel, GQA, RoPE, SwiGLU
├── training/
│   └── train.py            # Trainingsschleife
├── agents/
│   └── agent.py            # Agent mit Werkzeugen + Suche
├── tools/
│   └── registry.py         # Werkzeugsystem
├── data/
│   ├── tokenizer.py        # BPE-Tokenizer
│   └── dataset.py          # Datenladung
├── rag/
│   ├── simple_rag.py       # Einfaches RAG
│   └── better_rag.py       # Verbessertes RAG
├── memory/
│   └── light_memory.py     # Leichtes Gedächtnissystem
└── utils/
    ├── auto_config.py      # Hardware-Erkennung
    └── muon.py             # Muon-Optimierer
```

### Anforderungen

- Python 3.10+
- PyTorch 2.0+
- CUDA-fähige GPU empfohlen (CPU-fähig)
- HuggingFace `datasets` Bibliothek

### Architektur

- **GQA** — 6 Query-Köpfe, 3 KV-Köpfe für effizienten Speicher
- **RoPE** — Relative Positionscodierung ohne gelernte Parameter
- **SwiGLU** — Gegattertes FFN (Swish × GLU)
- **Refresh Gates** — Schichtweise selektive Zustandsaktualisierungen
- **KV-Cache** — O(1) Generierung pro Token

### Lizenz

Nur für nicht-kommerzielle Nutzung. Siehe [LICENSE](LICENSE).
