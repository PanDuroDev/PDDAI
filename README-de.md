<div align="center">

# PDDAI

**Transformer-Sprachmodell von Grund auf — 4,69M Parameter**

[🇬🇧 English](README-en.md) • [🇸🇦 العربية](README-ar.md) • [🇨🇳 中文](README-zh.md) • [🇯🇵 日本語](README-ja.md) • [🇷🇺 Русский](README-ru.md) • [🇫🇷 Français](README-fr.md) • [🇪🇸 Español](README-es.md) • [🇵🇹 Português](README-pt.md) • [🇮🇳 हिन्दी](README-hi.md)

</div>

Ein **4,69M Parameter** Transformer-Modell von Grund auf mit PyTorch entwickelt — mit GQA, RoPE, SwiGLU, Refresh Gates, KV-Cache-Inferenz, Multi-Quellen-Suche, Agentenwerkzeugen, RAG und 9Router-Integration.

## Funktionen

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

## Schnellstart

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

## Projektstruktur

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

## Anforderungen

- Python 3.10+
- PyTorch 2.0+
- CUDA-fähige GPU empfohlen (CPU-fähig)
- HuggingFace `datasets` Bibliothek

## Architektur

- **GQA** — 6 Query-Köpfe, 3 KV-Köpfe für effizienten Speicher
- **RoPE** — Relative Positionscodierung ohne gelernte Parameter
- **SwiGLU** — Gegattertes FFN (Swish × GLU)
- **Refresh Gates** — Schichtweise selektive Zustandsaktualisierungen
- **KV-Cache** — O(1) Generierung pro Token

## Lizenz

Nur für nicht-kommerzielle Nutzung. Siehe [LICENSE](LICENSE).
