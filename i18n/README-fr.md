<div align="center">

# PDDAI

**Modèle transformer construit de zéro — 4,69M paramètres**

[🇬🇧 English](README-en.md) • [🇸🇦 العربية](README-ar.md) • [🇨🇳 中文](README-zh.md) • [🇯🇵 日本語](README-ja.md) • [🇷🇺 Русский](README-ru.md) • [🇪🇸 Español](README-es.md) • [🇩🇪 Deutsch](README-de.md) • [🇵🇹 Português](README-pt.md) • [🇮🇳 हिन्दी](README-hi.md)

</div>

Un modèle transformer décodeur-only de **4,69M paramètres** construit de zéro avec PyTorch — avec GQA, RoPE, SwiGLU, Refresh Gates, inférence avec cache KV, recherche multi-source, outils agent, RAG, et intégration 9Router.

## Fonctionnalités

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

## Démarrage rapide

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

## Structure du projet

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

## Prérequis

- Python 3.10+
- PyTorch 2.0+
- GPU CUDA recommandé (fonctionne sur CPU)
- Bibliothèque HuggingFace `datasets`

## Architecture

- **GQA** — 6 têtes requêtes, 3 têtes KV pour une mémoire efficace
- **RoPE** — Encodage positionnel relatif sans paramètres appris
- **SwiGLU** — FFN à porte (Swish × GLU)
- **Refresh Gates** — Mises à jour sélectives par couche
- **Cache KV** — Génération en O(1) par token

## Licence

Usage non commercial uniquement. Voir [LICENSE](LICENSE).
