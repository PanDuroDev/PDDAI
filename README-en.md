<div align="center">

# PDDAI

**From-scratch transformer language model — 4.69M parameters**

[🇸🇦 العربية](README-ar.md) • [🇨🇳 中文](README-zh.md) • [🇯🇵 日本語](README-ja.md) • [🇷🇺 Русский](README-ru.md) • [🇫🇷 Français](README-fr.md) • [🇪🇸 Español](README-es.md) • [🇩🇪 Deutsch](README-de.md) • [🇵🇹 Português](README-pt.md) • [🇮🇳 हिन्दी](README-hi.md)

</div>

A **4.69M parameter** decoder-only transformer built from scratch with PyTorch — featuring GQA, RoPE, SwiGLU, Refresh Gates, KV-cached inference, multi-source search, agent tools, RAG, and 9Router integration.

## Features

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

## Quick Start

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

## Project Structure

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

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU recommended (falls back to CPU)
- HuggingFace `datasets` library

## Architecture

- **Grouped Query Attention** — 6 query heads, 3 KV heads for efficient memory
- **Rotary Positional Embeddings** — relative position encoding, no learned params
- **SwiGLU** — gated FFN (Swish × GLU)
- **Refresh Gates** — per-layer selective state updates
- **KV Cache** — O(1) per-token generation

## License

Non-Commercial Use Only. See [LICENSE](LICENSE).
