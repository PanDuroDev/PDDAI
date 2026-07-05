# PDDAI

*From-scratch transformer language model with GQA, RoPE, SwiGLU, Refresh Gates, and tool-assisted reasoning*

**4.69M parameters** — trained from scratch using PyTorch. Features KV-cached inference, multi-source search integration, agent tools, RAG pipeline, and AI-to-AI conversation.

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
# Train from scratch (downloads FineWeb corpus, builds tokenizer, trains)
python main.py

# Chat with the trained model
python chat.py

# Agent with tool access
python -m agents.agent

# AI-to-AI autonomous conversation
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
│   ├── tokenizer.py        # Tokenizer training (BPE)
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
- CUDA-capable GPU recommended (falls back to CPU with reduced performance)
- HuggingFace `datasets` library (for training corpus download)

## Architecture Details

The model is a decoder-only transformer with:

- **Grouped Query Attention** — 6 query heads, 3 key/value heads for efficient memory usage
- **Rotary Positional Embeddings** — relative position encoding without learned parameters
- **SwiGLU** — gated FFN activation function (Swish × Gated Linear Unit)
- **Refresh Gates** — per-layer gating mechanism for selective state updates
- **KV Cache** — cached key/value states during inference for O(1) per-token generation

## License

Non-Commercial Use Only. See [LICENSE](LICENSE) for details.
