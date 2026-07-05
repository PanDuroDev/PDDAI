# PDDAI

A small transformer language model (4.69M parameters) with GQA, RoPE, SwiGLU, Refresh Gates, and KV-cached inference. Built from scratch with PyTorch.

## Features

- **Transformer** with Grouped Query Attention (6 heads, 3 KV heads), Rotary Positional Embeddings, SwiGLU FFN, and Refresh Gates
- **Training** with mixed precision, gradient accumulation, Muon optimizer, Unlikelihood loss, label smoothing, and checkpointing (last/best/final)
- **Inference** with KV cache for fast generation, streamed output (token by token)
- **Chat** — `python chat.py` for plain conversation
- **Agent** — `python -m agents.agent` with tool-assisted reasoning: web search, calculator, read file
- **Multi-Source Search** — Wikipedia API, Wikidata, arXiv, PubMed, StackExchange, GitHub — routed by query intent
- **9Router Integration** — external AI API compresses search results into clean context before feeding to the local model
- **AI2AI** — `python ai2ai.py` for two-model conversation simulation
- **RAG** — retrieval-augmented generation with simple and better RAG implementations
- **Output Cleaner** — lightweight repetition reduction and response smoothing
- **Training Data Export** — conversations are formatted via 9Router and saved as clean training data on exit

## Quick Start

```bash
# Train from scratch (downloads corpus, builds tokenizer, trains)
python main.py

# Chat with the trained model
python chat.py

# Agent with tools (calculator, search, read)
python -m agents.agent

# AI-to-AI conversation
python ai2ai.py
```

## Project Structure

```
PDDAI/
  config.py              # Hyperparameters + 9Router settings
  main.py                # Training pipeline
  chat.py                # Plain chat interface
  ai2ai.py               # AI-to-AI conversation
  models/
    transformer.py       # TransformerModel, GQA, RoPE, SwiGLU
  training/
    train.py             # Training loop, checkpointing, logging
  agents/
    agent.py             # Agent with tool reasoning + multi-source search
  tools/
    registry.py          # Tool system (web_search, calculator, read_file)
  data/
    tokenizer.py         # Tokenizer training
    dataset.py           # Dataset loading
  rag/
    simple_rag.py        # Basic RAG
    better_rag.py        # Improved RAG with better retrieval
  memory/
    light_memory.py      # Lightweight memory system
  utils/
    auto_config.py       # Hardware auto-detection
    muon.py              # Muon optimizer
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU recommended (falls back to CPU)
- `datasets` library (for corpus download)

## License

Non-Commercial Use Only. See LICENSE for details.
