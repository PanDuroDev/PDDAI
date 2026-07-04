# PDDAI

A small transformer language model (4.69M parameters) with GQA, RoPE, SwiGLU, Refresh Gates, and KV-cached inference. Built from scratch with PyTorch.

## Features

- **Transformer** with Grouped Query Attention (6 heads, 3 KV heads), Rotary Positional Embeddings, SwiGLU FFN, and Refresh Gates
- **Training** with mixed precision, gradient accumulation, Muon optimizer, Unlikelihood loss, label smoothing, and checkpointing (last/best/final)
- **Inference** with KV cache for fast generation
- **Tools** — web_search, calculator, read_file (real execution, the model processes the results)
- **Chat** — `python chat.py` for plain conversation, `python -m agents.agent` with tool support
- **Agent** — tool-assisted reasoning: user selects the tool, model decides the input, tool executes, model writes the final answer

## Quick Start

```bash
# Train from scratch (downloads corpus, builds tokenizer, trains)
python main.py

# Chat with the trained model
python chat.py

# Agent with tools (calculator, search, read)
python -m agents.agent
```

## Project Structure

```
PDDAI/
  config.py              # Hyperparameters
  main.py                # Training pipeline
  chat.py                # Plain chat interface
  models/
    transformer.py       # TransformerModel, GQA, RoPE, SwiGLU
  training/
    train.py             # Training loop, checkpointing, logging
  agents/
    agent.py             # Agent with tool reasoning
  tools/
    registry.py          # Tool system (web_search, calculator, read_file)
  data/
    tokenizer.py         # Tokenizer training
    dataset.py           # Dataset loading
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
