<div align="center">

# PDDAI

**From-scratch transformer language model — 4.69M parameters**

A decoder-only transformer built from scratch with PyTorch — featuring GQA, RoPE, SwiGLU, Refresh Gates, KV-cached inference, multi-source search, agent tools, RAG, and 9Router integration.

 非商用 仅研究用途 | غير تجاري | Non-commercial only

</div>

<p align="center">
  Select your language / اختر لغتك / 选择语言 / 言語を選択 / Выберите язык / Choisissez votre langue
</p>

<p align="center">
  <a href="i18n/README-en.md">🇬🇧 English</a> •
  <a href="i18n/README-ar.md">🇸🇦 العربية</a> •
  <a href="i18n/README-zh.md">🇨🇳 中文</a> •
  <a href="i18n/README-ja.md">🇯🇵 日本語</a> •
  <a href="i18n/README-ru.md">🇷🇺 Русский</a> •
  <a href="i18n/README-fr.md">🇫🇷 Français</a> •
  <a href="i18n/README-es.md">🇪🇸 Español</a> •
  <a href="i18n/README-de.md">🇩🇪 Deutsch</a> •
  <a href="i18n/README-pt.md">🇵🇹 Português</a> •
  <a href="i18n/README-hi.md">🇮🇳 हिन्दी</a>
</p>

---

## Technical Overview

PDDAI is a **decoder-only transformer** built entirely from scratch in **PyTorch**, with no dependency on HuggingFace Transformers or similar libraries. Every component — attention, position encoding, feed-forward layers, training loop, inference engine — is hand-implemented.

### Architecture

| Component | Implementation |
|---|---|
| **Grouped Query Attention (GQA)** | 6 query heads, 3 key/value heads. Reduces KV-cache memory by 50% vs full MHA while retaining most of the quality. FlashAttention-style tiled matmul. |
| **Rotary Positional Embeddings (RoPE)** | Applies a rotation matrix to query and key vectors at each position, encoding relative position information without learned parameters. Enables better length generalization. |
| **SwiGLU** | Gated FFN with `swish(xW) ⊗ xV` — outperforms standard ReLU or GeLU FFNs at the same parameter count. Hidden dimension 2.5× the standard for equivalent compute. |
| **Refresh Gates** | Per-layer gating mechanism that learns which information to retain or update at each timestep, improving long-context coherence. |
| **KV Cache** | Caches key and value tensors during autoregressive decoding, reducing per-token complexity from O(n²) to O(1). |

### Training

- **Dataset**: FineWeb (HuggingFace `datasets`), filtered and tokenized with a custom **BPE tokenizer** trained on the corpus.
- **Optimizer**: **Muon** — a matrix-free optimizer that applies Newton-Schulz iterations to the gradient moments, achieving faster convergence than AdamW on transformer architectures.
- **Loss**: **Unlikelihood loss** combined with **label smoothing** — penalizes the model for assigning probability to incorrect tokens while softening the target distribution.
- **Precision**: **Automatic Mixed Precision (AMP)** — trains in fp16 with gradient scaling, 2× memory efficiency.
- **Gradient Accumulation**: Simulates larger batch sizes by accumulating gradients over multiple micro-batches.
- **Checkpointing**: Saves `last.pt` (every N steps), `best.pt` (lowest validation loss), and `final.pt` (training complete).

### Inference

- **Streaming**: Tokens are emitted one by one via callback during generation, enabling real-time display.
- **Output Cleaner**: Post-processing pass removes control characters, collapses excessive repetitions (3+ identical words), deduplicates repeated sentences, and normalizes punctuation spacing. Preserves ~90% of the model's original wording.
- **KV Cache**: Attention keys/values from previous steps are cached, making generation linear in sequence length.

### Search & Tool Pipeline

- **Multi-Source Router**: Automatically routes user queries to the most relevant APIs based on intent — Wikipedia (definitions), arXiv (papers), PubMed (medical), StackExchange (Q&A), GitHub (code), Wikidata (entities).
- **9Router Compression**: Search results (up to 3000 chars) are sent to an external 9Router API for summarization into a clean, focused context before being fed to the local model.
- **Output Cleaner** runs on agent responses to smooth tool output.

### Training Data Export

On clean exit, the entire conversation history is sent to 9Router with a formatting prompt that corrects factual errors, fixes grammar, and restructures the dialogue into high-quality training data. The result is saved to `training_conversations/` for future fine-tuning.
