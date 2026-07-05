# Configuration for 4-6M Agent Model

# Model Hyperparameters
VOCAB_SIZE = 5000  # Vocabulary size
EMBED_DIM = 192  # Embedding dimension
NUM_HEADS = 6  # Number of Q attention heads
NUM_KV_HEADS = 3  # Number of KV heads (GQA 6:3)
FF_HIDDEN_DIM = 512  # SwiGLU intermediate dimension
NUM_LAYERS = 9  # Number of transformer layers (deep-thin)
ROPE_THETA = 2500.0  # RoPE theta (2500 better for small models)
MAX_SEQ_LEN = 512  # Maximum sequence length

# Tied embeddings: share token_emb.weight with head (saves ~1M params, improves quality)
USE_TIED_EMBEDDINGS = True

# XSA Refresh Gate: re-inject token embedding into residual stream (layers 4,8)
USE_REFRESH_GATE = True

# Training Hyperparameters
BATCH_SIZE = 8  # Batch size (auto overridden if AUTO_TUNE=True)
EPOCHS = 80  # Number of epochs
LEARNING_RATE = 3e-4  # Learning rate
WARMUP_STEPS = 200  # Number of warmup steps
GRADIENT_CLIPPING = 1.0  # Gradient clipping threshold
GRAD_ACCUM_STEPS = 4  # Gradient accumulation (auto overridden if AUTO_TUNE=True)

# Loss Hyperparameters
LABEL_SMOOTHING = 0.1  # Label smoothing (anti-repetition)
USE_UNLIKELIHOOD = True  # Sequence-level unlikelihood loss

# Optimizer
USE_SOFTWARE_OPTIMIZER = True  # True=Muon+AdamW hybrid, False=AdamW only
MUON_LR = 0.02  # Muon learning rate (for 2D weights)
ADAMW_LR = 3e-4  # AdamW learning rate (for 1D/embeddings)

# Data
DATA_PATH = "data/corpus_rg.txt"

# Optimization
USE_GRADIENT_CHECKPOINTING = False  # 4.69M doesn't need it; enables ~30% faster
USE_MIXED_PRECISION = True  # Mixed precision (auto overridden if AUTO_TUNE=True)
USE_TORCH_COMPILE = False  # torch.compile (Linux/WSL2 recommended)

# Auto-tuning: overrides BATCH_SIZE, GRAD_ACCUM_STEPS, num_workers, AMP
AUTO_TUNE = True

# 9Router API (for summarizing web search results before feeding to local model)
NROUTER_API_URL = "http://localhost:20128/v1"
NROUTER_API_KEY = ""
NROUTER_MODEL = "my-combo"

# Checkpoint / Save settings
SAVE_LAST_MODEL = True  # Save last_model.pt after each epoch
SAVE_BEST_MODEL = True  # Save best_model.pt when val_loss improves
SAVE_EVERY_N_BATCHES = 0  # Save last_model.pt every N batches (0 = disabled)
