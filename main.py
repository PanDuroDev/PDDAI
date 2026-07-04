# Main Execution Script with diverse data and checkpointing

import sys
if not __import__('importlib').util.find_spec('torch'):
    sys.path.insert(0, 'D:\\torch_site')

import torch
import os
import config
from models.transformer import TransformerModel
from data.dataset import create_dataloaders
from training.train import train_model
from data.tokenizer import train_tokenizer, encode_text
from utils.auto_config import auto_config, tune_cuda, set_num_threads


def generate_synthetic_fallback():
    texts = []
    for _ in range(50):
        texts.append("""The universe began with the Big Bang 13.8 billion years ago. Stars are nuclear fusion reactors. DNA stores genetic information. Photosynthesis converts sunlight into chemical energy. Gravity attracts objects with mass. Evolution by natural selection was discovered by Charles Darwin. Quantum mechanics describes subatomic particles. Relativity theory revolutionized physics. The human brain has 86 billion neurons. Electricity flows through conductors. The periodic table organizes elements. Mesopotamia emerged between the Tigris and Euphrates. Ancient Egypt built the pyramids. The Roman Empire dominated the Mediterranean. The Islamic Golden Age advanced mathematics and medicine. The Renaissance revived classical learning. The Industrial Revolution mechanized production. Artificial intelligence creates machines that think. Machine learning finds patterns in data. Neural networks mimic the brain. Deep learning powers complex tasks. Natural language processing enables chatbots. The human body has 206 bones and 600 muscles. The heart pumps blood throughout the body. Vaccines train the immune system. Antibiotics kill bacteria. Aristotle wrote about ethics and politics. Confucius taught about moral conduct. Descartes said I think therefore I am. Nietzsche wrote about the will to power. Plato described the theory of forms. Socrates questioned everything. Mathematics studies patterns and relationships. Calculus deals with change and motion. Geometry explores shapes and spaces. Algebra uses symbols for equations. Oceans cover 71 percent of Earth. The Amazon rainforest produces oxygen. Mount Everest is the highest peak. The Great Barrier Reef is the largest living structure.""")
    full_text = "\n\n".join(texts)
    os.makedirs("data", exist_ok=True)
    with open("data/corpus_rg.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    wc = len(full_text.split())
    print(f"Fallback corpus: {wc:,} words")
    return "data/corpus_rg.txt"


def download_corpus():
    """
    Mixture (by doc count): FineWeb-Edu 50%, Cosmopedia v2 20%,
    SmolLM FW-Edu-dedup 15%, TinyStories 10%, OASST1 3%, WikiText-2 2%.
    ~34k docs total, ~70-90 MB — maximises signal density per param.
    """
    print("Downloading real-world training data...")
    try:
        from datasets import load_dataset

        all_texts = []

        for split in ["train", "validation", "test"]:
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split=split)
            all_texts.append("\n\n".join(ds["text"]))
            print(f"  WikiText-2 {split}: {len(ds)} examples")

        try:
            ds_ts = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            c = 0
            for row in ds_ts:
                txt = row.get("text", "")
                if len(txt) > 100:
                    all_texts.append(txt)
                    c += 1
                    if c >= 3000:
                        break
            print(f"  TinyStories: {c} docs")
        except Exception as e:
            print(f"  TinyStories: {e}")

        try:
            ds_fw = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
            c = 0
            for row in ds_fw:
                txt = row.get("text", "")
                if len(txt) > 200:
                    all_texts.append(txt)
                    c += 1
                    if c >= 15000:
                        break
            print(f"  FineWeb-Edu: {c} docs")
        except Exception as e:
            print(f"  FineWeb-Edu: {e}")

        try:
            ds_cosmo = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
            c = 0
            for row in ds_cosmo:
                txt = row.get("text", "")
                if len(txt) > 200:
                    all_texts.append(txt)
                    c += 1
                    if c >= 6000:
                        break
            print(f"  Cosmopedia v2: {c} docs")
        except Exception as e:
            print(f"  Cosmopedia v2: {e}")

        try:
            ds_sm = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)
            c = 0
            for row in ds_sm:
                txt = row.get("text", "")
                if len(txt) > 100:
                    all_texts.append(txt)
                    c += 1
                    if c >= 5000:
                        break
            print(f"  SmolLM FW-Edu-dedup: {c} docs")
        except Exception as e:
            print(f"  SmolLM FW-Edu-dedup: {e}")

        try:
            ds_oasst = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
            seen = set()
            c = 0
            for row in ds_oasst:
                txt = row.get("text", "")
                role = row.get("role", "")
                if role == "assistant" and len(txt) > 50 and txt not in seen:
                    seen.add(txt)
                    all_texts.append(txt)
                    c += 1
                    if c >= 1000:
                        break
            print(f"  OASST1 (assistant replies): {c} docs")
        except Exception as e:
            print(f"  OASST1: {e}")

        final_text = "\n\n".join(all_texts)
        os.makedirs("data", exist_ok=True)
        with open("data/corpus_rg.txt", "w", encoding="utf-8") as f:
            f.write(final_text)
        wc = len(final_text.split())
        cc = len(final_text)
        print(f"Total corpus: {wc:,} words, {cc:,} chars")
        return "data/corpus_rg.txt"
    except Exception as e:
        print(f"Download failed: {e}")
        print("Using fallback synthetic corpus...")
        return generate_synthetic_fallback()


def main():
    corpus_path = download_corpus()

    if os.path.exists("data/tokenizer.json"):
        print("Loading existing tokenizer...")

    special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]
    tokenizer = train_tokenizer(
        [corpus_path],
        vocab_size=config.VOCAB_SIZE,
        cache_file="data/tokenizer.json",
        special_tokens=special_tokens
    )

    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = encode_text(tokenizer, text)
    seq_len = config.MAX_SEQ_LEN
    stride = seq_len // 2

    ids_t = torch.tensor(ids, dtype=torch.long)
    num_seq = (len(ids_t) - seq_len) // stride + 1
    if (len(ids_t) - seq_len) % stride == 0:
        num_seq -= 1
    X = torch.as_strided(ids_t, (num_seq, seq_len), (stride, 1)).contiguous()
    y = torch.as_strided(ids_t[1:], (num_seq, seq_len), (stride, 1)).contiguous()
    print(f"Data shape: {X.shape}, {y.shape}")

    cfg = auto_config()
    tune_cuda()
    set_num_threads()

    if cfg['device_name'] == 'cpu' and torch.cuda.is_available():
        print("WARNING: CUDA is available but not detected!")
    if cfg['device_name'] == 'cpu' and not torch.cuda.is_available():
        print("WARNING: Running on CPU!")

    print(f"Auto config: {cfg['device_name']}, VRAM={cfg['vram_gb']}GB, batch={cfg['batch_size']}, workers={cfg['num_workers']}, amp={cfg['use_amp']}")

    batch_size = cfg['batch_size'] if config.AUTO_TUNE else config.BATCH_SIZE
    num_workers = cfg['num_workers'] if config.AUTO_TUNE else 0
    use_amp = cfg['use_amp'] if config.AUTO_TUNE else config.USE_MIXED_PRECISION
    grad_accum_steps = cfg['grad_accum_steps'] if config.AUTO_TUNE else config.GRAD_ACCUM_STEPS

    n = len(X)
    indices = torch.randperm(n)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    split = int(n * 0.95)
    X_train, X_val = X_shuffled[:split], X_shuffled[split:]
    y_train, y_val = y_shuffled[:split], y_shuffled[split:]

    dataloader = create_dataloaders(
        X_train, y_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_dataloader = create_dataloaders(
        X_val, y_val,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )
    print(f"Train: {len(X_train)} sequences, Val: {len(X_val)} sequences")

    if config.EMBED_DIM % config.NUM_HEADS != 0:
        print(f"ERROR: EMBED_DIM ({config.EMBED_DIM}) must be divisible by NUM_HEADS ({config.NUM_HEADS})")
        print(f"  {config.EMBED_DIM} / {config.NUM_HEADS} = {config.EMBED_DIM / config.NUM_HEADS:.2f} (not integer)")
        sys.exit(1)

    model = TransformerModel(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        ff_hidden_dim=config.FF_HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        max_len=config.MAX_SEQ_LEN,
        num_kv_heads=config.NUM_KV_HEADS,
        tied_embeddings=config.USE_TIED_EMBEDDINGS,
        rope_theta=config.ROPE_THETA,
        refresh_layers=[4, 8] if config.USE_REFRESH_GATE else [],
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e9:.2f}B ({total_params:,})")

    if config.USE_TORCH_COMPILE:
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    device = cfg['device']
    print(f"Using device: {device}")

    train_model(
        model, dataloader, device,
        val_dataloader=val_dataloader,
        seq_len=config.MAX_SEQ_LEN,
        epochs=config.EPOCHS,
        lr=config.LEARNING_RATE,
        grad_accum_steps=grad_accum_steps,
        use_amp=use_amp,
        use_checkpoint=config.USE_GRADIENT_CHECKPOINTING,
        label_smoothing=config.LABEL_SMOOTHING,
        use_unlikelihood=config.USE_UNLIKELIHOOD,
        use_muon=config.USE_SOFTWARE_OPTIMIZER,
        warmup_steps=config.WARMUP_STEPS,
        muon_lr=config.MUON_LR,
        adamw_lr=config.ADAMW_LR,
        gradient_clipping=config.GRADIENT_CLIPPING,
        save_last_model=config.SAVE_LAST_MODEL,
        save_best_model=config.SAVE_BEST_MODEL,
        save_every_n_batches=config.SAVE_EVERY_N_BATCHES,
    )

    os.makedirs("saved_model", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.get_vocab_size(),
        'tied_embeddings': config.USE_TIED_EMBEDDINGS,
        'model_config': {
            'embed_dim': config.EMBED_DIM,
            'num_heads': config.NUM_HEADS,
            'num_kv_heads': config.NUM_KV_HEADS,
            'ff_hidden_dim': config.FF_HIDDEN_DIM,
            'num_layers': config.NUM_LAYERS,
            'max_len': config.MAX_SEQ_LEN,
            'rope_theta': config.ROPE_THETA,
            'vocab_size': tokenizer.get_vocab_size(),
            'tied_embeddings': config.USE_TIED_EMBEDDINGS,
            'refresh_layers': [4, 8] if config.USE_REFRESH_GATE else [],
        },
    }, "saved_model/checkpoint_final.pt")
    print(f"Final model saved to saved_model/checkpoint_final.pt (vocab={tokenizer.get_vocab_size()})")


if __name__ == '__main__':
    main()
