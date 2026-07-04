# Simple Tokenizer using Hugging Face's tokenizers

import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def train_tokenizer(files, vocab_size=50000, cache_file=None, special_tokens=None):
    # Load cached if vocab stabilized (same size as last train)
    if cache_file and os.path.exists(cache_file):
        cached = Tokenizer.from_file(cache_file)
        cached_size = cached.get_vocab_size()

        if cached_size >= vocab_size:
            print(f"Loaded cached tokenizer: vocab={cached_size}")
            return cached

        # Check if vocab stopped growing by seeing if there's a "prev_vocab" marker
        prev_size_file = cache_file + ".size"
        if os.path.exists(prev_size_file):
            with open(prev_size_file) as f:
                prev_size = int(f.read().strip())
            if prev_size == cached_size:
                print(f"Tokenizer vocab stable at {cached_size} (cannot grow to {vocab_size} with current data)")
                return cached

        print(f"Retraining tokenizer: {cached_size} -> target {vocab_size}...")
        # Save current size before retraining
        with open(prev_size_file, 'w') as f:
            f.write(str(cached_size))

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    if special_tokens is None:
        special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens
    )

    if files:
        tokenizer.train(files=files, trainer=trainer)
        if cache_file:
            tokenizer.save(cache_file)

    return tokenizer

def load_tokenizer(path):
    return Tokenizer.from_file(path)

def encode_text(tokenizer, text, chunk_size=200000):
    ids = []
    total = len(text)
    for i in range(0, total, chunk_size):
        ids.extend(tokenizer.encode(text[i:i+chunk_size]).ids)
        if total > chunk_size:
            print(f"  Encoding: {min(i+chunk_size, total)//1000}k/{total//1000}k chars", end='\r')
    if total > chunk_size:
        print()
    return ids

def decode_text(tokenizer, ids):
    return tokenizer.decode(ids)
