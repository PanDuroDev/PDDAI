import sys
if not __import__('importlib').util.find_spec('torch'):
    sys.path.insert(0, 'D:\\torch_site')

import torch
import os
from models.transformer import TransformerModel
from data.tokenizer import load_tokenizer, encode_text, decode_text


def infer_config_from_state(checkpoint):
    """Infer model architecture from checkpoint.
    `checkpoint` is the full dict (not just state_dict).
    First tries 'model_config' metadata, falls back to inference from weights."""
    if 'model_config' in checkpoint:
        cfg = dict(checkpoint['model_config'])
        if 'vocab_size' not in cfg:
            state = checkpoint.get('model_state_dict', checkpoint)
            cfg['vocab_size'] = state['token_emb.weight'].shape[0]
        return cfg

    state = checkpoint.get('model_state_dict', checkpoint)
    embed_dim = state['token_emb.weight'].shape[1]
    vocab_size = state['token_emb.weight'].shape[0]
    tied = state.get('tied_embeddings',
                          'head.weight' not in state or
                          state['head.weight'].shape[0] == state['token_emb.weight'].shape[0])

    rope_cos = state.get('rope_cos')
    q_proj = state.get('blocks.0.attn.q_proj.weight')
    k_proj = state.get('blocks.0.attn.k_proj.weight')

    if rope_cos is not None and q_proj is not None and k_proj is not None:
        head_dim = rope_cos.shape[1] * 2
        num_heads = embed_dim // head_dim
        num_kv_heads = k_proj.shape[0] // head_dim
        num_layers = sum(1 for k in state if k.startswith('blocks.') and k.endswith('attn.q_proj.weight'))
    else:
        num_heads = 6
        num_kv_heads = 3
        num_layers = sum(1 for k in state if k.startswith('blocks.'))

    ffn_gate = state.get('blocks.0.ffn.gate.weight')
    ff_hidden_dim = ffn_gate.shape[0] if ffn_gate is not None else 512

    refresh_layers = sorted(int(k.split('.')[1]) for k in state if 'refresh.gate.weight' in k)

    return {
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'ff_hidden_dim': ff_hidden_dim,
        'num_layers': num_layers,
        'max_len': 256,
        'tied_embeddings': tied,
        'refresh_layers': refresh_layers,
    }


def load_model(checkpoint_path="checkpoints/last_model.pt", tokenizer_path="data/tokenizer.json"):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    cfg = infer_config_from_state(checkpoint)
    state = checkpoint.get('model_state_dict', checkpoint)

    model = TransformerModel(
        vocab_size=cfg['vocab_size'],
        embed_dim=cfg['embed_dim'],
        num_heads=cfg['num_heads'],
        num_kv_heads=cfg['num_kv_heads'],
        ff_hidden_dim=cfg['ff_hidden_dim'],
        num_layers=cfg['num_layers'],
        max_len=cfg['max_len'],
        tied_embeddings=cfg['tied_embeddings'],
        refresh_layers=cfg['refresh_layers'],
    )
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"  WARNING: Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  WARNING: Unexpected keys: {result.unexpected_keys}")
    if cfg['tied_embeddings']:
        model.head.weight = model.token_emb.weight
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    param_mb = total_params * 4 / (1024 * 1024)
    print(f"Model: {total_params/1e6:.2f}M parameters ({param_mb:.0f}MB) | "
          f"{cfg['vocab_size']} vocab, {cfg['embed_dim']}d, "
          f"{cfg['num_heads']}h/{cfg['num_kv_heads']}kv, {cfg['num_layers']}L, "
          f"{'tied' if cfg['tied_embeddings'] else 'untied'} emb | "
          f"refresh={cfg['refresh_layers']}")

    tokenizer = load_tokenizer(tokenizer_path)
    return model, tokenizer


def resolve_model_path(name):
    if name == "last":
        return "saved_model/last_model.pt"
    elif name == "best":
        return "saved_model/best_model.pt"
    elif name == "final":
        return "saved_model/checkpoint.pt"
    else:
        return name


def generate(model, tokenizer, prompt, max_tokens=128, temperature=0.7, top_k=40):
    input_ids = encode_text(tokenizer, prompt)
    input_ids = input_ids[-256:]
    prompt_len = len(input_ids)

    x = torch.tensor([input_ids], dtype=torch.long)
    with torch.inference_mode():
        logits, past = model(x, use_cache=True)

    tokens = []
    for i in range(max_tokens):
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            top_vals, _ = torch.topk(logits, top_k, dim=-1)
            logits[logits < top_vals[:, -1:]] = float('-inf')
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        tokens.append(next_id)
        if next_id == tokenizer.token_to_id("</s>"):
            break
        x = torch.tensor([[next_id]], dtype=torch.long)
        with torch.inference_mode():
            logits, past = model(x, past_key_values=past, use_cache=True)

    return decode_text(tokenizer, tokens)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chat with a trained model")
    parser.add_argument("--model", type=str, default="last",
                        help="Model to load: 'last', 'best', 'final', or a file path")
    args = parser.parse_args()
    path = resolve_model_path(args.model)
    model, tokenizer = load_model(checkpoint_path=path)
    if model is None or tokenizer is None:
        exit(1)

    print("\n" + "=" * 50)
    print("AI Chat Ready! Type 'quit' to exit.")
    print("=" * 50 + "\n")

    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in ["quit", "exit", "q"]:
                break

            response = generate(model, tokenizer, prompt, temperature=0.9)
            print(f"AI: {response}\n")
        except KeyboardInterrupt:
            print("\nBye!")
            break
