import sys
import threading
import time
import json
import urllib.request
from datetime import datetime
if not __import__('importlib').util.find_spec('torch'):
    sys.path.insert(0, 'D:\\torch_site')

import torch
import os
from models.transformer import TransformerModel
from data.tokenizer import load_tokenizer, encode_text, decode_text
import config


class Spinner:
    _spin_chars = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'

    def __init__(self, message=''):
        self.message = message
        self.running = False
        self._thread = None
        self._start = 0.0

    def start(self):
        self._start = time.time()
        self.running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        i = 0
        while self.running:
            sys.stdout.write(f'\r{self._spin_chars[i]} {self.message}   ')
            sys.stdout.flush()
            i = (i + 1) % len(self._spin_chars)
            time.sleep(0.12)

    def stop(self, done=''):
        elapsed = time.time() - self._start
        if elapsed < 1:
            label = f'{elapsed*1000:.0f}ms'
        elif elapsed < 60:
            label = f'{elapsed:.1f}s'
        else:
            label = f'{elapsed//60:.0f}m {elapsed%60:.0f}s'
        self.running = False
        if self._thread:
            self._thread.join(timeout=0.25)
        sys.stdout.write(f'\r{label} {self.message}   \n')
        sys.stdout.flush()


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


def generate(model, tokenizer, prompt, max_tokens=128, temperature=0.9, top_k=40, rep_penalty=1.15, on_token=None):
    input_ids = encode_text(tokenizer, prompt)[-256:]
    x = torch.tensor([input_ids], dtype=torch.long)
    with torch.inference_mode():
        logits, past = model(x, use_cache=True)

    tokens = []
    for i in range(max_tokens):
        logits = logits[:, -1, :] / temperature
        if rep_penalty > 1.0 and tokens:
            for tid in set(tokens[-8:]):
                txt = decode_text(tokenizer, [tid]).strip()
                if txt and (txt[0].isalnum() or txt[0].isalpha()):
                    logits[0, tid] /= rep_penalty
        if top_k > 0:
            top_vals, _ = torch.topk(logits, top_k, dim=-1)
            logits[logits < top_vals[:, -1:]] = float('-inf')
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        tokens.append(next_id)
        if on_token:
            on_token(decode_text(tokenizer, [next_id]))
        if next_id == tokenizer.token_to_id("</s>"):
            break
        x = torch.tensor([[next_id]], dtype=torch.long)
        with torch.inference_mode():
            logits, past = model(x, past_key_values=past, use_cache=True)

    return decode_text(tokenizer, tokens), len(tokens)


def _format_training_data(conversation):
    url = config.NROUTER_API_URL
    key = config.NROUTER_API_KEY
    model = getattr(config, 'NROUTER_MODEL', '') or ''
    if not url:
        return None
    lines = []
    for user_msg, ai_msg in conversation:
        lines.append(f'User: {user_msg}')
        lines.append(f'Assistant: {ai_msg}')
    text = '\n'.join(lines)
    api_url = f'{url}/chat/completions'
    body = {"messages": [{"role": "user", "content": f"Format this conversation as clean training data. Correct factual errors while keeping the original meaning. Output only the formatted conversation with User:/Assistant: markers.\n\n{text}"}], "max_tokens": 512}
    if model:
        body['model'] = model
    try:
        req = urllib.request.Request(api_url, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        if key:
            req.add_header('Authorization', f'Bearer {key}')
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
        try:
            result = json.loads(raw)
            return result['choices'][0]['message']['content']
        except (json.JSONDecodeError, KeyError):
            content = ''
            for line in raw.split('\n'):
                line = line.strip()
                if line.startswith('data: ') and line != 'data: [DONE]':
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk.get('choices', [{}])[0].get('delta', {})
                        if 'content' in delta:
                            content += delta['content']
                    except json.JSONDecodeError:
                        continue
            if content:
                return content
        return text
    except Exception:
        return None


def _save_conversation(conversation):
    if not conversation:
        return
    out_dir = 'training_conversations'
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(out_dir, f'conversation_{ts}.txt')
    spinner = Spinner('Formatting conversation for training')
    spinner.start()
    data = _format_training_data(conversation)
    spinner.stop()
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data if data else '\n'.join(f'User: {u}\nAssistant: {a}' for u, a in conversation))
    print(f'Saved: {path}')


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

    conversation = []

    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in ["quit", "exit", "q"]:
                break

            sys.stdout.write('AI: ')
            sys.stdout.flush()
            tokens = []
            def on_token(t):
                tokens.append(t)
                sys.stdout.write(t)
                sys.stdout.flush()
            response, _ = generate(model, tokenizer, prompt, temperature=0.9, on_token=on_token)
            print('\n')
            conversation.append((prompt, response))
        except KeyboardInterrupt:
            print("\nBye!")
            break

    _save_conversation(conversation)
