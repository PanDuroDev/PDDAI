import os
import torch
from tools.registry import ToolRegistry, make_default_registry


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
        'max_len': 512,
        'tied_embeddings': tied,
        'refresh_layers': refresh_layers,
    }


def load_model(checkpoint_path="saved_model/last_model.pt", tokenizer_path="data/tokenizer.json"):
    from models.transformer import TransformerModel
    from data.tokenizer import load_tokenizer

    print("Loading model...")
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None, None
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Checkpoint corrupted: {e}")
        return None, None
    tokenizer = load_tokenizer(tokenizer_path)
    state = checkpoint.get('model_state_dict', checkpoint)
    cfg = infer_config_from_state(checkpoint)

    model = TransformerModel(
        vocab_size=cfg['vocab_size'],
        embed_dim=cfg['embed_dim'],
        num_heads=cfg['num_heads'],
        ff_hidden_dim=cfg['ff_hidden_dim'],
        num_layers=cfg['num_layers'],
        max_len=cfg.get('max_len', 512),
        num_kv_heads=cfg['num_kv_heads'],
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
    print(f"Loaded: {cfg['vocab_size']} vocab, {cfg['embed_dim']} dim, "
          f"{cfg['num_heads']} heads/{cfg['num_kv_heads']} kv, {cfg['num_layers']} layers, "
          f"{'tied' if cfg['tied_embeddings'] else 'untied'} emb, "
          f"refresh={cfg['refresh_layers']}")
    return model, tokenizer


def resolve_model_path(model_spec):
    if model_spec == "last":
        return "saved_model/last_model.pt"
    if model_spec == "best":
        return "saved_model/best_model.pt"
    if model_spec == "final":
        return "saved_model/checkpoint_final.pt"
    return model_spec


SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's questions clearly and naturally."""


class Agent:
    def __init__(self, model, tokenizer, device='cpu',
                 tools: ToolRegistry = None, max_tokens=512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.tools = tools or make_default_registry()
        self.max_tokens = max_tokens
        self.history = []
        self._tool_map = {
            'calculator': ('calculator', 'expression'),
            'calc': ('calculator', 'expression'),
            'search': ('web_search', 'query'),
            'web_search': ('web_search', 'query'),
            'read': ('read_file', 'path'),
            'read_file': ('read_file', 'path'),
        }
        self._tool_prompts = {
            'calculator': {
                'input': "The user wants to calculate something.\nWhat math expression should I evaluate?",
                'result': 'Result: {result}',
            },
            'web_search': {
                'input': "The user wants to search the web.\nWhat should I search for?",
                'result': 'Search result: {result}',
            },
            'read_file': {
                'input': "The user wants to read a file.\nWhat file path should I read?",
                'result': 'File content:\n{result}',
            },
        }

    def encode(self, text):
        from data.tokenizer import encode_text
        return encode_text(self.tokenizer, text)

    def decode(self, ids):
        from data.tokenizer import decode_text
        return decode_text(self.tokenizer, ids)

    def generate(self, prompt, max_new=128, temperature=0.7, top_k=40):
        input_ids = self.encode(prompt)
        input_ids = input_ids[-256:]
        prompt_len = len(input_ids)

        x = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        with torch.inference_mode():
            logits, past = self.model(x, use_cache=True)

        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            top_vals, _ = torch.topk(logits, top_k, dim=-1)
            logits[logits < top_vals[:, -1:]] = float('-inf')
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        input_ids.append(next_id)
        if next_id == self.tokenizer.token_to_id("</s>"):
            return self.decode(input_ids[prompt_len:])

        for _ in range(max_new - 1):
            x = torch.tensor([[next_id]], dtype=torch.long).to(self.device)
            with torch.inference_mode():
                logits, past = self.model(x, past_key_values=past, use_cache=True)

            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                top_vals, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < top_vals[:, -1:]] = float('-inf')
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            input_ids.append(next_id)
            if next_id == self.tokenizer.token_to_id("</s>"):
                break

        return self.decode(input_ids[prompt_len:])

    def format_history(self):
        parts = [SYSTEM_PROMPT]
        for turn in self.history:
            if 'assistant' in turn:
                parts.append(f"\nUser: {turn.get('user', '')}")
                parts.append(f"\nAssistant: {turn['assistant']}")
        return "\n".join(parts)

    def _parse_tool_request(self, text):
        """Parse 'toolname' or 'toolname value'. Returns (tool_name, param_name, user_value) or None."""
        text = text.strip()
        if not text:
            return None
        first_word = text.split()[0].lower()
        rest = text[len(first_word):].strip()
        if first_word in self._tool_map:
            tool_name, param_name = self._tool_map[first_word]
            return tool_name, param_name, rest
        return None

    def _execute_with_tool(self, tool_name, param_name, user_value):
        """3-step: model decides input → execute real tool → model writes answer."""
        tool = self.tools.get(tool_name)
        prompts = self._tool_prompts[tool_name]

        if user_value:
            tool_input = user_value
        else:
            tool_input = self.generate(
                prompts['input'],
                max_new=64, temperature=0.5, top_k=20
            ).strip()
            if not tool_input:
                tool_input = "unknown"

        result = self.tools.call(tool_name, **{param_name: tool_input})

        answer_prompt = (
            f"{prompts['result'].format(result=result)}\n\n"
            f"Write a natural answer."
        )
        final = self.generate(answer_prompt, max_new=128, temperature=0.7)
        return final

    def run(self, user_input, max_steps=5):
        parsed = self._parse_tool_request(user_input)
        if parsed:
            tool_name, param_name, user_value = parsed
            response = self._execute_with_tool(tool_name, param_name, user_value)
            self.history.append({"user": user_input, "assistant": response})
            return response

        self.history.append({"user": user_input})
        prompt = self.format_history() + f"\nUser: {user_input}\nAssistant: "
        response = self.generate(prompt, max_new=128, temperature=0.7)
        self.history[-1]["assistant"] = response
        return response

    def chat(self):
        print("=" * 50)
        print("Agent Ready! Type 'quit' to exit.")
        print("Tools: calculator, search, read")
        print("  calculator  - evaluate math")
        print("  search      - search the web")
        print("  read        - read a file")
        print("=" * 50)

        while True:
            try:
                user = input("\nYou: ")
                if user.lower() in ["quit", "exit", "q"]:
                    break
                response = self.run(user)
                print(f"AI: {response}")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the AI agent")
    parser.add_argument("--model", type=str, default="last",
                        help="Model to load: 'last', 'best', 'final', or a file path")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json",
                        help="Path to tokenizer file")
    args = parser.parse_args()
    path = resolve_model_path(args.model)
    model, tokenizer = load_model(checkpoint_path=path, tokenizer_path=args.tokenizer)
    if model is None or tokenizer is None:
        exit(1)
    agent = Agent(model, tokenizer)
    agent.chat()
