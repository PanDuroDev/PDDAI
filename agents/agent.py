"""
Agent system designed for 4.69M parameter model.

Principles:
- Code handles all tool logic — model only generates natural text
- No formatting, no labels — raw text only (model trained on raw text)
- Three activation modes: auto-detect, explicit command, model suggestion
- Repetition prevention built into generation
"""

import torch
import re
from tools.registry import ToolRegistry, make_default_registry

# ── Intent detection rules (code-based) ───────────────────────────────

def _extract_expr(text):
    m = re.search(r'[\d][\d+\-*/().,%^ ]*', text)
    return m.group(0).strip() if m else text

def _extract_query(text):
    for kw in ['look up', 'search for', 'find', 'tell me about', 'google', 'search']:
        if kw in text.lower():
            idx = text.lower().index(kw) + len(kw)
            return text[idx:].strip().lstrip('"\' ').rstrip('"?\'.,!')
    return text

def _extract_path(text):
    m = re.search(r'(?:read|open|show)\s+file\s+(.+)', text, re.IGNORECASE)
    return m.group(1).strip() if m else text

_INTENT = [
    (r'\b(?:sum|add|plus|total|calculate|compute)\b', 'calculator', 'expression', _extract_expr),
    (r'\b(?:search|find|look\s*up|google|tell\s*me\s*about)\b', 'web_search', 'query', _extract_query),
    (r'\b(?:read|open|show)\s+file\b', 'read_file', 'path', _extract_path),
]

_COMMANDS = {
    'calculator': ('calculator', 'expression'), 'calc': ('calculator', 'expression'),
    'search': ('web_search', 'query'),           'web_search': ('web_search', 'query'),
    'read': ('read_file', 'path'),               'read_file': ('read_file', 'path'),
}

_SUGGEST_RE = re.compile(r'Tool\s*:\s*(\w+)\s*\(([^)]*)\)', re.IGNORECASE)


class Agent:
    def __init__(self, model, tokenizer, device='cpu',
                 tools: ToolRegistry = None, max_tokens=256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.tools = tools or make_default_registry()
        self.max_tokens = max_tokens
        self.history: list[tuple[str, str]] = []

    def encode(self, text):
        from data.tokenizer import encode_text
        return encode_text(self.tokenizer, text)

    def decode(self, ids):
        from data.tokenizer import decode_text
        return decode_text(self.tokenizer, ids)

    def generate(self, prompt, max_new=96, temperature=0.9, top_k=40, rep_penalty=1.15):
        input_ids = self.encode(prompt)
        input_ids = input_ids[-256:]
        prompt_len = len(input_ids)

        x = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        with torch.inference_mode():
            logits, past = self.model(x, use_cache=True)

        tokens = []
        for i in range(max_new):
            logits = logits[:, -1, :] / temperature

            if rep_penalty > 1.0 and tokens:
                recent = set(tokens[-8:])
                for tid in recent:
                    logits[0, tid] /= rep_penalty

            if top_k > 0:
                vals, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < vals[:, -1:]] = float('-inf')

            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            tokens.append(next_id)

            if next_id == self.tokenizer.token_to_id("</s>"):
                break

            x = torch.tensor([[next_id]], dtype=torch.long).to(self.device)
            with torch.inference_mode():
                logits, past = self.model(x, past_key_values=past, use_cache=True)

        return self.decode(tokens)

    def _build_prompt(self, user_input, result=None):
        parts = []
        for u, a in self.history[-2:]:
            parts.append(u)
            parts.append(a)
        parts.append(user_input)
        if result:
            parts.append(result)
        return "\n".join(parts)

    def _detect_intent(self, text):
        lower = text.lower()
        for pat, tool, param, extract in _INTENT:
            if re.search(pat, lower):
                value = extract(text)
                return tool, param, value
        return None

    def _parse_command(self, text):
        first = text.strip().split()[0].lower()
        if first in _COMMANDS:
            return _COMMANDS[first] + (text[len(first):].strip(),)
        return None

    def _find_suggestions(self, text):
        results = []
        for name, val in _SUGGEST_RE.findall(text):
            name = name.lower()
            if name in _COMMANDS:
                tool, param = _COMMANDS[name]
                results.append((tool, param, val.strip()))
        return results

    def _execute(self, name, param, value):
        return self.tools.call(name, **{param: value or '?'})

    def run(self, user_input):
        intent = self._detect_intent(user_input)
        if intent:
            tool, param, value = intent
            result = self._execute(tool, param, value)
            prompt = self._build_prompt(user_input, result)
            response = self.generate(prompt, max_new=96)
            self.history.append((user_input, response))
            return response

        cmd = self._parse_command(user_input)
        if cmd:
            tool, param, value = cmd
            result = self._execute(tool, param, value)
            prompt = self._build_prompt(user_input, result)
            response = self.generate(prompt, max_new=96)
            self.history.append((user_input, response))
            return response

        prompt = self._build_prompt(user_input)
        response = self.generate(prompt, max_new=128)

        suggestions = self._find_suggestions(response)
        if suggestions:
            for tool, param, value in suggestions:
                result = self._execute(tool, param, value)
                cont = self.generate(result, max_new=64)
                response += "\n" + cont

        self.history.append((user_input, response))
        return response

    def chat(self):
        print("=" * 50)
        print("Agent ready. Type 'quit' to exit.")
        print("=" * 50)

        while True:
            try:
                user = input("\nYou: ")
                if user.lower() in ("quit", "exit", "q"):
                    break
                print(f"AI: {self.run(user)}")
            except KeyboardInterrupt:
                print("\nBye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="last")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    args = parser.parse_args()
    from chat import load_model, resolve_model_path
    path = resolve_model_path(args.model)
    model, tokenizer = load_model(path, args.tokenizer)
    if model and tokenizer:
        Agent(model, tokenizer).chat()
