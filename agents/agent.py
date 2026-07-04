"""
Agent system for 4.69M parameter model.
User picks tools primarily; model can suggest secondarily.
"""

import torch
import re
from tools.registry import ToolRegistry, make_default_registry

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

    def encode(self, text):
        from data.tokenizer import encode_text
        return encode_text(self.tokenizer, text)

    def decode(self, ids):
        from data.tokenizer import decode_text
        return decode_text(self.tokenizer, ids)

    def generate(self, prompt, max_new=96, temperature=0.9, top_k=40, rep_penalty=1.15):
        input_ids = self.encode(prompt)
        input_ids = input_ids[-256:]
        x = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        with torch.inference_mode():
            logits, past = self.model(x, use_cache=True)

        tokens = []
        for i in range(max_new):
            logits = logits[:, -1, :] / temperature
            if rep_penalty > 1.0 and tokens:
                for tid in set(tokens[-8:]):
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
        return self.decode(tokens), len(tokens)

    def _count_tokens(self, text):
        return len(self.encode(text))

    def _parse_command(self, text):
        t = text.strip()
        if not t:
            return None
        first = t.split()[0].lower()
        name = first.lstrip('\\/')
        if name in _COMMANDS:
            return _COMMANDS[name] + (t[len(first):].strip(),)
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
        cmd = self._parse_command(user_input)
        if cmd:
            tool, param, value = cmd
            result = self._execute(tool, param, value)
            prompt = f"{value} = {result}" if tool == 'calculator' else result
            response, _ = self.generate(prompt, max_new=96)
            return response

        response, _ = self.generate(user_input, max_new=128)

        suggestions = self._find_suggestions(response)
        if suggestions:
            response = re.sub(_SUGGEST_RE.pattern, '', response).strip()
            response = re.sub(r' +', ' ', response)
            for tool, param, value in suggestions:
                result = self._execute(tool, param, value)
                cont, _ = self.generate(result, max_new=64)
                response += "\n" + cont

        return response

    def chat(self):
        print("Tools: \\calculator  \\search  \\read  (quit to exit)")

        while True:
            try:
                user = input("\n> ")
                if user.lower() in ("quit", "exit", "q"):
                    break
                prompt_tok = self._count_tokens(user)
                response = self.run(user)
                out_tok = self._count_tokens(response)
                print(f"{response}  [{out_tok}]")
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
