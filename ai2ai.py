import sys
import json
import os
import torch
import time
import random
import urllib.request
from datetime import datetime
from chat import load_model, resolve_model_path, generate, Spinner
from data.tokenizer import encode_text
import config


class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def color(text, c):
    return f'{c}{text}{Colors.RESET}'


def _format_training(conversation):
    url = config.NROUTER_API_URL
    key = config.NROUTER_API_KEY
    model_name = getattr(config, 'NROUTER_MODEL', '') or ''
    if not url:
        return None
    lines = []
    for speaker, text in conversation:
        lines.append(f'{speaker}: {text}')
    text = '\n'.join(lines)
    api_url = f'{url}/chat/completions'
    body = {"messages": [{"role": "user", "content": f"Format this AI-to-AI conversation as high-quality training data. Correct factual errors, fix grammar, and ensure both sides are coherent and informative. Preserve the multi-turn dialogue structure. Output only the conversation with speaker labels.\n\n{text}"}], "max_tokens": 512}
    if model_name:
        body['model'] = model_name
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = f'Bearer {key}'
    try:
        req = urllib.request.Request(api_url, data=json.dumps(body).encode(), headers=headers)
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
    path = os.path.join(out_dir, f'ai2ai_{ts}.txt')
    spinner = Spinner('Formatting AI2AI conversation for training')
    spinner.start()
    data = _format_training(conversation)
    spinner.stop()
    if data:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            for speaker, text in conversation:
                f.write(f'{speaker}: {text}\n')
    print(f'Saved: {path}')


def ai2ai(model, tokenizer, seed_msg, turns=12, delay_range=(3.0, 5.0), conversation=None):
    print(color('\n' + '=' * 60, Colors.CYAN + Colors.BOLD))
    print(color('  AI2AI — Two Models Talking to Each Other', Colors.CYAN + Colors.BOLD))
    print(color('=' * 60, Colors.CYAN + Colors.BOLD))
    print(color(f'Seed: "{seed_msg}"\n', Colors.YELLOW))

    params_a = dict(temperature=0.9, top_k=40, max_tokens=96, rep_penalty=1.15)
    params_b = dict(temperature=0.75, top_k=50, max_tokens=96, rep_penalty=1.2)

    ctx_a = seed_msg
    ctx_b = ''

    for turn in range(turns):
        delay = random.uniform(*delay_range)
        time.sleep(delay)

        sys.stdout.write(color(f'[{turn+1}] A: ', Colors.GREEN + Colors.BOLD))
        sys.stdout.flush()
        tokens = []
        def on_a(t):
            tokens.append(t)
            sys.stdout.write(t)
            sys.stdout.flush()
        out_a, _ = generate(model, tokenizer, ctx_a, on_token=on_a, **params_a)
        out_a = out_a.strip()
        if not out_a:
            out_a = '(silence)'
        print()
        if conversation is not None:
            conversation.append(('A', out_a))

        ctx_b = out_a

        delay = random.uniform(*delay_range)
        time.sleep(delay)

        sys.stdout.write(color(f'[{turn+1}] B: ', Colors.BLUE + Colors.BOLD))
        sys.stdout.flush()
        tokens = []
        def on_b(t):
            tokens.append(t)
            sys.stdout.write(t)
            sys.stdout.flush()
        out_b, _ = generate(model, tokenizer, ctx_b, on_token=on_b, **params_b)
        out_b = out_b.strip()
        if not out_b:
            out_b = '(silence)'
        print()
        if conversation is not None:
            conversation.append(('B', out_b))

        ctx_a = out_b

        if out_a == out_b:
            print(color('(Both said the same thing — stopping)', Colors.RED + Colors.DIM))
            break

    print(color('\n' + '=' * 60, Colors.CYAN + Colors.BOLD))
    print(color(f'  Conversation finished after {turn+1} rounds', Colors.CYAN + Colors.BOLD))
    print(color('=' * 60, Colors.CYAN + Colors.BOLD))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='AI2AI — Two models talk to each other')
    parser.add_argument('--model', default='last', help='model to load')
    parser.add_argument('--tokenizer', default='data/tokenizer.json', help='tokenizer path')
    parser.add_argument('--turns', type=int, default=12, help='max conversation turns')
    parser.add_argument('--min-delay', type=float, default=3.0, help='min delay between responses')
    parser.add_argument('--max-delay', type=float, default=5.0, help='max delay between responses')
    args = parser.parse_args()

    path = resolve_model_path(args.model)
    model, tokenizer = load_model(path, args.tokenizer)
    if model is None or tokenizer is None:
        exit(1)

    print(color('AI2AI started.', Colors.CYAN + Colors.BOLD))
    print('You send ONE message, then the two models talk to each other.')
    seed = input(color('You: ', Colors.YELLOW + Colors.BOLD))

    conversation = [('User', seed)]

    try:
        ai2ai(model, tokenizer, seed, turns=args.turns, delay_range=(args.min_delay, args.max_delay), conversation=conversation)
    except KeyboardInterrupt:
        print(color('\nInterrupted.', Colors.RED))
    finally:
        _save_conversation(conversation)


if __name__ == '__main__':
    main()
