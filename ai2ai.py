import sys
import torch
import time
import random
from chat import load_model, resolve_model_path, generate
from data.tokenizer import load_tokenizer, encode_text


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


def ai2ai(model, tokenizer, seed_msg, turns=12, delay_range=(3.0, 5.0)):
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

        out_a, _ = generate(model, tokenizer, ctx_a, **params_a)
        out_a = out_a.strip()
        if not out_a:
            out_a = '(silence)'
        print(color(f'  [{turn+1}] A:', Colors.GREEN + Colors.BOLD), color(out_a, Colors.GREEN))
        sys.stdout.flush()

        ctx_b = out_a

        delay = random.uniform(*delay_range)
        time.sleep(delay)

        out_b, _ = generate(model, tokenizer, ctx_b, **params_b)
        out_b = out_b.strip()
        if not out_b:
            out_b = '(silence)'
        print(color(f'  [{turn+1}] B:', Colors.BLUE + Colors.BOLD), color(out_b, Colors.BLUE))
        sys.stdout.flush()

        ctx_a = out_b

        if out_a == out_b:
            print(color('  (Both said the same thing — stopping)', Colors.RED + Colors.DIM))
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

    try:
        ai2ai(model, tokenizer, seed, turns=args.turns, delay_range=(args.min_delay, args.max_delay))
    except KeyboardInterrupt:
        print(color('\nInterrupted. Bye!', Colors.RED))


if __name__ == '__main__':
    main()
