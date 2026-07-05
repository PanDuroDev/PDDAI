import sys
import os
import argparse
from chat import Colors, color, box_top, box_bottom, box_line, print_box


MODES = [
    ('chat', 'Plain conversation with the model'),
    ('agent', 'Agent with tools (search, calc, read)'),
    ('ai2ai', 'Two models talking to each other'),
]


def show_menu():
    print()
    print_box('PDDAI', 'Select a mode to start', 'primary', 'text')
    print()
    for i, (name, desc) in enumerate(MODES, 1):
        print(color('text', f'  {i}. {color("accent", "bold", name):<8} {desc}'))
    print()
    print(color('muted', '  /exit  to quit'))
    print()


def main():
    parser = argparse.ArgumentParser(description='PDDAI — Unified launcher')
    parser.add_argument('--model', default='last', help='model to load')
    parser.add_argument('mode', nargs='?', default=None,
                        help='mode: chat, agent, ai2ai')
    args = parser.parse_args()

    model_flag = ['--model', args.model]

    if args.mode:
        mode = args.mode.lower()
        if mode not in ('chat', 'agent', 'ai2ai'):
            print(color('error', f'  unknown mode: {mode}'))
            print(color('muted', '  use: chat, agent, ai2ai'))
            return
        _launch(mode, model_flag)
        return

    show_menu()

    while True:
        try:
            inp = input(color('primary', 'bold', '> ')).strip()
            if not inp:
                continue
            if inp in ('exit', 'quit', 'q', '/exit', '/quit', '/q'):
                print(color('muted', '  bye'))
                break
            if inp.isdigit():
                n = int(inp)
                if 1 <= n <= len(MODES):
                    mode = MODES[n - 1][0]
                    _launch(mode, model_flag)
                    show_menu()
                    continue
            print(color('error', f'  choose 1-{len(MODES)}'))
        except KeyboardInterrupt:
            print(color('muted', '\n  bye'))
            break


def _launch(mode, model_flag):
    print()
    if mode == 'chat':
        import chat as ch
        ch.main(model_flag[1])
    elif mode == 'agent':
        from agents.agent import Agent
        from chat import load_model, resolve_model_path, Spinner
        path = resolve_model_path(model_flag[1])
        sp = Spinner('Loading model')
        sp.start()
        m, tok = load_model(path)
        sp.stop()
        if m and tok:
            Agent(m, tok).chat()
    elif mode == 'ai2ai':
        import ai2ai as a2
        a2.main_cli(model_flag[1])


if __name__ == '__main__':
    main()
