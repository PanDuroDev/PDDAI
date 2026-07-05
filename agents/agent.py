import json
import os
import re
import sys
import threading
import time
from datetime import datetime
import urllib.parse
import urllib.request
import torch
from tools.registry import ToolRegistry, make_default_registry
import config
from chat import Colors, color, box_top, box_bottom, box_line, box_content, print_box, wrap, Spinner

_CONFIG_FILE = os.path.join(os.path.dirname(__file__), '.agent_config.json')

_COMMANDS = {
    'calculator': ('calculator', 'expression'), 'calc': ('calculator', 'expression'),
    'search': ('web_search', 'query'),           'web_search': ('web_search', 'query'),
    'read': ('read_file', 'path'),               'read_file': ('read_file', 'path'),
}


class Agent:
    def __init__(self, model, tokenizer, device='cpu',
                 tools: ToolRegistry = None, max_tokens=256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.tools = tools or make_default_registry()
        self.max_tokens = max_tokens
        saved = self._load_config()
        self.api_url = saved.get('api_url') or config.NROUTER_API_URL or None
        self.api_key = saved.get('api_key') or getattr(config, 'NROUTER_API_KEY', None) or None
        self.api_model = saved.get('api_model') or getattr(config, 'NROUTER_MODEL', None) or None

    def _load_config(self):
        try:
            with open(_CONFIG_FILE) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_config(self):
        with open(_CONFIG_FILE, 'w') as f:
            json.dump({'api_url': self.api_url, 'api_key': self.api_key, 'api_model': self.api_model}, f)

    def encode(self, text):
        from data.tokenizer import encode_text
        return encode_text(self.tokenizer, text)

    def decode(self, ids):
        from data.tokenizer import decode_text
        return decode_text(self.tokenizer, ids)

    def generate(self, prompt, max_new=96, temperature=0.9, top_k=40, rep_penalty=1.15, on_token=None):
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
                    txt = self.decode([tid]).strip()
                    if txt and (txt[0].isalnum() or txt[0].isalpha()):
                        logits[0, tid] /= rep_penalty
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < vals[:, -1:]] = float('-inf')
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            tokens.append(next_id)
            if on_token:
                on_token(self.decode([next_id]))
            if next_id == self.tokenizer.token_to_id("</s>"):
                break
            x = torch.tensor([[next_id]], dtype=torch.long).to(self.device)
            with torch.inference_mode():
                logits, past = self.model(x, past_key_values=past, use_cache=True)
        return self.decode(tokens), len(tokens)

    def _parse_command(self, text):
        t = text.strip()
        if not t:
            return None
        first = t.split()[0].lower()
        name = first.lstrip('\\/')
        if name in _COMMANDS:
            return _COMMANDS[name] + (t[len(first):].strip(),)
        return None

    def _execute(self, name, param, value):
        if not value or not value.strip():
            value = ''
        return self.tools.call(name, **{param: value})

    def _parse_sse(self, raw):
        content = ""
        for line in raw.strip().split('\n'):
            line = line.strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            try:
                chunk = json.loads(line[6:])
                delta = chunk.get('choices', [{}])[0].get('delta', {})
                if 'content' in delta:
                    content += delta['content']
            except json.JSONDecodeError:
                continue
        return content

    def _compress_context(self, text):
        url = f"{self.api_url}/chat/completions"
        body = {"messages": [{"role": "user", "content": f"Compress the following information into a clear summary. Keep only key facts, remove redundancy.\n\n{text}"}], "max_tokens": 256}
        model = self.api_model or "my-combo"
        body["model"] = model
        payload = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode()
            try:
                result = json.loads(raw)
                return result["choices"][0]["message"]["content"]
            except (json.JSONDecodeError, KeyError):
                content = self._parse_sse(raw)
                if content:
                    return content
            return f"(9Router returned no content)"
        except Exception as e:
            print(f"(9Router error: {e})")
            return text

    def _test_api(self):
        url = f"{self.api_url}/models"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
        return [m['id'] for m in result.get('data', [])]

    def _route_query(self, query):
        q = query.lower().strip()
        starts = ['what is', 'what are', 'who is', 'who are', 'define', 'meaning of',
                   'tell me about', 'explain', 'what does', 'what do', 'describe']
        if any(q.startswith(w) for w in starts):
            return ['wikipedia', 'wikidata']
        tech = ['code', 'programming', 'python', 'javascript', 'java', 'html', 'css',
                'api', 'function', 'algorithm', 'software', 'library', 'framework',
                'how to', 'tutorial', 'install', 'setup', 'command', 'terminal',
                'linux', 'windows', 'git', 'docker', 'sql', 'database', 'debug',
                'error', 'bug', 'syntax', 'compiler', 'runtime', 'deploy']
        if any(k in q for k in tech):
            return ['stackexchange', 'github', 'wikipedia']
        medical = ['disease', 'symptom', 'treatment', 'diagnosis', 'patient', 'clinical',
                   'drug', 'therapy', 'syndrome', 'disorder', 'infection', 'vaccine',
                   'gene', 'protein', 'cell', 'molecular', 'biochemical']
        if any(k in q for k in medical):
            return ['pubmed', 'wikipedia', 'wikidata']
        scientific = ['research', 'study', 'science', 'theory', 'experiment', 'analysis',
                      'mathematics', 'physics', 'chemistry', 'biology', 'algorithm',
                      'neural', 'quantum', 'equation', 'hypothesis', 'paper']
        if any(k in q for k in scientific):
            return ['arxiv', 'pubmed', 'wikipedia']
        news = ['news', 'latest', 'today', 'current', 'breaking', 'election',
                'president', 'war', 'earthquake', 'storm', 'price', 'stock',
                'update', 'announcement', 'release']
        if any(k in q for k in news):
            return ['wikipedia', 'github']
        return ['wikipedia', 'wikidata', 'stackexchange']

    def _search_wikipedia(self, query):
        try:
            search = json.loads(urllib.request.urlopen(
                f'https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(query)}&limit=2&format=json',
                timeout=8).read())
            titles = search[1]
            if not titles:
                return ''
            parts = []
            for title in titles[:2]:
                try:
                    summ = json.loads(urllib.request.urlopen(
                        f'https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title.replace(" ", "_"))}',
                        timeout=8).read())
                    ext = summ.get('extract', '')
                    if ext:
                        parts.append(f'{title}: {ext[:500]}')
                except Exception:
                    continue
            return '\n\n'.join(parts)
        except Exception:
            return ''

    def _search_wikidata(self, query):
        try:
            search = json.loads(urllib.request.urlopen(
                f'https://www.wikidata.org/w/api.php?action=wbsearchentities&search={urllib.parse.quote(query)}&language=en&limit=2&format=json',
                timeout=8).read())
            entities = search.get('search', [])
            if not entities:
                return ''
            parts = []
            for ent in entities[:2]:
                eid = ent.get('id', '')
                label = ent.get('label', '')
                desc = ent.get('description', '')
                if label and desc:
                    parts.append(f'{label}: {desc}')
                elif eid:
                    try:
                        data = json.loads(urllib.request.urlopen(
                            f'https://www.wikidata.org/wiki/Special:EntityData/{eid}.json',
                            timeout=8).read())
                        entity = data.get('entities', {}).get(eid, {})
                        claims = entity.get('claims', {})
                        facts = []
                        for pid, claim_list in list(claims.items())[:6]:
                            for claim in claim_list[:1]:
                                mainsnak = claim.get('mainsnak', {})
                                if mainsnak.get('snaktype') == 'value':
                                    datavalue = mainsnak.get('datavalue', {})
                                    value = datavalue.get('value', {})
                                    if isinstance(value, dict):
                                        v = value.get('value', value.get('id', str(value)))
                                    else:
                                        v = str(value)
                                    facts.append(v)
                        if facts:
                            parts.append(f'{label}: {", ".join(facts)}')
                    except Exception:
                        continue
            return '\n\n'.join(parts)
        except Exception:
            return ''

    def _search_arxiv(self, query):
        try:
            resp = urllib.request.urlopen(
                f'http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&max_results=2&sortBy=relevance',
                timeout=12)
            raw = resp.read().decode('utf-8', errors='replace')
            entries = re.findall(r'<entry>(.*?)</entry>', raw, re.DOTALL)
            if not entries:
                return ''
            parts = []
            for entry in entries[:2]:
                title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                authors = re.findall(r'<name>(.*?)</name>', entry)
                t = title.group(1).strip() if title else ''
                s = summary.group(1).strip()[:400] if summary else ''
                a = ', '.join(authors[:3]) if authors else ''
                if t:
                    text = f'{t}'
                    if a:
                        text += f' by {a}'
                    if s:
                        text += f'\n{s}'
                    parts.append(text)
            return '\n\n'.join(parts)
        except Exception:
            return ''

    def _search_pubmed(self, query):
        try:
            search = json.loads(urllib.request.urlopen(
                f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={urllib.parse.quote(query)}&retmax=3&format=json',
                timeout=10).read())
            ids = search.get('esearchresult', {}).get('idlist', [])
            if not ids:
                return ''
            summary = json.loads(urllib.request.urlopen(
                f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={",".join(ids)}&format=json&retmode=json',
                timeout=10).read())
            parts = []
            for uid in ids:
                rec = summary.get('result', {}).get(uid, {})
                title = rec.get('title', '')
                source = rec.get('source', '')
                pubdate = rec.get('pubdate', '')
                if title:
                    parts.append(f'{title} ({source}, {pubdate})')
            return '\n\n'.join(parts)
        except Exception:
            return ''

    def _search_stackexchange(self, query):
        try:
            resp = urllib.request.urlopen(
                f'https://api.stackexchange.com/2.3/questions?order=desc&sort=relevance&intitle={urllib.parse.quote(query)}&site=stackoverflow&pagesize=3&filter=withbody',
                timeout=10)
            data = json.loads(resp.read())
            items = data.get('items', [])
            if not items:
                return ''
            parts = []
            for item in items[:2]:
                title = item.get('title', '')
                score = item.get('score', 0)
                answer_count = item.get('answer_count', 0)
                tags = ', '.join(item.get('tags', []))
                body = re.sub(r'<[^>]+>', '', item.get('body', ''))[:300]
                if title:
                    text = f'{title} (score: {score}, answers: {answer_count})'
                    if tags:
                        text += f' [{tags}]'
                    if body:
                        text += f'\n{body}'
                    parts.append(text)
            return '\n\n'.join(parts)
        except Exception:
            return ''

    def _search_github(self, query):
        try:
            resp = urllib.request.urlopen(
                f'https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars&per_page=3',
                timeout=10)
            data = json.loads(resp.read())
            items = data.get('items', [])
            if not items:
                return ''
            parts = []
            for item in items[:2]:
                name = item.get('full_name', '')
                desc = item.get('description', '') or ''
                stars = item.get('stargazers_count', 0)
                lang = item.get('language', '') or ''
                url = item.get('html_url', '')
                if name:
                    text = f'{name}'
                    if lang:
                        text += f' ({lang})'
                    text += f' - {stars} stars'
                    if desc:
                        text += f'\n{desc}'
                    parts.append(text)
            return '\n\n'.join(parts)
        except Exception:
            return ''

    def _aggregate_results(self, query, sources):
        results = {}
        for src in sources:
            method = getattr(self, f'_search_{src}', None)
            if method:
                try:
                    content = method(query)
                    if content:
                        results[src] = content
                except Exception:
                    continue
        if not results:
            return ''
        parts = []
        for src, content in results.items():
            parts.append(f'[{src}]\n{content}')
        merged = '\n\n---\n\n'.join(parts)
        lines = merged.split('\n')
        seen = set()
        unique = []
        for line in lines:
            key = re.sub(r'\s+', ' ', line.strip().lower())
            if key in seen or len(key) < 15:
                continue
            seen.add(key)
            unique.append(line)
        return '\n'.join(unique)[:3000]

    def _output_cleaner(self, text, user_query=''):
        cleaned = text
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\uFFFD]', '', cleaned)
        cleaned = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', cleaned)
        cleaned = re.sub(r'([^.!?]{30,}[.!?])\s*\1\s*', r'\1 ', cleaned)
        cleaned = re.sub(r'\s+([.,!?;:])', r'\1', cleaned)
        cleaned = re.sub(r'[ \t]{3,}', ' ', cleaned)
        cleaned = re.sub(r'\n{4,}', '\n\n', cleaned)
        cleaned = cleaned.strip()
        if text.strip() == cleaned:
            return text
        return cleaned

    def run(self, user_input):
        t = user_input.strip().lower()
        if t == 'api' or t.startswith('api '):
            parts = user_input.strip().split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                return f"URL: {self.api_url or 'not set'}  |  key: {'set' if self.api_key else 'not set'}  |  model: {self.api_model or 'not set'}"
            self.api_url = parts[1].strip().rstrip('/')
            self._save_config()
            try:
                models = self._test_api()
                return f"API URL set to: {self.api_url}\nAvailable models: {', '.join(models)}"
            except Exception as e:
                return f"API URL set to: {self.api_url}\n(Connection test failed: {e})"
        if t == 'key' or t.startswith('key '):
            parts = user_input.strip().split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                return f"API key: {'set' if self.api_key else 'not set'}"
            self.api_key = parts[1].strip()
            self._save_config()
            if not self.api_url:
                return "API key set. Now set the API URL with: api <url>"
            try:
                models = self._test_api()
                return f"API key set. Connection OK.\nAvailable models: {', '.join(models)}"
            except Exception as e:
                return f"API key set. (Test failed: {e})"
        if t == 'model' or t.startswith('model '):
            parts = user_input.strip().split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                return f"API model: {self.api_model or 'not set (auto)'}"
            self.api_model = parts[1].strip()
            self._save_config()
            return f"API model set to: {self.api_model}"

        cmd = self._parse_command(user_input)
        if cmd:
            tool, param, value = cmd
            if tool == 'web_search':
                print()
                spinner = Spinner('Searching sources')
                spinner.start()
                sources = self._route_query(value)
                raw = self._aggregate_results(value, sources)
                if not raw:
                    result = self._execute(tool, param, value)
                    context = result[:2000]
                else:
                    context = raw
                spinner.stop()
                if self.api_url and raw:
                    spinner = Spinner('Compressing with 9Router')
                    spinner.start()
                    compressed = self._compress_context(context)
                    if not compressed.startswith('(9Router'):
                        context = compressed
                    spinner.stop()
                sys.stdout.write('AI: ')
                sys.stdout.flush()
                tokens = []
                def on_token(t):
                    tokens.append(t)
                    sys.stdout.write(t)
                    sys.stdout.flush()
                prompt = f"{context}\n\nBased on this information, {value}"
                response, _ = self.generate(prompt, max_new=192, on_token=on_token)
                print()
                return self._output_cleaner(response, value)
            result = self._execute(tool, param, value)
            sys.stdout.write(f'AI: {result}')
            sys.stdout.flush()
            return result

        sys.stdout.write('AI: ')
        sys.stdout.flush()
        tokens = []
        def on_token(t):
            tokens.append(t)
            sys.stdout.write(t)
            sys.stdout.flush()
        response, _ = self.generate(user_input, max_new=128, on_token=on_token)
        print()
        return self._output_cleaner(response, user_input)

    def _format_training(self, conversation):
        if not self.api_url:
            return None
        lines = []
        for user_msg, ai_msg in conversation:
            lines.append(f'User: {user_msg}')
            lines.append(f'Assistant: {ai_msg}')
        text = '\n'.join(lines)
        url = f'{self.api_url}/chat/completions'
        body = {"messages": [{"role": "user", "content": f"Format this conversation as clean training data. Correct factual errors while keeping the original meaning. Output only the formatted conversation with User:/Assistant: markers.\n\n{text}"}], "max_tokens": 512}
        if self.api_model:
            body['model'] = self.api_model
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
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

    def _save_conversation(self, conversation):
        if not conversation:
            return
        out_dir = 'training_conversations'
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(out_dir, f'conversation_{ts}.txt')
        spinner = Spinner('Formatting conversation for training')
        spinner.start()
        data = self._format_training(conversation)
        spinner.stop()
        if data:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(data)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                for u, a in conversation:
                    f.write(f'User: {u}\nAssistant: {a}\n\n')
        print(color('muted', f'  Saved: {path}'))

    def _show_tool_menu(self):
        tools = self.tools.list()
        print()
        print_box('Tools', '\n'.join(f'\\{t.name:<12} {t.description}' for t in tools), 'primary', 'text')
        print(color('muted', '  \\<tool> <args>  to use a tool'))
        print()

    def _show_help(self):
        print()
        print_box('Commands',
            '/exit           Exit\n'
            '/help           Show this\n'
            '/api <url>      Set API URL\n'
            '/key <k>        Set API key\n'
            '/model <m>      Set model\n'
            '/tools          List tools\n'
            '\\<tool> <args>   Use a tool',
            'primary', 'text')
        print()

    def chat(self):
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print()
        lbl = f'{total_params:.2f}M params'
        w = max(len(lbl) + 2, 14)
        print(color('primary', 'bold', f'  ╭─ PDDAI Agent {"─" * (w - 8)}╮'))
        print(color('primary', f'  │ {lbl} │'))
        print(color('primary', 'bold', f'  ╰{"─" * w}╯'))
        print(color('muted', f'  tools: calculator, search, read'))
        print()

        conversation = []

        while True:
            try:
                inp = input(color('primary', 'bold', '> '))
                if not inp.strip():
                    continue

                if inp.startswith('/'):
                    cmd = inp[1:].strip().lower()
                    if cmd in ('exit', 'quit', 'q'):
                        print(color('muted', '  closing session...'))
                        break
                    elif cmd == 'help':
                        self._show_help()
                        continue
                    elif cmd == 'tools':
                        self._show_tool_menu()
                        continue
                    elif cmd.startswith('api '):
                        print_box('You', inp, 'accent')
                        resp = self.run(inp)
                        print_box('Agent', resp, 'primary', 'text')
                        print()
                        conversation.append((inp, resp))
                        continue
                    elif cmd.startswith('key '):
                        print_box('You', inp, 'accent')
                        resp = self.run(inp)
                        print_box('Agent', resp, 'primary', 'text')
                        print()
                        conversation.append((inp, resp))
                        continue
                    elif cmd.startswith('model '):
                        print_box('You', inp, 'accent')
                        resp = self.run(inp)
                        print_box('Agent', resp, 'primary', 'text')
                        print()
                        conversation.append((inp, resp))
                        continue
                    else:
                        print(color('error', f'  unknown: /{cmd}'))
                        print(color('muted', '  try /help'))
                        continue

                if inp.startswith('\\'):
                    parts = inp[1:].strip().split(maxsplit=1)
                    tool_name = parts[0].lower() if parts else ''
                    tool_arg = parts[1] if len(parts) > 1 else ''
                    if not tool_name:
                        self._show_tool_menu()
                        continue
                    if tool_name not in ('calculator', 'calc', 'search', 'web_search', 'read', 'read_file'):
                        print(color('error', f'  unknown tool: \\{tool_name}'))
                        print(color('muted', '  try \\help or \\tools'))
                        continue
                    print_box('You', inp, 'accent')
                    if tool_name in ('search', 'web_search'):
                        sp = Spinner('Searching sources')
                        sp.start()
                        sources = self._route_query(tool_arg)
                        raw = self._aggregate_results(tool_arg, sources)
                        if raw and self.api_url:
                            sp.stop()
                            sp = Spinner('Compressing with 9Router')
                            sp.start()
                            compressed = self._compress_context(raw)
                            if not compressed.startswith('(9Router'):
                                raw = compressed
                            sp.stop()
                        elif raw:
                            sp.stop()
                        else:
                            sp.stop()
                            result = self.tools.call('web_search', query=tool_arg)
                            raw = result[:2000]
                    if tool_name in ('search', 'web_search'):
                        prompt = f"{raw}\n\nAnswer the question based on this: {tool_arg}"
                        sp = Spinner('Generating')
                        sp.start()
                        response, _ = self.generate(prompt, max_new=192)
                        sp.stop()
                        disp = response.strip()
                    else:
                        sp = Spinner('Running')
                        sp.start()
                        if tool_name in ('calc', 'calculator'):
                            result = self.tools.call('calculator', expression=tool_arg)
                        elif tool_name in ('read', 'read_file'):
                            result = self.tools.call('read_file', path=tool_arg)
                        else:
                            result = "unknown tool"
                        sp.stop()
                        disp = result.strip()
                    cleaned = self._output_cleaner(disp)
                    print_box('Agent', cleaned, 'primary', 'text')
                    print()
                    conversation.append((inp, cleaned))
                    continue

                print_box('You', inp, 'accent')
                sp = Spinner('Generating')
                sp.start()
                response, _ = self.generate(inp, max_new=128)
                sp.stop()
                disp = response.strip()
                cleaned = self._output_cleaner(disp)
                print_box('Agent', cleaned, 'primary', 'text')
                print()
                conversation.append((inp, cleaned))

            except KeyboardInterrupt:
                print(color('muted', '\n  closing session...'))
                break
            except Exception as e:
                print(color('error', f'  Error: {e}'))

        self._save_conversation(conversation)


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
