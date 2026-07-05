import json
import re
import os
import urllib.request
import urllib.parse

_RESULT_MAX_CHARS = 1500


def _truncate(text, limit=_RESULT_MAX_CHARS):
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _safe_path(base, path):
    base_abs = os.path.realpath(base)
    full = os.path.realpath(os.path.join(base, path))
    if not full.startswith(base_abs):
        return None
    return full


class Tool:
    __slots__ = ('name', 'description', 'func', 'parameters')

    def __init__(self, name, description, func, parameters=None):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {}

    def to_mcp_format(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": [k for k, v in self.parameters.items() if v.get("required")],
                },
            },
        }

    def call(self, **kwargs):
        return str(self.func(**kwargs))


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, tool):
        self._tools[tool.name] = tool

    def get(self, name):
        return self._tools.get(name)

    def list(self):
        return list(self._tools.values())

    def to_mcp_list(self):
        return [t.to_mcp_format() for t in self._tools.values()]

    def call(self, name, **kwargs):
        tool = self.get(name)
        if tool is None:
            return f"Tool '{name}' not found"
        try:
            return tool.call(**kwargs)
        except Exception as e:
            return f"{tool.name} error: {e}"

    def build_prompt_section(self):
        lines = ["\nAvailable tools:"]
        for t in self._tools.values():
            lines.append(f"\n- {t.name}: {t.description}")
        return "\n".join(lines)


def make_web_search_tool():
    _HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    _SEARCH_ENGINES = [
        'https://lite.duckduckgo.com/lite/?q=',
        'https://html.duckduckgo.com/html/?q=',
    ]
    _SCRAPE_TIMEOUT = 6

    def _has_arabic(text):
        return bool(re.search(r'[\u0600-\u06FF]', text))

    def _fetch(url):
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=_SCRAPE_TIMEOUT) as resp:
            return resp.read().decode('utf-8', errors='replace')

    def _extract_text(html):
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
        html = re.sub(r'<[^>]+>', ' ', html)
        html = re.sub(r'&[a-z]+;', ' ', html)
        html = re.sub(r'\s+', ' ', html).strip()
        return html

    def _scrape_page(url):
        try:
            html = _fetch(url)
            text = _extract_text(html)
            lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 40]
            return '\n'.join(lines[:_MAX_SUMMARY_LINES])
        except Exception:
            return None

    def _try_wikipedia(q, lang='en'):
        domain = f'{lang}.wikipedia.org'
        try:
            html = _fetch(f'https://{domain}/wiki/{urllib.parse.quote(q.replace(" ", "_"))}')
            text = _extract_text(html)
            paragraphs = [l.strip() for l in text.split('\n') if len(l.strip()) > 60]
            result = '\n'.join(paragraphs[:_MAX_SUMMARY_LINES])
            if len(result) > 200:
                return result
        except Exception:
            pass
        try:
            url = f'https://{domain}/w/api.php?action=opensearch&search={urllib.parse.quote(q)}&limit=5&format=json'
            data = json.loads(_fetch(url))
            titles = data[1] if len(data) > 1 else []
            if not titles:
                return None
            parts = []
            for title in titles[:3]:
                try:
                    su = f'https://{domain}/api/rest_v1/page/summary/{urllib.parse.quote(title)}'
                    s = json.loads(_fetch(su))
                    ext = s.get('extract', '')
                    if ext:
                        parts.append(f'{title}: {ext}')
                except Exception:
                    pass
            return '\n\n'.join(parts) if parts else None
        except Exception:
            return None

    def _try_ddg_instant(q):
        try:
            url = f'https://api.duckduckgo.com/?q={urllib.parse.quote(q)}&format=json&no_html=1'
            d = json.loads(_fetch(url))
            abstract = d.get('AbstractText', '')
            if abstract:
                return abstract
            results = []
            for topic in d.get('RelatedTopics', []):
                if isinstance(topic, dict):
                    txt = topic.get('Text', '')
                    if txt:
                        results.append(txt)
                elif isinstance(topic, list):
                    for sub in topic:
                        if isinstance(sub, dict):
                            txt = sub.get('Text', '')
                            if txt:
                                results.append(txt)
            return '\n'.join(results[:_MAX_RESULTS]) if results else None
        except Exception:
            return None

    def _try_html_search(q):
        for base in _SEARCH_ENGINES:
            try:
                html = _fetch(base + urllib.parse.quote(q))
                text = _extract_text(html)
                snippets = re.findall(r'(?:^|\n)([A-Z][^.]{50,}\.)', text)
                results = [s.strip() for s in snippets if len(s.strip()) > 60]
                if results:
                    return '\n'.join(results[:_MAX_RESULTS])
            except Exception:
                continue
        return None

    def _extract_keywords(q):
        q2 = re.sub(r'\b(what|is|are|the|best|how|to|do|does|can|i|a|an|in|of|for|with|from|about)\b', '', q, flags=re.IGNORECASE)
        q2 = re.sub(r'[?]', '', q2).strip()
        return q2 if q2 else q

    _MAX_RESULTS = 5
    _MAX_SUMMARY_LINES = 30

    def search(query):
        if not query or not query.strip():
            return 'Please provide a search query.'
        q = query.strip()
        kw = _extract_keywords(q)
        attempts = list(dict.fromkeys([q, kw])) if kw != q else [q]

        sources = []

        langs = ['ar', 'en'] if _has_arabic(q) else ['en', 'ar']

        for lang in langs:
            for attempt in attempts:
                try:
                    r = _try_wikipedia(attempt, lang)
                    if r:
                        sources.append(('Wikipedia', r))
                        break
                except Exception:
                    continue

        for attempt in attempts:
            try:
                r = _try_ddg_instant(attempt)
                if r:
                    sources.append(('DuckDuckGo', r))
                    break
            except Exception:
                continue

        for attempt in attempts:
            try:
                r = _try_html_search(attempt)
                if r:
                    sources.append(('Web', r))
                    break
            except Exception:
                continue

        if not sources:
            return 'No information found.'

        parts = []
        for src, content in sources:
            truncated = _truncate(content, _RESULT_MAX_CHARS)
            parts.append(f'[Source: {src}]\n{truncated}')
        return '\n\n---\n\n'.join(parts)

    return Tool(
        name="web_search",
        description="Search the web for information about any topic. Supports complex multi-word queries in any language.",
        func=search,
        parameters={
            "query": {"type": "string", "description": "search query", "required": True},
        },
    )


def make_calculator_tool():
    _SAFE_CHARS = set("0123456789+-*/().,% ")

    def calculate(expression):
        if not expression or not expression.strip():
            return "No expression provided."
        if len(expression) > 500:
            return "Expression too long."
        safe = expression.strip().replace("^", "**").replace(",", ".")
        safe = re.sub(r'(\d+)%', r'\1/100', safe)
        allowed_check = safe.replace("**", "").replace("//", "")
        if not all(c in _SAFE_CHARS for c in allowed_check):
            return "Invalid expression."
        try:
            result = eval(safe, {"__builtins__": {}}, {})
            if isinstance(result, (int, float)):
                if abs(result - int(result)) < 1e-12:
                    return str(int(result))
                return f"{result:.10g}"
            return str(result)
        except ZeroDivisionError:
            return "Cannot divide by zero."
        except SyntaxError:
            return "Invalid syntax."
        except Exception:
            return "Error in calculation."

    return Tool(
        name="calculator",
        description="Evaluate math expressions. Supports + - * / ( ) and decimals.",
        func=calculate,
        parameters={
            "expression": {
                "type": "string",
                "description": "math expression",
                "required": True,
            },
        },
    )


def make_read_file_tool(base_dir="."):
    def read_file(path):
        if not path or not path.strip():
            return "No path provided."
        full = _safe_path(base_dir, path.strip())
        if full is None:
            return "Path outside allowed directory."
        if not os.path.exists(full):
            return f"File not found: {path.strip()}"
        if not os.path.isfile(full):
            return f"Not a file: {path.strip()}"
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(_RESULT_MAX_CHARS + 500)
            return _truncate(content)
        except PermissionError:
            return f"Cannot read: {path.strip()}"
        except Exception as e:
            return f"Error: {e}"

    return Tool(
        name="read_file",
        description="Read a text file from the local filesystem.",
        func=read_file,
        parameters={
            "path": {
                "type": "string",
                "description": "file path",
                "required": True,
            },
        },
    )


def make_default_registry():
    reg = ToolRegistry()
    reg.register(make_web_search_tool())
    reg.register(make_calculator_tool())
    reg.register(make_read_file_tool())
    return reg
