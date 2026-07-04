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
    def search(query):
        if not query or not query.strip():
            return "Please provide a search query."

        q = query.strip()

        # Primary: Wikipedia API (clean, reliable text)
        try:
            search_url = (
                "https://en.wikipedia.org/w/api.php?"
                f"action=opensearch&search={urllib.parse.quote(q)}&limit=3&format=json"
            )
            with urllib.request.urlopen(search_url, timeout=5) as resp:
                data = json.loads(resp.read())

            titles = data[1] if len(data) > 1 else []
            descs = data[2] if len(data) > 2 else []

            if titles:
                try:
                    summary_url = (
                        f"https://en.wikipedia.org/api/rest_v1/page/summary/"
                        f"{urllib.parse.quote(titles[0])}"
                    )
                    with urllib.request.urlopen(summary_url, timeout=5) as resp:
                        summary_data = json.loads(resp.read())
                    extract = summary_data.get("extract", "")
                    if extract:
                        parts = [extract]
                        for i in range(1, min(len(titles), 3)):
                            d = descs[i] if i < len(descs) else ""
                            if d:
                                parts.append(f"\n{titles[i]}: {d}")
                        return _truncate("\n".join(parts))
                except Exception:
                    pass

                # Fallback within Wikipedia: show titles + descriptions
                parts = []
                for i in range(min(len(titles), 3)):
                    t = titles[i]
                    d = descs[i] if i < len(descs) else ""
                    parts.append(f"{t}: {d}" if d else t)
                if parts:
                    return _truncate("\n".join(parts))
        except Exception:
            pass

        # Fallback: DuckDuckGo instant answer
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(q)}&format=json&no_html=1"
            with urllib.request.urlopen(url, timeout=5) as resp:
                d = json.loads(resp.read())

            abstract = d.get("AbstractText", "")
            if abstract:
                return _truncate(abstract)

            for topic in d.get("RelatedTopics", []):
                if isinstance(topic, dict):
                    txt = topic.get("Text", "")
                    if txt:
                        return _truncate(txt)
        except Exception:
            pass

        return "No information found."

    return Tool(
        name="web_search",
        description="Search the web for information about any topic.",
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
