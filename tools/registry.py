import json
import re
import os
import urllib.request
import urllib.parse

_RESULT_MAX_CHARS = 1500
_ERROR_PREFIX = "Error"


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
        result = self.func(**kwargs)
        if isinstance(result, str) and result.startswith(_ERROR_PREFIX):
            return result
        return str(result)


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
            return f"{_ERROR_PREFIX}: tool '{name}' not found"
        try:
            return tool.call(**kwargs)
        except Exception as e:
            return f"{_ERROR_PREFIX}: {tool.name} — {e}"

    def build_prompt_section(self):
        lines = ["\nAvailable tools:"]
        for t in self._tools.values():
            lines.append(f"\n- {t.name}: {t.description}")
        return "\n".join(lines)


def make_web_search_tool():
    def search(query, num_results=3):
        if not query or not query.strip():
            return f"{_ERROR_PREFIX}: query is empty"
        try:
            num = max(1, min(10, int(num_results)))
        except (ValueError, TypeError):
            num = 3
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query.strip())}&format=json&no_html=1"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read())
            parts = []
            abstract = data.get("AbstractText", "")
            if abstract:
                parts.append(f"Summary: {abstract}")
            topics = data.get("RelatedTopics", [])
            for topic in topics[:num]:
                if isinstance(topic, dict):
                    txt = topic.get("Text", "")
                    if txt:
                        parts.append(f"- {txt}")
            if not parts:
                return "No results found"
            return _truncate("\n".join(parts))
        except urllib.error.HTTPError as e:
            return f"{_ERROR_PREFIX}: web_search returned status {e.code}"
        except urllib.error.URLError:
            return f"{_ERROR_PREFIX}: web_search — network error"
        except json.JSONDecodeError:
            return f"{_ERROR_PREFIX}: web_search — invalid response"
        except Exception as e:
            return f"{_ERROR_PREFIX}: web_search — {e}"

    return Tool(
        name="web_search",
        description="Search web for current information. Returns text summaries.",
        func=search,
        parameters={
            "query": {"type": "string", "description": "search query", "required": True},
            "num_results": {"type": "integer", "description": "results count (1-10)"},
        },
    )


def make_calculator_tool():
    _SAFE_CHARS = set("0123456789+-*/().,% ")

    def calculate(expression):
        if not expression or not expression.strip():
            return f"{_ERROR_PREFIX}: expression is empty"
        if len(expression) > 500:
            return f"{_ERROR_PREFIX}: expression too long"
        safe = expression.strip().replace("^", "**").replace(",", ".")
        safe = re.sub(r'(\d+)%', r'\1/100', safe)
        allowed_check = safe.replace("**", "").replace("//", "")
        if not all(c in _SAFE_CHARS for c in allowed_check):
            return f"{_ERROR_PREFIX}: invalid characters in expression"
        try:
            result = eval(safe, {"__builtins__": {}}, {})
            if isinstance(result, (int, float)):
                if abs(result - int(result)) < 1e-12:
                    return str(int(result))
                return f"{result:.10g}"
            return str(result)
        except ZeroDivisionError:
            return f"{_ERROR_PREFIX}: division by zero"
        except SyntaxError:
            return f"{_ERROR_PREFIX}: invalid syntax"
        except Exception as e:
            return f"{_ERROR_PREFIX}: {e}"

    return Tool(
        name="calculator",
        description="Evaluate math expressions. Supports + - * / ** ( ) % and decimals.",
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
            return f"{_ERROR_PREFIX}: path is empty"
        full = _safe_path(base_dir, path.strip())
        if full is None:
            return f"{_ERROR_PREFIX}: path outside allowed directory"
        if not os.path.exists(full):
            return f"{_ERROR_PREFIX}: file not found — {path.strip()}"
        if not os.path.isfile(full):
            return f"{_ERROR_PREFIX}: not a file — {path.strip()}"
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(_RESULT_MAX_CHARS + 500)
            return _truncate(content)
        except PermissionError:
            return f"{_ERROR_PREFIX}: permission denied — {path.strip()}"
        except Exception as e:
            return f"{_ERROR_PREFIX}: read_file — {e}"

    return Tool(
        name="read_file",
        description="Read a local text file. Returns content or error message.",
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
