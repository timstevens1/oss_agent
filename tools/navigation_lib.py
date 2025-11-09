import os
import re
import ast
import inspect
from typing import List, Dict, Any, Tuple

from mcp.server import FastMCP


mcp = FastMCP(name='code_utils',instructions='these utilities will allow you to perform various operations over code bases efficiently.')

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------
def _list_python_files(root: str) -> List[str]:
    """Return a list of .py file paths inside root (recursive)."""
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith('.py'):
                files.append(os.path.join(dirpath, fname))
    return files

def _extract_file_content(filepath: str) -> Dict[int, str]:
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return {i + 1: line.rstrip('\n') for i, line in enumerate(lines)}

def _find_definitions_in_file(filepath: str, symbol: str) -> List[Tuple[int, str]]:
    """Return (lineno,line) tuples that define or assign symbol in file."""
    matches: List[Tuple[int, str]] = []
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())
    for node in _ast_nodes(tree):
        # Function / class definition
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == symbol:
            matches.append((node.lineno, None))
        # Assignment targets
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    matches.append((node.lineno, None))
    return matches

def _ast_nodes(node):
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _ast_nodes(child)

# ------------------------------------------------------------------
# Public API functions
# ------------------------------------------------------------------
@mcp.tool()
def get_definition(symbol_name: str, repo_root: str = '.') -> Tuple[str, Dict[int, str]]:
    """Return (file_path, lines_map) for symbol definition."""
    file_path, lines_map = None, None
    for file_path in _list_python_files(repo_root):
        try:
            matches = _find_definitions_in_file(file_path, symbol_name)
            if matches:
                return file_path, _extract_file_content(file_path)
        except Exception:
            continue
    return None, {}

@mcp.tool()
def get_references(symbol_name: str, repo_root: str = '.') -> Dict[str, Dict[int, str]]:
    """Return nested dict mapping file->lines where symbol referenced."""
    refs: Dict[str, Dict[int, str]] = {}
    for file_path in _list_python_files(repo_root):
        try:
            with open(file_path, 'r') as f:
                data = f.readlines()
            for idx, line in enumerate(data, start=1):
                if re.search(rf'\b{re.escape(symbol_name)}\b', line):
                    refs.setdefault(file_path, {})[idx] = line.strip()
        except Exception:
            continue
    return refs

@mcp.tool()
def get_attributes(obj: Any) -> List[str]:
    """Return list of public attributes of *obj* (no leading underscore)."""
    return [name for name, value in inspect.getmembers(obj) if not name.startswith('_')]

@mcp.tool()
def get_methods(obj: Any) -> List[Dict[str, Any]]:
    """Return list of dicts describing public methods of obj. Each dict contains: name, docstring, params (list of dicts with name, annotation, default), return_annotation, signature."""
    methods: List[Dict[str, Any]] = []
    for name, value in inspect.getmembers(obj):
        if callable(value) and not name.startswith('_'):
            sig = inspect.signature(value)
            params = []
            for param in sig.parameters:
                pinfo = {'name': param.name}
                if param.default is not param.empty:
                    pinfo['default'] = param.default
                if param.annotation is not inspect._empty:
                    pinfo['annotation'] = param.annotation
                params.append(pinfo)
            method_info = {
                'name': name,
                'docstring': inspect.getdoc(value),
                'signature': str(sig),
                'parameters': params,
                'return_annotation': str(sig.return_annotation)
            }
            methods.append(method_info)
    return methods


if __name__ == '__main__':
    mcp.run(transport='stdio')