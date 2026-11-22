"""Select OCR alternatives without silently changing recognized literal values."""

import ast
import io
import re
import tokenize
from difflib import SequenceMatcher


def line_parses(text: str, opens_block: bool) -> bool:
    text = text.strip()
    if not text or text.startswith("#"):
        return not opens_block
    word = text.split(maxsplit=1)[0].rstrip(":")
    if opens_block:
        if word in {"elif", "else"}:
            text = "if True:\n    pass\n" + text
        elif word in {"except", "finally"}:
            text = "try:\n    pass\n" + text
        elif word == "case":
            text = "match x:\n    " + text
        if word == "case":
            text += "\n        pass"
        elif word == "match":
            text += "\n    case _:\n        pass"
        else:
            text += "\n    pass"
        if word == "try":
            text += "\nfinally:\n    pass"
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    if opens_block:
        return bool(tree.body and isinstance(
            tree.body[-1], (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith,
                            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Try, ast.TryStar, ast.Match)
        ))
    return not any(
        isinstance(statement, ast.Expr)
        and not isinstance(statement.value, (ast.Call, ast.Await, ast.Yield, ast.YieldFrom, ast.Constant))
        for statement in tree.body
    )


def _literal_values(text: str) -> tuple[str, ...]:
    values: list[str] = []
    previous = None
    before_previous = None
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.string.isspace():
                continue
            if token.type == tokenize.NUMBER:
                expression_end = before_previous and (
                    before_previous.type in (tokenize.NUMBER, tokenize.STRING)
                    or before_previous.string in (")", "]", "}")
                    or (before_previous.type == tokenize.NAME and before_previous.string not in
                        {"return", "yield", "assert", "in", "and", "or", "not", "if", "else"})
                )
                sign = previous.string if previous and previous.string in ("+", "-") and not expression_end else ""
                values.append(sign + token.string)
            elif token.type in (tokenize.STRING, tokenize.COMMENT) or token.string in ("True", "False", "None"):
                values.append(token.string)
            before_previous, previous = previous, token
    except (tokenize.TokenError, SyntaxError):
        pass
    return tuple(values)


def choose_line(original: str, alternatives: list[str], opens_block: bool) -> str:
    original_values = _literal_values(original)
    original_letters = re.sub(r"[^a-zA-Z0-9_]", "", original)
    for candidate in alternatives:
        candidate = candidate.strip()
        candidate_letters = re.sub(r"[^a-zA-Z0-9_]", "", candidate)
        if (
            candidate != original
            and line_parses(candidate, opens_block)
            and _literal_values(candidate) == original_values
            and SequenceMatcher(None, original_letters, candidate_letters, autojunk=False).ratio() >= 0.8
        ):
            return candidate
    return original


def format_token_spacing(text: str) -> str:
    """Remove OCR spacing around delimiters, leaving token contents intact."""
    try:
        tokens = [token for token in tokenize.generate_tokens(io.StringIO(text).readline)
                  if token.type not in {tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER}]
    except (tokenize.TokenError, SyntaxError):
        return text
    edits = []
    for left, right in zip(tokens, tokens[1:]):
        start, end = left.end[1], right.start[1]
        if start >= end or not text[start:end].isspace():
            continue
        if (
            left.string in {"(", "[", "{"}
            or right.string in {")", "]", "}", ",", ":"}
            or (
                right.string == "(" and left.type == tokenize.NAME
                and left.string not in {"if", "elif", "while", "for", "return", "yield", "assert", "with", "in", "not", "and", "or", "except", "match", "case"}
            )
        ):
            edits.append((start, end))
    for start, end in reversed(edits):
        text = text[:start] + text[end:]
    return text
