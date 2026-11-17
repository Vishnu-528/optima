"""Lexer: source text -> list of tokens.

Token kinds:
    NUMBER   3, 2.5, 1e-6      (tok.is_int is True for plain digit strings)
    IDENT    names
    KEYWORD  print minimize maximize from using gradient at iterations
    STRING   "text"            (only allowed in print)
    OP       + - * / ^ = ( ) [ ] ,
    NEWLINE  end of a statement
    EOF
"""

from dataclasses import dataclass

from .errors import OptimaError

KEYWORDS = {
    "print", "minimize", "maximize", "from", "using",
    "gradient", "at", "iterations",
}

OPERATORS = set("+-*/^=()[],")


@dataclass
class Token:
    kind: str
    text: str
    line: int
    col: int
    value: object = None   # float for NUMBER, str for STRING
    is_int: bool = False   # NUMBER written without '.' or exponent

    def __repr__(self):
        return f"{self.kind}({self.text!r})@{self.line}:{self.col}"


def tokenize(source):
    tokens = []
    i, line, col = 0, 1, 1
    n = len(source)
    depth = 0  # bracket depth: newlines inside ( ) or [ ] do not end a statement

    def error(msg, l=None, c=None):
        raise OptimaError(msg, l or line, c or col)

    while i < n:
        ch = source[i]

        # whitespace
        if ch in " \t\r":
            i += 1
            col += 1
            continue

        # comment: '#' to end of line
        if ch == "#":
            while i < n and source[i] != "\n":
                i += 1
            continue

        # newline
        if ch == "\n":
            if depth == 0 and tokens and tokens[-1].kind != "NEWLINE":
                tokens.append(Token("NEWLINE", "\\n", line, col))
            i += 1
            line += 1
            col = 1
            continue

        # number
        if ch.isdigit() or (ch == "." and i + 1 < n and source[i + 1].isdigit()):
            start, start_col = i, col
            is_int = True
            while i < n and source[i].isdigit():
                i += 1
            if i < n and source[i] == ".":
                is_int = False
                i += 1
                while i < n and source[i].isdigit():
                    i += 1
            if i < n and source[i] in "eE":
                j = i + 1
                if j < n and source[j] in "+-":
                    j += 1
                if j < n and source[j].isdigit():
                    is_int = False
                    i = j
                    while i < n and source[i].isdigit():
                        i += 1
            text = source[start:i]
            if i < n and (source[i].isalpha() or source[i] == "_"):
                error(f"invalid number '{text}{source[i]}...'", line, start_col)
            col += i - start
            tokens.append(Token("NUMBER", text, line, start_col,
                                value=float(text), is_int=is_int))
            continue

        # identifier / keyword
        if ch.isalpha() or ch == "_":
            start, start_col = i, col
            while i < n and (source[i].isalnum() or source[i] == "_"):
                i += 1
            text = source[start:i]
            col += i - start
            kind = "KEYWORD" if text in KEYWORDS else "IDENT"
            tokens.append(Token(kind, text, line, start_col))
            continue

        # string literal
        if ch == '"':
            start_col = col
            j = i + 1
            while j < n and source[j] not in '"\n':
                j += 1
            if j >= n or source[j] != '"':
                error("unterminated string (missing closing \")", line, start_col)
            text = source[i + 1:j]
            for k, c in enumerate(text):
                if ord(c) < 32 or ord(c) > 126 or c == "\\":
                    error("strings may only contain plain printable characters "
                          "(no backslashes)", line, start_col + 1 + k)
            tokens.append(Token("STRING", source[i:j + 1], line, start_col, value=text))
            col += j + 1 - i
            i = j + 1
            continue

        # operators and punctuation
        if ch in OPERATORS:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth = max(0, depth - 1)
            tokens.append(Token("OP", ch, line, col))
            i += 1
            col += 1
            continue

        error(f"unexpected character '{ch}'")

    if tokens and tokens[-1].kind != "NEWLINE":
        tokens.append(Token("NEWLINE", "\\n", line, col))
    tokens.append(Token("EOF", "", line, col))
    return tokens
