"""Recursive-descent parser: tokens -> syntax tree.

Grammar (see docs/SYNTAX.md):

    program      = { [ statement ] NEWLINE } ;
    statement    = assignment | function_def | print_stmt ;
    assignment   = IDENT "=" expr ;
    function_def = IDENT "(" IDENT { "," IDENT } ")" "=" expr ;
    print_stmt   = "print" item { "," item } ;
    item         = STRING | expr ;

    expr     = term  { ("+" | "-") term } ;
    term     = unary { ("*" | "/") unary } ;
    unary    = "-" unary | power ;
    power    = postfix [ "^" [ "-" ] INT ] ;
    postfix  = primary { "[" INT "]" } ;
    primary  = NUMBER | IDENT | IDENT "(" expr { "," expr } ")"
             | "(" expr { "," expr } ")"
             | "iterations"
             | "gradient" IDENT "at" unary
             | ("minimize" | "maximize") IDENT "from" unary [ "using" solver ] ;
    solver   = IDENT [ "(" [ option { "," option } ] ")" ] ;
    option   = IDENT "=" expr ;
"""

from . import syntax as S
from .errors import OptimaError
from .lexer import tokenize


class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.pos = 0

    # ------------------------------------------------------------ helpers
    @property
    def tok(self):
        return self.toks[self.pos]

    def peek(self, offset=1):
        return self.toks[min(self.pos + offset, len(self.toks) - 1)]

    def advance(self):
        t = self.tok
        if t.kind != "EOF":
            self.pos += 1
        return t

    def is_op(self, text):
        return self.tok.kind == "OP" and self.tok.text == text

    def is_kw(self, text):
        return self.tok.kind == "KEYWORD" and self.tok.text == text

    def describe(self, t=None):
        t = t or self.tok
        if t.kind == "NEWLINE":
            return "end of line"
        if t.kind == "EOF":
            return "end of file"
        return f"'{t.text}'"

    def error(self, msg, t=None):
        t = t or self.tok
        raise OptimaError(msg, t.line, t.col)

    def expect_op(self, text, context=""):
        if not self.is_op(text):
            where = f" {context}" if context else ""
            self.error(f"expected '{text}'{where}, found {self.describe()}")
        return self.advance()

    def expect_kw(self, text, context=""):
        if not self.is_kw(text):
            where = f" {context}" if context else ""
            self.error(f"expected '{text}'{where}, found {self.describe()}")
        return self.advance()

    def expect_ident(self, what):
        if self.tok.kind != "IDENT":
            extra = ""
            if self.tok.kind == "KEYWORD":
                extra = f" ('{self.tok.text}' is a reserved word)"
            self.error(f"expected {what}, found {self.describe()}{extra}")
        return self.advance()

    def expect_int(self, what):
        if self.tok.kind != "NUMBER" or not self.tok.is_int:
            self.error(f"expected {what} (a whole number), found {self.describe()}")
        return self.advance()

    # --------------------------------------------------------- statements
    def parse_program(self):
        stmts = []
        while self.tok.kind != "EOF":
            if self.tok.kind == "NEWLINE":
                self.advance()
                continue
            stmts.append(self.parse_statement())
            if self.tok.kind not in ("NEWLINE", "EOF"):
                self.error(f"unexpected {self.describe()} after the end of the statement")
        return stmts

    def parse_statement(self):
        t = self.tok
        if self.is_kw("print"):
            return self.parse_print()
        if t.kind == "IDENT":
            nxt = self.peek()
            if nxt.kind == "OP" and nxt.text == "=":
                self.advance()
                self.advance()
                value = self.parse_expr()
                return S.Assign(t.line, t.col, t.text, value)
            if nxt.kind == "OP" and nxt.text == "(":
                return self.parse_funcdef()
            self.error("a statement must be an assignment (name = ...), a function "
                       "definition (f(x, y) = ...) or a print", nxt)
        if t.kind == "KEYWORD":
            self.error(f"'{t.text}' cannot start a statement; assign its result to a "
                       f"name or print it")
        self.error(f"unexpected {self.describe()} at the start of a statement")

    def parse_funcdef(self):
        name = self.advance()
        self.expect_op("(")
        params = []
        if self.is_op(")"):
            self.error("a function needs at least one parameter")
        while True:
            if self.tok.kind != "IDENT":
                self.error(
                    f"expected a parameter name in the definition of '{name.text}', found "
                    f"{self.describe()} (a function call cannot be a statement on its own; "
                    f"assign it to a name or print it)")
            p = self.advance()
            if any(p.text == q[0] for q in params):
                self.error(f"duplicate parameter '{p.text}'", p)
            params.append((p.text, p.line, p.col))
            if self.is_op(","):
                self.advance()
                continue
            break
        self.expect_op(")", "after the parameter list")
        if not self.is_op("="):
            self.error(
                f"expected '=' to define function '{name.text}', found {self.describe()} "
                f"(a function call cannot be a statement on its own; assign it to a name "
                f"or print it)")
        self.advance()
        body = self.parse_expr()
        return S.FuncDef(name.line, name.col, name.text, params, body)

    def parse_print(self):
        kw = self.advance()
        if self.tok.kind in ("NEWLINE", "EOF"):
            self.error("print needs something to print")
        items = []
        while True:
            if self.tok.kind == "STRING":
                s = self.advance()
                items.append(S.Str(s.line, s.col, s.value))
            else:
                items.append(self.parse_expr())
            if self.is_op(","):
                self.advance()
                continue
            break
        return S.Print(kw.line, kw.col, items)

    # -------------------------------------------------------- expressions
    def parse_expr(self):
        left = self.parse_term()
        while self.is_op("+") or self.is_op("-"):
            op = self.advance()
            right = self.parse_term()
            left = S.Binary(op.line, op.col, op.text, left, right)
        return left

    def parse_term(self):
        left = self.parse_unary()
        while self.is_op("*") or self.is_op("/"):
            op = self.advance()
            right = self.parse_unary()
            left = S.Binary(op.line, op.col, op.text, left, right)
        return left

    def parse_unary(self):
        if self.is_op("-"):
            op = self.advance()
            operand = self.parse_unary()
            return S.Unary(op.line, op.col, operand)
        if self.is_op("+"):
            self.error("unary '+' is not supported")
        return self.parse_power()

    def parse_power(self):
        base = self.parse_postfix()
        if self.is_op("^"):
            op = self.advance()
            negative = False
            if self.is_op("-"):
                self.advance()
                negative = True
            if self.tok.kind != "NUMBER" or not self.tok.is_int:
                self.error("the exponent after '^' must be a whole number such as 2 or -1 "
                           f"(found {self.describe()})")
            n = int(self.advance().text)
            if n > 64:
                self.error("exponents above 64 are not supported", op)
            if self.is_op("^"):
                self.error("chained '^' is not supported; write the exponent as one number")
            return S.Power(op.line, op.col, base, -n if negative else n)
        return base

    def parse_postfix(self):
        node = self.parse_primary()
        while self.is_op("["):
            br = self.advance()
            idx = self.expect_int("an index")
            self.expect_op("]", "after the index")
            node = S.Index(br.line, br.col, node, int(idx.text))
        return node

    def parse_primary(self):
        t = self.tok
        if t.kind == "NUMBER":
            self.advance()
            return S.Number(t.line, t.col, t.value, t.is_int)

        if t.kind == "IDENT":
            self.advance()
            if self.is_op("("):
                self.advance()
                if self.is_op(")"):
                    self.error(f"'{t.text}' needs at least one argument")
                args = [self.parse_expr()]
                while self.is_op(","):
                    self.advance()
                    args.append(self.parse_expr())
                self.expect_op(")", "to close the argument list")
                return S.Call(t.line, t.col, t.text, args)
            return S.Name(t.line, t.col, t.text)

        if self.is_op("("):
            self.advance()
            elems = [self.parse_expr()]
            while self.is_op(","):
                self.advance()
                elems.append(self.parse_expr())
            self.expect_op(")", "to close the parenthesis")
            if len(elems) == 1:
                return elems[0]
            return S.PointLit(t.line, t.col, elems)

        if self.is_kw("iterations"):
            self.advance()
            return S.Iterations(t.line, t.col)

        if self.is_kw("gradient"):
            self.advance()
            f = self.expect_ident("a function name after 'gradient'")
            self.expect_kw("at", f"after 'gradient {f.text}'")
            at = self.parse_unary()
            return S.Gradient(t.line, t.col, f.text, (f.line, f.col), at)

        if self.is_kw("minimize") or self.is_kw("maximize"):
            self.advance()
            f = self.expect_ident(f"a function name after '{t.text}'")
            self.expect_kw("from", f"after '{t.text} {f.text}'")
            start = self.parse_unary()
            node = S.Optimize(t.line, t.col, t.text, f.text, (f.line, f.col), start)
            if self.is_kw("using"):
                self.advance()
                s = self.expect_ident("a solver name after 'using'")
                node.solver = s.text
                node.solver_pos = (s.line, s.col)
                if self.is_op("("):
                    self.advance()
                    while not self.is_op(")"):
                        name = self.expect_ident("an option name such as lr or steps")
                        self.expect_op("=", f"after option '{name.text}'")
                        value = self.parse_expr()
                        if name.text in node.options:
                            self.error(f"option '{name.text}' given twice", name)
                        node.options[name.text] = (value, (name.line, name.col))
                        if self.is_op(","):
                            self.advance()
                            continue
                        break
                    self.expect_op(")", "to close the solver options")
            return node

        if t.kind == "STRING":
            self.error("strings can only be used directly in a print statement")
        if t.kind == "KEYWORD":
            self.error(f"'{t.text}' cannot be used here")
        self.error(f"expected a value, found {self.describe()}")


def parse(source):
    return Parser(tokenize(source)).parse_program()
