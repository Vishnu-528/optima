"""Semantic analysis: syntax tree -> checked program in IR form.

Checks names, argument counts and point sizes, and turns every expression
into `ir.E` nodes. Calls from one function to another are inlined here, so a
function body ends up as one self-contained expression that can be
differentiated.

Types: "scalar" (a float), "int" (only `iterations`), ("point", n).
"""

import difflib
from dataclasses import dataclass, field

from . import ir
from . import syntax as S
from .errors import OptimaError
from .ir import E

BUILTINS = ("sqrt", "exp", "log", "sin", "cos")

# solver name -> option name -> default value
SOLVERS = {
    "gradient_descent": {"lr": 0.1, "steps": 1000, "tol": 1e-6},
    "momentum": {"lr": 0.01, "beta": 0.9, "steps": 1000, "tol": 1e-6},
    "newton": {"steps": 50, "tol": 1e-6},
}

MAX_NEWTON_DIM = 16
MAX_PARAMS = 256


@dataclass
class Function:
    name: str
    params: list
    body: E
    line: int
    needs_grad: bool = False
    needs_hess: bool = False


@dataclass
class Program:
    globals: dict = field(default_factory=dict)     # name -> "scalar" | ("point", n)
    functions: dict = field(default_factory=dict)   # name -> Function
    statements: list = field(default_factory=list)  # see Analyzer.statement
    uses_iterations: bool = False


def _suggest(name, candidates):
    close = difflib.get_close_matches(name, list(candidates), n=1)
    return f" (did you mean '{close[0]}'?)" if close else ""


def _type_name(t):
    if t == "scalar":
        return "a number"
    if t == "int":
        return "a whole number"
    return f"a point with {t[1]} coordinate{'s' if t[1] != 1 else ''}"


class Analyzer:
    def __init__(self):
        self.prog = Program()

    def err(self, msg, node=None, pos=None):
        if pos is not None:
            raise OptimaError(msg, pos[0], pos[1])
        raise OptimaError(msg, node.line, node.col)

    # ---------------------------------------------------------- statements
    def analyze(self, stmts):
        for st in stmts:
            self.statement(st)
        return self.prog

    def statement(self, st):
        if isinstance(st, S.FuncDef):
            self.funcdef(st)
        elif isinstance(st, S.Assign):
            self.assign(st)
        elif isinstance(st, S.Print):
            items = []
            for item in st.items:
                if isinstance(item, S.Str):
                    items.append(("str", item.text))
                    continue
                e, t = self.expr(item, None)
                if t == "int":
                    items.append(("int", e))
                elif t == "scalar" or t[1] == 1:
                    # a point with one coordinate prints as a plain number
                    items.append(("scalar", self.to_scalar(e, t, item)))
                else:
                    items.append(("point", e, t[1]))
            self.prog.statements.append(("print", st.line, items))
        else:
            raise AssertionError(st)

    def check_new_name(self, name, node, what):
        if name in BUILTINS:
            self.err(f"'{name}' is a built-in function and cannot be used as a {what} name", node)
        if name in SOLVERS:
            self.err(f"'{name}' is a solver name and cannot be used as a {what} name", node)

    def funcdef(self, st):
        self.check_new_name(st.name, st, "function")
        if st.name in self.prog.functions:
            self.err(f"function '{st.name}' is already defined "
                     f"(line {self.prog.functions[st.name].line})", st)
        if st.name in self.prog.globals:
            self.err(f"'{st.name}' is already a variable; pick another name for the function", st)
        params = [p[0] for p in st.params]
        if len(params) > MAX_PARAMS:
            self.err(f"a function can have at most {MAX_PARAMS} parameters", st)
        for p, line, col in st.params:
            if p in self.prog.functions or p == st.name:
                self.err(f"parameter '{p}' has the same name as a function", pos=(line, col))
        scope = {p: i for i, p in enumerate(params)}
        body, t = self.expr(st.body, scope)
        body = self.to_scalar(body, t, st.body)
        self.prog.functions[st.name] = Function(st.name, params, body, st.line)

    def assign(self, st):
        self.check_new_name(st.name, st, "variable")
        if st.name in self.prog.functions:
            self.err(f"'{st.name}' is a function; pick another name for the variable", st)
        e, t = self.expr(st.value, None)
        if t == "int":
            t = "scalar"       # stored as a float
        old = self.prog.globals.get(st.name)
        if old is not None and old != t:
            self.err(f"'{st.name}' was {_type_name(old)} before and cannot become "
                     f"{_type_name(t)}", st)
        self.prog.statements.append(("assign", st.line, st.name, e, t, old is None))
        self.prog.globals[st.name] = t

    # --------------------------------------------------------- expressions
    def to_scalar(self, e, t, node):
        if t in ("scalar", "int"):
            return e
        if t[1] == 1:
            return self.element(e, 0)
        self.err(f"this is {_type_name(t)}, but a single number is needed here; "
                 f"pick one coordinate with [0] .. [{t[1] - 1}]", node)

    def to_point(self, e, t, n, node, fname):
        if t == ("point", n):
            return e
        if t in ("scalar", "int") and n == 1:
            return E("plit", (e,))
        f = self.prog.functions[fname]
        self.err(f"'{fname}' has {n} parameter{'s' if n != 1 else ''} "
                 f"({', '.join(f.params)}), so it needs a point with {n} "
                 f"coordinate{'s' if n != 1 else ''}, but this is {_type_name(t)}", node)

    def element(self, p, i):
        if p.kind == "pvar":
            return E("global", p.args[0], i)
        if p.kind == "plit":
            return p.args[0][i]
        return E("elem", p, i)

    def lookup_function(self, name, pos):
        if name in self.prog.functions:
            return self.prog.functions[name]
        if name in BUILTINS:
            self.err(f"'{name}' is a built-in function; define your own function with it, "
                     f"for example  f(x) = {name}(x)", pos=pos)
        if name in self.prog.globals:
            self.err(f"'{name}' is a variable, not a function", pos=pos)
        self.err(f"unknown function '{name}'{_suggest(name, self.prog.functions)}; "
                 f"functions must be defined before they are used", pos=pos)

    def expr(self, node, scope):
        """Returns (E, type). `scope` maps parameter names to indexes inside a
        function body and is None at the top level."""
        in_func = scope is not None

        if isinstance(node, S.Number):
            return ir.const(node.value), "scalar"

        if isinstance(node, S.Name):
            name = node.name
            if in_func and name in scope:
                return E("param", scope[name]), "scalar"
            if name in self.prog.globals:
                t = self.prog.globals[name]
                if t == "scalar":
                    return E("global", name, -1), "scalar"
                return E("pvar", name, t[1]), t
            if name in self.prog.functions or name in BUILTINS:
                self.err(f"'{name}' is a function; call it with arguments, like {name}(...)", node)
            known = list(self.prog.globals) + (list(scope) if in_func else [])
            self.err(f"unknown name '{name}'{_suggest(name, known)}", node)

        if isinstance(node, S.Unary):
            e, t = self.expr(node.operand, scope)
            return ir.neg(self.to_scalar(e, t, node.operand)), "scalar"

        if isinstance(node, S.Binary):
            a, ta = self.expr(node.left, scope)
            b, tb = self.expr(node.right, scope)
            a = self.to_scalar(a, ta, node.left)
            b = self.to_scalar(b, tb, node.right)
            build = {"+": ir.add, "-": ir.sub, "*": ir.mul, "/": ir.div}[node.op]
            return build(a, b), "scalar"

        if isinstance(node, S.Power):
            a, ta = self.expr(node.base, scope)
            return ir.powi(self.to_scalar(a, ta, node.base), node.exponent), "scalar"

        if isinstance(node, S.Index):
            p, t = self.expr(node.base, scope)
            if t in ("scalar", "int"):
                self.err("only points can be indexed with [ ]; this is a single number", node)
            if not 0 <= node.index < t[1]:
                self.err(f"index {node.index} is out of range: this point has {t[1]} "
                         f"coordinate{'s' if t[1] != 1 else ''} (valid: 0 to {t[1] - 1})", node)
            return self.element(p, node.index), "scalar"

        if isinstance(node, S.PointLit):
            elems = []
            for el in node.elems:
                e, t = self.expr(el, scope)
                elems.append(self.to_scalar(e, t, el))
            if in_func:
                # only meaningful as the argument list of a call, handled there
                self.err("a point cannot be used inside a function body", node)
            return E("plit", tuple(elems)), ("point", len(elems))

        if isinstance(node, S.Call):
            return self.call(node, scope)

        if isinstance(node, S.Iterations):
            if in_func:
                self.err("'iterations' cannot be used inside a function body", node)
            self.prog.uses_iterations = True
            return E("iters"), "int"

        if isinstance(node, S.Gradient):
            if in_func:
                self.err("'gradient' cannot be used inside a function body", node)
            f = self.lookup_function(node.func, node.func_pos)
            n = len(f.params)
            p, t = self.expr(node.at, scope)
            p = self.to_point(p, t, n, node.at, f.name)
            f.needs_grad = True
            return E("pgrad", f.name, p), ("point", n)

        if isinstance(node, S.Optimize):
            if in_func:
                self.err(f"'{node.kind}' cannot be used inside a function body", node)
            return self.optimize(node)

        raise AssertionError(node)

    def call(self, node, scope):
        name = node.name
        if name in BUILTINS:
            if len(node.args) != 1:
                self.err(f"{name}() takes exactly 1 argument, but {len(node.args)} were given", node)
            a, t = self.expr(node.args[0], scope)
            return ir.fn(name, self.to_scalar(a, t, node.args[0])), "scalar"

        if scope is not None and name in scope:
            self.err(f"'{name}' is a parameter, not a function", node)
        f = self.lookup_function(name, (node.line, node.col))
        n = len(f.params)

        args = [self.expr(a, scope) for a in node.args]
        # f(p) with a point that supplies all the arguments
        if len(args) == 1 and args[0][1] == ("point", n):
            p = args[0][0]
            if scope is not None or p.kind == "plit":
                elems = [self.element(p, i) for i in range(n)]
                if scope is not None:
                    return ir.substitute(f.body, elems), "scalar"
                return E("call", name, tuple(elems)), "scalar"
            return E("callp", name, p), "scalar"
        if len(args) != n:
            self.err(f"'{name}' takes {n} argument{'s' if n != 1 else ''} "
                     f"({', '.join(f.params)}), but {len(args)} "
                     f"{'was' if len(args) == 1 else 'were'} given", node)
        scalars = [self.to_scalar(e, t, a) for (e, t), a in zip(args, node.args)]
        if scope is not None:
            return ir.substitute(f.body, scalars), "scalar"     # inline
        return E("call", name, tuple(scalars)), "scalar"

    def optimize(self, node):
        f = self.lookup_function(node.func, node.func_pos)
        n = len(f.params)
        start, t = self.expr(node.start, None)
        start = self.to_point(start, t, n, node.start, f.name)

        solver = node.solver
        if solver not in SOLVERS:
            self.err(f"unknown solver '{solver}'{_suggest(solver, SOLVERS)}; "
                     f"available: {', '.join(SOLVERS)}", pos=node.solver_pos)
        defaults = SOLVERS[solver]
        opts = {}
        for name, (value, pos) in node.options.items():
            if name not in defaults:
                self.err(f"solver '{solver}' has no option '{name}'"
                         f"{_suggest(name, defaults)}; its options are: "
                         f"{', '.join(defaults)}", pos=pos)
            if name == "steps":
                if not (isinstance(value, S.Number) and value.is_int and value.value >= 1):
                    self.err("steps must be a whole number of at least 1, such as steps=500",
                             value)
                if value.value > 2_000_000_000:
                    self.err("steps is too large", value)
                opts[name] = int(value.value)
            else:
                e, vt = self.expr(value, None)
                opts[name] = self.to_scalar(e, vt, value)
        for name, default in defaults.items():
            if name not in opts:
                opts[name] = default if name == "steps" else ir.const(default)

        f.needs_grad = True
        if solver == "newton":
            if n > MAX_NEWTON_DIM:
                self.err(f"newton supports functions with at most {MAX_NEWTON_DIM} parameters; "
                         f"'{f.name}' has {n}", node)
            f.needs_hess = True
        opt_tuple = tuple(sorted(opts.items()))
        return E("popt", node.kind, f.name, start, solver, opt_tuple), ("point", n)


def analyze(stmts):
    return Analyzer().analyze(stmts)
