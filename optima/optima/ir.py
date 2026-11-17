"""Expression IR, simplification and symbolic differentiation.

After the semantic pass every expression is a tree of `E` nodes. Nodes are
*interned*: building the same expression twice returns the same object, so
"is this the same subexpression?" is a cheap identity check. The code generator
relies on that to compute each shared subexpression only once.

Scalar kinds (allowed inside function bodies, and therefore differentiable):
    const  value            a number
    param  i                the i-th parameter of the enclosing function
    global name index       a global scalar (index -1) or element of a global point
    add/sub/mul/div a b
    neg    a
    pow    a n              integer power (n is a Python int)
    fn     name a           built-in: sqrt exp log sin cos

Top-level-only scalar kinds:
    call   fname args       user function called with scalar arguments
    callp  fname P          user function called with a point
    elem   P i              coordinate i of a point-valued expression
    iters                   iterations used by the last minimize/maximize

Point kinds:
    plit   elems            (a, b, c)
    pvar   name n           a global point variable
    pgrad  fname P          gradient f at P
    popt   kind fname P solver opts     minimize / maximize
"""

import math


class E:
    __slots__ = ("kind", "args")
    _table = {}

    def __new__(cls, kind, *args):
        key = (kind,) + args
        if kind == "const":
            # keep +0.0 and -0.0 apart, and make ints and floats the same node
            key = (kind, float(args[0]), math.copysign(1.0, args[0]))
        node = cls._table.get(key)
        if node is None:
            node = super().__new__(cls)
            node.kind = kind
            node.args = args if kind != "const" else (float(args[0]),)
            cls._table[key] = node
        return node

    def __repr__(self):
        return show(self)


# ------------------------------------------------------------- constructors
# These simplify as they build, which keeps derivatives small.

def const(v):
    return E("const", float(v))


ZERO = const(0.0)
ONE = const(1.0)
TWO = const(2.0)


def is_const(e, v=None):
    return e.kind == "const" and (v is None or e.args[0] == v)


def _finite(v):
    return not (math.isinf(v) or math.isnan(v))


def neg(a):
    if a.kind == "const":
        return const(-a.args[0])
    if a.kind == "neg":
        return a.args[0]
    return E("neg", a)


def add(a, b):
    if a.kind == "const" and b.kind == "const":
        return const(a.args[0] + b.args[0])
    if is_const(a, 0):
        return b
    if is_const(b, 0):
        return a
    if b.kind == "neg":
        return sub(a, b.args[0])
    if a.kind == "neg":
        return sub(b, a.args[0])
    return E("add", a, b)


def sub(a, b):
    if a.kind == "const" and b.kind == "const":
        return const(a.args[0] - b.args[0])
    if is_const(b, 0):
        return a
    if is_const(a, 0):
        return neg(b)
    if b.kind == "neg":
        return add(a, b.args[0])
    if a is b:
        return ZERO
    return E("sub", a, b)


def mul(a, b):
    if a.kind == "const" and b.kind == "const":
        return const(a.args[0] * b.args[0])
    if is_const(a, 0) or is_const(b, 0):
        return ZERO
    if is_const(a, 1):
        return b
    if is_const(b, 1):
        return a
    if is_const(a, -1):
        return neg(b)
    if is_const(b, -1):
        return neg(a)
    if a.kind == "neg" and b.kind == "neg":
        return mul(a.args[0], b.args[0])
    if a.kind == "neg":
        return neg(mul(a.args[0], b))
    if b.kind == "neg":
        return neg(mul(a, b.args[0]))
    # put constants on the left so that 2*x and x*2 become the same node
    if b.kind == "const":
        a, b = b, a
    return E("mul", a, b)


def div(a, b):
    if a.kind == "const" and b.kind == "const" and b.args[0] != 0:
        return const(a.args[0] / b.args[0])
    if is_const(a, 0) and not is_const(b, 0):
        return ZERO
    if is_const(b, 1):
        return a
    if is_const(b, -1):
        return neg(a)
    if a.kind == "neg" and b.kind == "neg":
        return div(a.args[0], b.args[0])
    if a.kind == "neg":
        return neg(div(a.args[0], b))
    if b.kind == "neg":
        return neg(div(a, b.args[0]))
    return E("div", a, b)


def powi(a, n):
    if n == 0:
        return ONE
    if n == 1:
        return a
    if a.kind == "const":
        try:
            v = a.args[0] ** n
            if _finite(v):
                return const(v)
        except (ZeroDivisionError, OverflowError):
            pass
    if a.kind == "pow":
        return powi(a.args[0], a.args[1] * n)
    return E("pow", a, n)


_FOLD = {"sqrt": math.sqrt, "exp": math.exp, "log": math.log,
         "sin": math.sin, "cos": math.cos}


def fn(name, a):
    if a.kind == "const":
        try:
            v = _FOLD[name](a.args[0])
            if _finite(v):
                return const(v)
        except (ValueError, OverflowError):
            pass   # e.g. log(-1): leave it for run time (gives nan)
    return E("fn", name, a)


_BIN = {"add": add, "sub": sub, "mul": mul, "div": div}


# ---------------------------------------------------------- differentiation

def diff(e, i, _memo=None):
    """Partial derivative of scalar expression `e` with respect to parameter i."""
    if _memo is None:
        _memo = {}
    key = (e, i)
    if key in _memo:
        return _memo[key]
    k = e.kind
    if k in ("const", "global"):
        r = ZERO
    elif k == "param":
        r = ONE if e.args[0] == i else ZERO
    elif k == "add":
        r = add(diff(e.args[0], i, _memo), diff(e.args[1], i, _memo))
    elif k == "sub":
        r = sub(diff(e.args[0], i, _memo), diff(e.args[1], i, _memo))
    elif k == "mul":
        a, b = e.args
        r = add(mul(diff(a, i, _memo), b), mul(a, diff(b, i, _memo)))
    elif k == "div":
        # (a/b)' = (a' - (a/b) * b') / b      -- reuses the quotient itself
        a, b = e.args
        r = div(sub(diff(a, i, _memo), mul(e, diff(b, i, _memo))), b)
    elif k == "neg":
        r = neg(diff(e.args[0], i, _memo))
    elif k == "pow":
        # (a^n)' = n * a^(n-1) * a'
        a, n = e.args
        r = mul(mul(const(n), powi(a, n - 1)), diff(a, i, _memo))
    elif k == "fn":
        name, u = e.args
        du = diff(u, i, _memo)
        if name == "exp":
            r = mul(e, du)                       # exp(u)' = exp(u) * u'
        elif name == "log":
            r = div(du, u)                       # log(u)' = u' / u
        elif name == "sin":
            r = mul(fn("cos", u), du)            # sin(u)' = cos(u) * u'
        elif name == "cos":
            r = neg(mul(fn("sin", u), du))       # cos(u)' = -sin(u) * u'
        elif name == "sqrt":
            r = div(du, mul(TWO, e))             # sqrt(u)' = u' / (2 sqrt(u))
        else:
            raise AssertionError(name)
    else:
        raise AssertionError(f"cannot differentiate '{k}'")
    _memo[key] = r
    return r


def gradient(e, n):
    memo = {}
    return [diff(e, i, memo) for i in range(n)]


def hessian(e, n):
    """n x n matrix of second derivatives (symmetric: [i][j] is [j][i])."""
    g = gradient(e, n)
    memo = {}
    h = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            h[i][j] = h[j][i] = diff(g[i], j, memo)
    return h


# --------------------------------------------------------------- utilities

def substitute(e, args, _memo=None):
    """Replace param i by args[i] (used to inline one function into another)."""
    if _memo is None:
        _memo = {}
    if e in _memo:
        return _memo[e]
    k = e.kind
    if k == "param":
        r = args[e.args[0]]
    elif k in ("const", "global"):
        r = e
    elif k in _BIN:
        r = _BIN[k](substitute(e.args[0], args, _memo), substitute(e.args[1], args, _memo))
    elif k == "neg":
        r = neg(substitute(e.args[0], args, _memo))
    elif k == "pow":
        r = powi(substitute(e.args[0], args, _memo), e.args[1])
    elif k == "fn":
        r = fn(e.args[0], substitute(e.args[1], args, _memo))
    else:
        raise AssertionError(k)
    _memo[e] = r
    return r


def expand_powers(e, _memo=None):
    """Rewrite integer powers as multiplications (square-and-multiply).

    x^5 becomes (x*x)*(x*x)*x; because nodes are interned the repeated x*x is
    one node and is computed once.
    """
    if _memo is None:
        _memo = {}
    if e in _memo:
        return _memo[e]
    k = e.kind
    if k == "pow":
        base = expand_powers(e.args[0], _memo)
        n = e.args[1]
        result, sq, m = None, base, abs(n)
        while m:
            if m & 1:
                result = sq if result is None else E("mul", result, sq)
            m >>= 1
            if m:
                sq = E("mul", sq, sq)
        r = result if n > 0 else E("div", ONE, result)
    elif k in _BIN:
        r = E(k, expand_powers(e.args[0], _memo), expand_powers(e.args[1], _memo))
    elif k == "neg":
        r = E("neg", expand_powers(e.args[0], _memo))
    elif k == "fn":
        r = E("fn", e.args[0], expand_powers(e.args[1], _memo))
    elif k == "call":
        r = E("call", e.args[0], tuple(expand_powers(a, _memo) for a in e.args[1]))
    else:
        r = e      # leaves and point-related nodes are handled by the code generator
    _memo[e] = r
    return r


def show(e, names=None):
    """Readable text form, used in comments in the generated assembly.

    `names` are the parameter names of the enclosing function, if any.
    """
    def show_(x):
        return show(x, names)
    k = e.kind
    if k == "const":
        v = e.args[0]
        return str(int(v)) if v == int(v) and abs(v) < 1e15 else repr(v)
    if k == "param":
        return names[e.args[0]] if names else f"x{e.args[0]}"
    if k == "global":
        return e.args[0] if e.args[1] < 0 else f"{e.args[0]}[{e.args[1]}]"
    if k in ("add", "sub", "mul", "div"):
        op = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[k]
        return f"({show_(e.args[0])} {op} {show_(e.args[1])})"
    if k == "neg":
        return f"-{show_(e.args[0])}"
    if k == "pow":
        return f"{show_(e.args[0])}^{e.args[1]}"
    if k == "fn":
        return f"{e.args[0]}({show_(e.args[1])})"
    if k == "call":
        return f"{e.args[0]}({', '.join(show_(a) for a in e.args[1])})"
    if k == "callp":
        return f"{e.args[0]}({show_(e.args[1])})"
    if k == "elem":
        return f"{show_(e.args[0])}[{e.args[1]}]"
    if k == "iters":
        return "iterations"
    if k == "plit":
        return "(" + ", ".join(show_(a) for a in e.args[0]) + ")"
    if k == "pvar":
        return e.args[0]
    if k == "pgrad":
        return f"gradient {e.args[0]} at {show_(e.args[1])}"
    if k == "popt":
        return f"{e.args[0]} {e.args[1]} from {show_(e.args[2])} using {e.args[3]}"
    return k
