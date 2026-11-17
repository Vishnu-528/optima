#!/usr/bin/env python3
"""End-to-end tests: compile Optima programs, run them, check the output.

    python tests/test_compiler.py             run in the built-in simulator
    OPTIMA_TARGET=qemu python tests/test_compiler.py        QEMU back end, run in QEMU
    RIPES=/path/to/Ripes python tests/test_compiler.py      run in real Ripes
"""

import math
import os
import random
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from optima import compile_source, ir, sema, sim   # noqa: E402
from optima.errors import OptimaError               # noqa: E402
from optima.parser import parse                     # noqa: E402

FAILED = []


TARGET = os.environ.get("OPTIMA_TARGET", "ripes")


def compile_for_target(source, name):
    return compile_source(source, name, target=TARGET)


def execute(asm):
    if TARGET == "qemu":
        from optima import qemu
        return qemu.run(asm)
    ripes = os.environ.get("RIPES")
    if not ripes:
        return sim.run(asm)[0]
    with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as fh:
        fh.write(asm)
        path = fh.name
    try:
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        res = subprocess.run([ripes, "--mode", "cli", "--src", path, "-t", "asm",
                              "--proc", "RV32_SS", "--isaexts", "M"],
                             capture_output=True, text=True, env=env, timeout=600)
        out = res.stdout
        cut = out.rfind("\nProgram exited with code")
        return out[:cut] if cut >= 0 else out
    finally:
        os.unlink(path)


NUM = re.compile(r"[-+]?(?:\d+\.?\d*(?:e[-+]?\d+)?|inf|nan)")


def check(name, source, expected, tol=2e-4):
    """Run `source`; its output must match `expected` line by line. Numbers
    are compared with tolerance (absolute or relative `tol`), text exactly."""
    try:
        got = execute(compile_for_target(source, name)).strip("\n").split("\n")
    except OptimaError as e:
        FAILED.append(name)
        print(f"FAIL {name}: {e.format(source, name)}")
        return
    want = expected.strip("\n").split("\n")
    ok = len(got) == len(want)
    if ok:
        for g, w in zip(got, want):
            gn, wn = NUM.findall(g), NUM.findall(w)
            if NUM.sub("#", g).split() != NUM.sub("#", w).split() or len(gn) != len(wn):
                ok = False
                break
            for a, b in zip(gn, wn):
                a, b = float(a), float(b)
                if math.isnan(b) or math.isinf(b):
                    if not (str(a) == str(b)):
                        ok = False
                elif abs(a - b) > tol * max(1.0, abs(b)):
                    ok = False
    if ok:
        print(f"ok   {name}")
    else:
        FAILED.append(name)
        print(f"FAIL {name}\n  got:      {got}\n  expected: {want}")


def check_error(name, source, fragment, line=None):
    try:
        compile_source(source, name)
    except OptimaError as e:
        if fragment in e.message and (line is None or e.line == line):
            print(f"ok   error: {name}")
        else:
            FAILED.append(name)
            print(f"FAIL error: {name}\n  got: line {e.line}: {e.message}\n  expected: line {line}: ...{fragment}...")
        return
    FAILED.append(name)
    print(f"FAIL error: {name}: compiled without error")


# ------------------------------------------------------------ program tests

def program_tests():
    check("arithmetic", """
a = 7
b = 2
print a + b, a - b, a * b, a / b
print -a + b * 3 - (a - b) / 2
print a^2, b^10, a^0, b^-2, -b^2, (-b)^2
print 1e3, 2.5e-3, .5, 10 / 4
""", """
9 5 14 3.5
-3.5
49 1024 1 0.25 -4 4
1000 0.0025 0.5 2.5
""")

    check("builtins", """
print sqrt(16), sqrt(2), exp(1), log(10), sin(1), cos(1)
x = 0.3
print sin(x)^2 + cos(x)^2, exp(log(x)), sqrt(x * x)
print log(0), sqrt(-1), 1 / 0, exp(1000)
""", f"""
4 {math.sqrt(2)} {math.e} {math.log(10)} {math.sin(1)} {math.cos(1)}
1 0.3 0.3
-inf nan inf inf
""")

    check("functions and calls", """
k = 2
sq(x) = x^2
f(x, y) = sq(x) + k * y
g(a) = f(a, a + 1) - sq(a)
print sq(3), f(3, 4), g(5)
k = 10
print f(3, 4)
p = (1, 2)
print f(p), f(p[1], p[0]), f((5, 6))
q = (p[1] * 10, f(p))
print q, q[0] + q[1]
p = (q[1], q[0])
print p
""", """
9 17 12
49
21 14 85
(20, 21) 41
(21, 20)
""")

    check("gradient", """
f(x, y) = x^3 * y + sin(x * y) - y / x + exp(-x) * log(y) + sqrt(x + y)
p = (1.5, 2)
g = gradient f at p
print g
print (gradient f at (1.5, 2))[1]
h(t) = (t^2 + 1) / (t - 3)
print gradient h at 1, gradient h at -2.5
""", f"""
({3*1.5**2*2 + 2*math.cos(3) + 2/1.5**2 - math.exp(-1.5)*math.log(2) + 0.5/math.sqrt(3.5)}, {1.5**3 + 1.5*math.cos(3) - 1/1.5 + math.exp(-1.5)/2 + 0.5/math.sqrt(3.5)})
{1.5**3 + 1.5*math.cos(3) - 1/1.5 + math.exp(-1.5)/2 + 0.5/math.sqrt(3.5)}
{(2*1*(1-3) - (1+1))/(1-3)**2} {(2*-2.5*(-2.5-3) - (6.25+1))/(-5.5)**2}
""")

    check("gradient descent", """
target = 3
f(x, y) = (x - target)^2 + (y + 1)^2
r = minimize f from (0, 0) using gradient_descent(lr=0.1, steps=500, tol=1e-6)
print r, f(r), r[0], r[1]
print iterations
s = minimize f from (0, 0) using gradient_descent(lr=0.1, steps=5)
print s, iterations
d = minimize f from r
print d, iterations
""", f"""
(3, -1) 0 3 -1
71
({3 * (1 - 0.8**5)}, {-(1 - 0.8**5)}) 5
(3, -1) 0
""")

    check("maximize", """
h(x) = 5 - (x - 2)^2
top = maximize h from 0 using gradient_descent(lr=0.2)
print top, top[0], h(top), top * 2
g(x, y) = -(x - 1)^2 - 2 * (y + 3)^2 + x * y
m = maximize g from (0, 0) using momentum(lr=0.05, beta=0.5)
print m
n = maximize g from (0, 0) using newton
print n, iterations
""", f"""
2 2 5 4
({-4/7}, {-22/7})
({-4/7}, {-22/7}) 1
""")

    check("newton", """
rosen(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
p = minimize rosen from (-1.2, 1) using newton(steps=50, tol=1e-5)
print p, iterations
q(a, b, c) = (a - 1)^2 + (b - 2)^2 + (c - 3)^2 + a * b + b * c + exp(a - c) / 10
s = minimize q from (0, 0, 0) using newton
t = minimize q from (0, 0, 0) using gradient_descent(lr=0.2, steps=2000, tol=1e-5)
print s[0] - t[0], s[1] - t[1], s[2] - t[2]
w(x) = x^4 - 3 * x^2 + x
print minimize w from 2 using newton, minimize w from -2 using newton
""", """
(1, 1) 6
0 0 0
1.1309 -1.30084
""", tol=5e-4)

    check("newton needs pivoting", """
# the Hessian [[0, 1, 0], [1, 0, 0], [0, 0, 2]] has a zero in the first pivot position
f(x, y, z) = x * y + (z - 1)^2 + x + 2 * y
p = minimize f from (5, 5, 5) using newton
print p, iterations
""", """
(-2, -1, 1) 1
""")

    check("newton singular", """
f(x, y) = (x + y - 2)^2
p = minimize f from (3, 4) using newton(steps=7)
print p, iterations
""", """
(3, 4) 0
""")

    check("momentum", """
f(x, y) = (x - 1)^2 + 10 * (y - 2)^2 + x * y
b = minimize f from (0, 0) using momentum(lr=0.02, beta=0.8, steps=2000)
print b[0], b[1]
c = minimize f from (0, 0) using momentum
print c[0], c[1]
""", """
0 2
0 2
""", tol=1e-4)

    check("options as expressions", """
rate = 0.05
f(x) = (x - 4)^2
a = minimize f from 1 using gradient_descent(lr=rate * 2, tol=rate / 1000, steps=300)
print a, iterations
n = iterations
print n + 1, iterations * 2, n / 2
""", """
4 53
54 106 26.5
""", tol=1e-3)

    check("iterations before any solve, strings", """
print iterations
print "a b", 1, "c, d (e) #f", 2.5
print "x = ", 3
""", """
0
a b 1 c, d (e) #f 2.5
x =  3
""")

    check("continuation and comments", """
# comment line
f(x,
  y) = (x     # inside parentheses a newline does not end the statement
        + y)
print f(1,
        2)   # trailing comment
""", """
3
""")

    check("shared subexpressions", """
f(x, y) = exp(x * y) * exp(x * y) + sin(x * y) / (1 + exp(x * y))
print f(0.5, 0.25), gradient f at (0.5, 0.25)
""", f"""
{math.exp(0.125)**2 + math.sin(0.125)/(1+math.exp(0.125))} ({0.25*(2*math.exp(0.25) + (math.cos(0.125)*(1+math.exp(0.125)) - math.sin(0.125)*math.exp(0.125))/(1+math.exp(0.125))**2)}, {0.5*(2*math.exp(0.25) + (math.cos(0.125)*(1+math.exp(0.125)) - math.sin(0.125)*math.exp(0.125))/(1+math.exp(0.125))**2)})
""")

    check("deep right-nested expression", """
f(x) = 1 + x * (2 + x * (3 + x * (4 + x * (5 + x * (6 + x * (7 + x * 8))))))
g(x) = (((((((8 / x + 7) / x + 6) / x + 5) / x + 4) / x + 3) / x + 2) / x + 1)
print f(1), f(2), g(1), g(2)
print (1 + (2 * (3 + (4 * (5 + (6 * (7 + 8)))))))
print (f(1) + f(2)) * (g(1) + g(2)) - (f(1) - g(2)) * (f(2) - g(1))
""", f"""
36 1793 36 {sum((i + 1) / 2**i for i in range(8))}
767
{(36 + 1793) * (36 + sum((i + 1) / 2**i for i in range(8))) - (36 - sum((i + 1) / 2**i for i in range(8))) * (1793 - 36)}
""")

    check("reassigned point used by the function", """
c = (1, 2)
f(x, y) = (x - c[0])^2 + (y - c[1])^2 + x * y / 4
c = minimize f from c using newton
print c
c = (c[1], c[0])
print c
""", f"""
({16/21}, {40/21})
({40/21}, {16/21})
""")

    # many parameters: a frame bigger than 2 KiB and a 12-variable gradient
    n = 12
    params = ", ".join(f"v{i}" for i in range(n))
    body = " + ".join(f"(v{i} - {i + 1})^2" for i in range(n))
    start = ", ".join("0" for _ in range(n))
    want = ", ".join(str(i + 1) for i in range(n))
    check("twelve variables", f"""
f({params}) = {body}
p = minimize f from ({start}) using gradient_descent(lr=0.25, steps=200)
print p
q = minimize f from ({start}) using newton
print q, iterations
""", f"""
({want})
({want}) 1
""")

    # 200 parameters: argument areas and solver scratch far from the frame pointer
    n = 200
    params = ", ".join(f"v{i}" for i in range(n))
    body = " + ".join(f"(v{i} - {i % 7})^2" for i in range(n))
    zeros = ", ".join("0" for _ in range(n))
    ones = ", ".join("1" for _ in range(n))
    check("two hundred variables", f"""
f({params}) = {body}
p = minimize f from ({zeros}) using gradient_descent(lr=0.5, steps=3)
print p[0], p[6], p[199], f(p), iterations, f({ones})
print (gradient f at ({ones}))[5]
""", f"""
0 6 {199 % 7} 0 1 {sum((1 - i % 7) ** 2 for i in range(n))}
-8
""")

    # hundreds of shared subexpressions: a function frame bigger than 2 KiB
    terms = 300
    body = " + ".join(f"sin(x * {i + 1}) * sin(x * {i + 1}) / {i + 1}" for i in range(terms))
    x = 0.37
    check("large function frame", f"""
f(x) = {body}
print f({x}), gradient f at {x}
""", f"""
{sum(math.sin(x * (i + 1)) ** 2 / (i + 1) for i in range(terms))} {sum(2 * math.sin(x * (i + 1)) * math.cos(x * (i + 1)) for i in range(terms))}
""", tol=2e-3)


# ----------------------------------------------------------- error messages

def error_tests():
    check_error("unknown name", "x = 1\nprint y\n", "unknown name 'y'", 2)
    check_error("unknown name suggestion", "target = 1\nprint targat\n", "did you mean 'target'", 2)
    check_error("unknown function", "print f(1)\n", "unknown function 'f'", 1)
    check_error("arg count", "f(x, y) = x + y\nprint f(1)\n", "takes 2 arguments (x, y), but 1 was given", 2)
    check_error("builtin arg count", "print sin(1, 2)\n", "takes exactly 1 argument", 1)
    check_error("point size", "f(x, y) = x + y\nr = minimize f from (0, 0, 0)\n", "needs a point with 2 coordinates", 2)
    check_error("point as number", "p = (1, 2)\nx = p + 1\n", "a single number is needed", 2)
    check_error("index range", "p = (1, 2)\nprint p[2]\n", "out of range", 2)
    check_error("index scalar", "x = 1\nprint x[0]\n", "only points can be indexed", 2)
    check_error("unknown solver", "f(x) = x^2\nr = minimize f from 0 using adam\n", "unknown solver 'adam'", 2)
    check_error("unknown option", "f(x) = x^2\nr = minimize f from 0 using newton(lr=1)\n", "has no option 'lr'", 2)
    check_error("bad steps", "f(x) = x^2\nr = minimize f from 0 using newton(steps=1.5)\n", "steps must be a whole number", 2)
    check_error("float exponent", "x = 2 ^ 1.5\n", "must be a whole number", 1)
    check_error("variable exponent", "n = 2\nx = 2 ^ n\n", "must be a whole number", 2)
    check_error("redefine function", "f(x) = x\nf(x) = x + 1\n", "already defined", 2)
    check_error("function as variable", "f(x) = x\nf = 3\n", "is a function", 2)
    check_error("variable as function", "a = 3\nprint a(1)\n", "is a variable, not a function", 2)
    check_error("type change", "a = 3\na = (1, 2)\n", "cannot become", 2)
    check_error("minimize in function", "f(x) = x\ng(x) = minimize f from x\n", "cannot be used inside a function body", 2)
    check_error("builtin as objective", "r = minimize sin from 0\n", "built-in function", 1)
    check_error("statement", "3 + 4\n", "start of a statement", 1)
    check_error("call statement", "f(x) = x\nf(3)\n", "function call cannot be a statement", 2)
    check_error("missing paren", "x = (1 + 2\n", "expected ')'", None)
    check_error("bad char", "x = 1 $ 2\n", "unexpected character '$'", 1)
    check_error("unterminated string", 'print "abc\n', "unterminated string", 1)
    check_error("string in expression", 'x = "abc"\n', "strings can only be used", 1)
    check_error("keyword as name", "from = 1\n", "cannot start a statement", 1)
    check_error("use before definition", "print g(1)\ng(x) = x\n", "must be defined before", 1)
    check_error("extra tokens", "x = 1 2\n", "after the end of the statement", 1)
    check_error("empty print", "print\n", "needs something to print", 1)
    check_error("missing from", "f(x) = x\nr = minimize f (0)\n", "expected 'from'", 2)
    check_error("builtin name", "sin = 3\n", "built-in function", 1)


# ------------------------------------------------- differentiation property

def evaluate(e, params):
    k = e.kind
    if k == "const":
        return e.args[0]
    if k == "param":
        return params[e.args[0]]
    a = [evaluate(x, params) if isinstance(x, ir.E) else x for x in e.args]
    if k == "add":
        return a[0] + a[1]
    if k == "sub":
        return a[0] - a[1]
    if k == "mul":
        return a[0] * a[1]
    if k == "div":
        return a[0] / a[1]
    if k == "neg":
        return -a[0]
    if k == "pow":
        return a[0] ** a[1]
    if k == "fn":
        return {"sqrt": math.sqrt, "exp": math.exp, "log": math.log,
                "sin": math.sin, "cos": math.cos}[a[0]](a[1])
    raise AssertionError(k)


def random_expr(rng, depth):
    if depth == 0 or rng.random() < 0.15:
        return rng.choice(["x", "y", "z", f"{rng.uniform(0.5, 3):.3f}"])
    kind = rng.randrange(8)
    a = random_expr(rng, depth - 1)
    b = random_expr(rng, depth - 1)
    if kind == 0:
        return f"({a} + {b})"
    if kind == 1:
        return f"({a} - {b})"
    if kind == 2:
        return f"({a} * {b})"
    if kind == 3:
        return f"({a} / (2 + ({b})^2))"
    if kind == 4:
        return f"({a})^{rng.choice([2, 3, -1, 4])}" if "^-1" not in a else f"({a})^2"
    if kind == 5:
        return f"{rng.choice(['sin', 'cos'])}({a})"
    if kind == 6:
        return f"exp(({a}) / 8)"
    return f"{rng.choice(['sqrt', 'log'])}(1 + ({a})^2)"


def differentiation_tests():
    """Symbolic derivatives must agree with central finite differences."""
    rng = random.Random(7)
    bad = checked = 0
    for _ in range(300):
        text = random_expr(rng, 4)
        prog = sema.analyze(parse(f"f(x, y, z) = {text}\n"))
        body = prog.functions["f"].body
        grad = ir.gradient(body, 3)
        hess = ir.hessian(body, 3)
        low_body = ir.expand_powers(body)
        point = [rng.uniform(0.6, 2.0) for _ in range(3)]
        try:
            f0 = evaluate(body, point)
            if abs(evaluate(low_body, point) - f0) > 1e-9 * max(1, abs(f0)):
                bad += 1
                print(f"  power expansion changed the value of: {text}")
            for i in range(3):
                h = 1e-5
                up = point[:]
                dn = point[:]
                up[i] += h
                dn[i] -= h
                numeric = (evaluate(body, up) - evaluate(body, dn)) / (2 * h)
                symbolic = evaluate(grad[i], point)
                checked += 1
                if abs(numeric - symbolic) > 1e-5 * max(1.0, abs(symbolic)):
                    bad += 1
                    print(f"  d/d{'xyz'[i]} of {text}: symbolic {symbolic}, numeric {numeric}")
                for j in range(3):
                    numeric2 = (evaluate(grad[i], [p + (h if q == j else 0) for q, p in enumerate(point)])
                                - evaluate(grad[i], [p - (h if q == j else 0) for q, p in enumerate(point)])) / (2 * h)
                    symbolic2 = evaluate(hess[i][j], point)
                    checked += 1
                    if abs(numeric2 - symbolic2) > 1e-4 * max(1.0, abs(symbolic2)):
                        bad += 1
                        print(f"  d2/d{'xyz'[i]}d{'xyz'[j]} of {text}: symbolic {symbolic2}, numeric {numeric2}")
        except (ZeroDivisionError, OverflowError, ValueError):
            continue
    if bad:
        FAILED.append("differentiation")
    print(f"{'FAIL' if bad else 'ok  '} differentiation: {checked - bad}/{checked} derivatives agree "
          f"with finite differences")


def random_program_tests():
    """Compiled value and gradient of random functions must match Python."""
    rng = random.Random(99)
    for idx in range(12):
        text = random_expr(rng, 4)
        prog = sema.analyze(parse(f"f(x, y, z) = {text}\n"))
        body = prog.functions["f"].body
        point = [round(rng.uniform(0.6, 2.0), 3) for _ in range(3)]
        try:
            value = evaluate(body, point)
            grad = [evaluate(g, point) for g in ir.gradient(body, 3)]
        except (ZeroDivisionError, OverflowError, ValueError):
            continue
        if max(abs(v) for v in [value] + grad) > 1e6:
            continue
        pt = ", ".join(str(p) for p in point)
        check(f"random function {idx}", f"f(x, y, z) = {text}\nprint f({pt})\nprint gradient f at ({pt})\n",
              f"{value}\n({grad[0]}, {grad[1]}, {grad[2]})\n", tol=2e-3)


def main():
    backend = "Ripes" if os.environ.get("RIPES") else "the built-in simulator"
    if TARGET == "qemu":
        backend = "QEMU (hardware floats)"
    print(f"Compiler tests on {backend}")
    program_tests()
    random_program_tests()
    error_tests()
    differentiation_tests()
    print("ALL PASSED" if not FAILED else f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
