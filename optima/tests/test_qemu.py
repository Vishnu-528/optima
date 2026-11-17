#!/usr/bin/env python3
"""Tests for the QEMU back end's runtime routines (needs clang, ld.lld, QEMU).

    python tests/test_qemu.py

Checks, against Python's own arithmetic:
  * number printing: the text must be exactly what C's "%g" gives;
  * exp, log, sin, cos, sqrt and + - * /: the printed result (6 significant
    digits) must be the correctly rounded single-precision answer, allowing
    a difference of one in the last printed digit.
"""

import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from optima import compile_source, qemu      # noqa: E402

BATCH = 420        # a multiple of 2 and 3, so statement groups stay together


def f32(v):
    """Round a Python float to single precision."""
    try:
        return struct.unpack("<f", struct.pack("<f", v))[0]
    except OverflowError:
        return math.copysign(math.inf, v)


def literal(v):
    """Optima source text for the single-precision value v."""
    text = "%.9g" % abs(v)
    return f"(-{text})" if v < 0 or (v == 0 and math.copysign(1, v) < 0) else text


def run_lines(lines):
    out = []
    for i in range(0, len(lines), BATCH):
        src = "\n".join(lines[i:i + BATCH]) + "\n"
        out += qemu.run(compile_source(src, "test", target="qemu")).strip("\n").split("\n")
    return out


def close(got, want_text):
    """Same text, or numerically within one unit of the 6th significant digit."""
    if got == want_text:
        return True
    try:
        g, w = float(got), float(want_text)
    except ValueError:
        return False
    if math.isnan(g) or math.isnan(w) or math.isinf(g) or math.isinf(w):
        return False
    return abs(g - w) <= 1.5e-5 * max(abs(g), abs(w))


def random_values(rng, n, lo_exp, hi_exp, signed=True):
    vals = []
    for _ in range(n):
        v = rng.uniform(1, 10) * 10.0 ** rng.uniform(lo_exp, hi_exp)
        if signed and rng.random() < 0.5:
            v = -v
        v = f32(v)
        if math.isfinite(v):
            vals.append(v)
    return vals


def test_printing(rng):
    vals = [0.0, 1.0, -1.0, 0.5, 0.1, 10.0, 100000.0, 999999.0, 1000000.0, 1234567.0,
            123456.0, 0.0001, 0.00001, 0.000123456, 3.14159265, 2.5e10, 1e-30, 1e30,
            1e-38, 3.4e38, 1e-45, 0.999999, 0.9999999, 9.999995, 99999.95, 16777216.0,
            1.5, 2.0 / 3.0, 100.0, 0.001, 123.456, 1e-5, 1e-4, 1e5, 1e6]
    vals = [f32(v) for v in vals] + random_values(rng, 3000, -44, 38)
    got = run_lines([f"print {literal(v)}" for v in vals])
    exact = sum(g == "%g" % v for g, v in zip(got, vals))
    bad = [(v, g) for g, v in zip(got, vals) if not close(g, "%g" % v)]
    print(f"{'ok  ' if not bad else 'FAIL'} printing: {exact}/{len(vals)} identical to %g, "
          f"{len(vals) - exact - len(bad)} differ by one in the last digit, {len(bad)} wrong")
    for v, g in bad[:10]:
        print(f"       {v!r}: got {g!r}, want {'%g' % v!r}")
    return len(bad)


def test_function(name, expr, vals, ref):
    lines = []
    for v in vals:
        lines.append(f"a = {literal(v)}")
        lines.append(f"print {expr}")
    got = run_lines(lines)
    bad = []
    for v, g in zip(vals, got):
        try:
            want = f32(ref(v))
        except (ValueError, OverflowError):
            want = math.nan
        w = "%g" % want
        if not close(g, w):
            bad.append((v, g, w))
    print(f"{'ok  ' if not bad else 'FAIL'} {name}: {len(vals) - len(bad)}/{len(vals)}")
    for v, g, w in bad[:10]:
        print(f"       {name}({v!r}): got {g}, want {w}")
    return len(bad)


def safe_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def safe_log(x):
    if x == 0:
        return -math.inf
    return math.log(x) if x > 0 else math.nan


def main():
    rng = random.Random(2)
    bad = test_printing(rng)

    xs = [0.0, 1.0, -1.0, 0.5, 88.0, 88.7, 89.0, -87.0, -100.0, -104.0, 200.0, -200.0, 1e-10]
    xs += [f32(rng.uniform(-100, 88)) for _ in range(1500)] + random_values(rng, 300, -20, 0)
    bad += test_function("exp", "exp(a)", xs, safe_exp)

    xs = [1.0, 2.0, 0.5, 2.718281828, 10.0, 1e-38, 1e-45, 3.4e38, 0.0, -1.0, 1.0000001, 0.9999999]
    xs = [f32(x) for x in xs] + random_values(rng, 1500, -44, 38, signed=False)
    xs += [f32(rng.uniform(0.5, 2.0)) for _ in range(500)]
    bad += test_function("log", "log(a)", xs, safe_log)

    xs = [0.0, math.pi / 2, math.pi, -math.pi, 2 * math.pi, 1e-10, 100.0, 1e6, 1e8, 9e8]
    xs = [f32(x) for x in xs] + [f32(rng.uniform(-20, 20)) for _ in range(1500)]
    xs += random_values(rng, 500, -10, 8)
    bad += test_function("sin", "sin(a)", xs, math.sin)
    bad += test_function("cos", "cos(a)", xs, math.cos)

    xs = [0.0, 1.0, 2.0, 4.0, 1e-45, 3.4e38] + random_values(rng, 800, -44, 38, signed=False)
    bad += test_function("sqrt", "sqrt(a)", [f32(x) for x in xs], math.sqrt)

    ys = random_values(rng, 800, -6, 6)
    for name, op, ref in (("add", "+", lambda a, b: a + b), ("sub", "-", lambda a, b: a - b),
                          ("mul", "*", lambda a, b: a * b), ("div", "/", lambda a, b: a / b)):
        lines, wants = [], []
        for a, b in zip(ys[::2], ys[1::2]):
            lines += [f"a = {literal(a)}", f"b = {literal(b)}", f"print a {op} b"]
            wants.append("%g" % f32(ref(a, b)))
        got = run_lines(lines)
        wrong = sum(not close(g, w) for g, w in zip(got, wants))
        print(f"{'ok  ' if not wrong else 'FAIL'} {name}: {len(wants) - wrong}/{len(wants)}")
        bad += wrong

    print("ALL PASSED" if not bad else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
