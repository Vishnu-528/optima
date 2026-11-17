#!/usr/bin/env python3
"""Check the software floating-point routines against exact reference results.

For + - * / the reference is computed with exact fractions and rounded to
single precision (round to nearest, ties to even), so results must match bit
for bit. For sqrt/exp/log/sin/cos the result must be within a few units in
the last place of the true value.

    python tests/test_softfloat.py            run in the built-in simulator
    RIPES=/path/to/Ripes python tests/test_softfloat.py     run in real Ripes
"""

import math
import os
import random
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from optima import runtime, sim  # noqa: E402

NAN = 0x7FC00000
INF = 0x7F800000


def bits_to_float(b):
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def is_nan(b):
    return (b & 0x7FFFFFFF) > INF


def decode(b):
    """bits -> ('nan' | 'inf' | 'zero' | 'num', sign, Fraction). Denormals count as zero."""
    sign = b >> 31
    e = (b >> 23) & 0xFF
    m = b & 0x7FFFFF
    if e == 255:
        return ("nan" if m else "inf"), sign, None
    if e == 0:
        return "zero", sign, Fraction(0)
    v = Fraction(m | 0x800000) * Fraction(2) ** (e - 150)
    return "num", sign, -v if sign else v


def round_f32(q, sign_if_zero=0):
    """Exact Fraction -> float32 bits, nearest-even, underflow flushes to zero."""
    if q == 0:
        return sign_if_zero << 31
    sign = 1 if q < 0 else 0
    q = abs(q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q:
        e -= 1
    scaled = q / Fraction(2) ** (e - 23)          # in [2^23, 2^24)
    m = scaled.numerator // scaled.denominator
    rem = scaled - m
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and m & 1):
        m += 1
        if m == 1 << 24:
            m >>= 1
            e += 1
    be = e + 127
    if be <= 0:
        return sign << 31
    if be >= 255:
        return (sign << 31) | INF
    return (sign << 31) | (be << 23) | (m & 0x7FFFFF)


def ref_binary(op, a, b):
    ka, sa, va = decode(a)
    kb, sb, vb = decode(b)
    if op == "sub":
        b ^= 0x80000000
        sb ^= 1
        vb = -vb if vb is not None else None
        op = "add"
    if ka == "nan" or kb == "nan":
        return NAN
    if op == "add":
        if ka == "inf" or kb == "inf":
            if ka == "inf" and kb == "inf" and sa != sb:
                return NAN
            return ((sa if ka == "inf" else sb) << 31) | INF
        if ka == "zero" and kb == "zero":
            return (sa & sb) << 31
        if kb == "zero":
            return a
        if ka == "zero":
            return b
        s = va + vb
        return round_f32(s) if s != 0 else 0
    sign = sa ^ sb
    if op == "mul":
        if ka == "inf" or kb == "inf":
            return NAN if "zero" in (ka, kb) else (sign << 31) | INF
        if ka == "zero" or kb == "zero":
            return sign << 31
        return round_f32(va * vb, sign)
    if op == "div":
        if ka == "inf":
            return NAN if kb == "inf" else (sign << 31) | INF
        if kb == "inf":
            return sign << 31
        if kb == "zero":
            return NAN if ka == "zero" else (sign << 31) | INF
        if ka == "zero":
            return sign << 31
        return round_f32(va / vb, sign)
    raise AssertionError(op)


# --------------------------------------------------------------- test vectors

def rand_float(rng, emin=1, emax=254):
    return (rng.getrandbits(1) << 31) | (rng.randint(emin, emax) << 23) | rng.getrandbits(23)


SPECIALS = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x40000000, 0x3F000000,
            INF, 0xFF800000, NAN, 0x00800000, 0x7F7FFFFF, 0x3F7FFFFF, 0x3F800001,
            0x40490FDB, 0x3DCCCCCD, 0x4B800000, 0x33800000]


def binary_vectors(rng, count):
    vecs = [(a, b) for a in SPECIALS for b in SPECIALS]
    for _ in range(count):
        kind = rng.randrange(5)
        if kind == 0:                        # anything
            a, b = rand_float(rng), rand_float(rng)
        elif kind == 1:                      # everyday magnitudes
            a, b = rand_float(rng, 110, 140), rand_float(rng, 110, 140)
        elif kind == 2:                      # nearly equal: cancellation
            a = rand_float(rng, 100, 150)
            b = (a ^ rng.getrandbits(1) << 31) + rng.randint(-3, 3)
        elif kind == 3:                      # few mantissa bits: exact ties
            a = (rand_float(rng, 110, 140) >> 12) << 12
            b = (rand_float(rng, 110, 140) >> 16) << 16
        else:                                # extremes: overflow / underflow
            a, b = rand_float(rng, 1, 30), rand_float(rng, 225, 254)
            if rng.getrandbits(1):
                a, b = b, a
        vecs.append((a & 0xFFFFFFFF, b & 0xFFFFFFFF))
    return vecs


# -------------------------------------------------------------------- harness

def harness(routine, vectors):
    words = ", ".join(f"0x{w:08x}" for pair in vectors for w in pair)
    lines = [
        ".data",
        f"tbl: .word {words}",
        ".text",
        "main:",
        "    la   s8, tbl",
        f"    li   s9, {len(vectors)}",
        "next:",
        "    lw   a0, 0(s8)",
        "    lw   a1, 4(s8)",
        f"    jal  {routine}",
        "    li   a7, 34",
        "    ecall",
        "    li   a0, 10",
        "    li   a7, 11",
        "    ecall",
        "    addi s8, s8, 8",
        "    addi s9, s9, -1",
        "    bnez s9, next",
        "    li   a7, 10",
        "    ecall",
        "",
        runtime.emit({routine}),
    ]
    return "\n".join(lines) + "\n"


def execute(asm):
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
        return res.stdout
    finally:
        os.unlink(path)


def run_routine(routine, vectors):
    out = execute(harness(routine, vectors))
    results = [int(line, 16) for line in out.split("\n") if line.startswith("0x")]
    if len(results) != len(vectors):
        raise SystemExit(f"{routine}: expected {len(vectors)} results, got {len(results)}\n{out[:500]}")
    return results


def check_binary(name, routine, vectors):
    results = run_routine(routine, vectors)
    bad = 0
    for (a, b), got in zip(vectors, results):
        want = ref_binary(name, a, b)
        if want is None:
            ok = (got & 0x7FFFFFFF) == 0
        elif is_nan(want):
            ok = is_nan(got)
        else:
            ok = got == want
        if not ok:
            bad += 1
            if bad <= 5:
                print(f"  {name}(0x{a:08x}, 0x{b:08x}) = 0x{got:08x}, expected 0x{want:08x}"
                      f"   [{bits_to_float(a)!r} , {bits_to_float(b)!r}]")
    print(f"{name:5s}: {len(vectors) - bad}/{len(vectors)} exact")
    return bad


def ulp_error(got_bits, true_value):
    """Distance between a float32 result and the true value, in units in the last place."""
    got = bits_to_float(got_bits)
    if math.isnan(got) or math.isinf(got):
        return float("inf")
    if true_value == 0:
        return 0.0 if got == 0 else float("inf")
    e = math.frexp(true_value)[1] - 1
    ulp = 2.0 ** (max(e, -126) - 23)
    return abs(got - true_value) / ulp


def check_math(name, routine, inputs, func, limit):
    vectors = [(struct.unpack("<I", struct.pack("<f", x))[0], 0) for x in inputs]
    results = run_routine(routine, vectors)
    worst, worst_x, bad = 0.0, None, 0
    for (a, _), got in zip(vectors, results):
        x = bits_to_float(a)
        err = ulp_error(got, func(x))
        if err > worst:
            worst, worst_x = err, x
        if err > limit:
            bad += 1
            if bad <= 5:
                print(f"  {name}({x!r}) = {bits_to_float(got)!r}, expected {func(x)!r} ({err:.1f} ulp)")
    print(f"{name:5s}: {len(vectors) - bad}/{len(vectors)} within {limit} ulp "
          f"(worst {worst:.2f} ulp at {worst_x!r})")
    return bad


def check_special(name, routine, cases):
    results = run_routine(routine, [(a, 0) for a, _ in cases])
    bad = 0
    for (a, want), got in zip(cases, results):
        ok = is_nan(got) if want == "nan" else got == want
        if not ok:
            bad += 1
            print(f"  {name}(0x{a:08x}) = 0x{got:08x}, expected {want if want == 'nan' else hex(want)}")
    print(f"{name:5s}: {len(cases) - bad}/{len(cases)} special cases")
    return bad


def check_compare_and_convert(rng):
    bad = 0
    vecs = [(a, b) for a in SPECIALS for b in SPECIALS]
    vecs += [(rand_float(rng, 100, 150), rand_float(rng, 100, 150)) for _ in range(300)]
    res = run_routine("__flt", vecs)
    for (a, b), got in zip(vecs, res):
        fa, fb = bits_to_float(a), bits_to_float(b)
        want = int(fa < fb)
        if got != want:
            bad += 1
            print(f"  flt({fa!r}, {fb!r}) = {got}, expected {want}")
    print(f"flt  : {len(vecs) - bad}/{len(vecs)}")

    ints = [0, 1, -1, 2, 7, 100, -100, 16777216, 16777217, 16777219, 2147483647, -2147483648,
            123456789, -987654321] + [rng.randint(-2**31, 2**31 - 1) for _ in range(200)]
    res = run_routine("__itof", [(i & 0xFFFFFFFF, 0) for i in ints])
    b2 = 0
    for i, got in zip(ints, res):
        want = round_f32(Fraction(i))
        if got != want:
            b2 += 1
            print(f"  itof({i}) = 0x{got:08x}, expected 0x{want:08x}")
    print(f"itof : {len(ints) - b2}/{len(ints)}")

    floats = [0.0, 0.4, 0.5, 0.6, 1.0, 1.5, 2.5, -0.5, -0.6, -1.5, 127.49, 1e6, -1e6, 3e9, -3e9,
              1e-20] + [rng.uniform(-1e5, 1e5) for _ in range(200)]
    res = run_routine("__ftoi", [(struct.unpack("<I", struct.pack("<f", x))[0], 0) for x in floats])
    b3 = 0
    for x, got in zip(floats, res):
        xf = struct.unpack("<f", struct.pack("<f", x))[0]
        want = math.floor(abs(xf) + 0.5) * (1 if xf >= 0 else -1)   # ties away from zero
        want = max(-2**31, min(2**31 - 1, want))
        got_s = got - (1 << 32) if got & 0x80000000 else got
        if got_s != want:
            b3 += 1
            print(f"  ftoi({xf!r}) = {got_s}, expected {want}")
    print(f"ftoi : {len(floats) - b3}/{len(floats)}")
    return bad + b2 + b3


def main():
    rng = random.Random(12345)
    count = int(os.environ.get("COUNT", "3000"))
    backend = "Ripes" if os.environ.get("RIPES") else "built-in simulator"
    print(f"Soft-float tests on the {backend}")
    bad = 0
    vecs = binary_vectors(rng, count)
    bad += check_binary("add", "__fadd", vecs)
    bad += check_binary("sub", "__fsub", vecs)
    bad += check_binary("mul", "__fmul", vecs)
    bad += check_binary("div", "__fdiv", vecs)
    bad += check_compare_and_convert(rng)

    n = max(200, count // 5)
    f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
    xs = [f32(math.exp(rng.uniform(-80, 80))) for _ in range(n)] + [1.0, 4.0, 9.0, 2.0, 0.25, 1e-30, 1e30]
    bad += check_math("sqrt", "__fsqrt", xs, math.sqrt, 1.0)
    xs = [f32(rng.uniform(-87, 88)) for _ in range(n)] + [f32(rng.uniform(-1, 1)) for _ in range(n)] + [0.0, 1.0, -1.0]
    bad += check_math("exp", "__fexp", xs, math.exp, 2.0)
    xs = [f32(math.exp(rng.uniform(-80, 80))) for _ in range(n)] + [f32(rng.uniform(0.5, 2)) for _ in range(n)] + [1.0, 2.0, 10.0]
    bad += check_math("log", "__flog", xs, math.log, 2.0)
    xs = [f32(rng.uniform(-20, 20)) for _ in range(n)] + [f32(rng.uniform(-1, 1)) for _ in range(n)] + [0.0]
    bad += check_math("sin", "__fsin", xs, math.sin, 3.0)
    bad += check_math("cos", "__fcos", xs, math.cos, 3.0)

    minus_inf = 0xFF800000
    bad += check_special("sqrt", "__fsqrt", [(0, 0), (0x80000000, 0x80000000), (0xBF800000, "nan"),
                                              (INF, INF), (NAN, "nan"), (0x40800000, 0x40000000)])
    bad += check_special("exp", "__fexp", [(0, 0x3F800000), (INF, INF), (minus_inf, 0), (NAN, "nan"),
                                            (0x42C80000, INF), (0xC2C80000, 0)])
    bad += check_special("log", "__flog", [(0, minus_inf), (0xBF800000, "nan"), (INF, INF),
                                            (NAN, "nan"), (0x3F800000, 0)])
    bad += check_special("sin", "__fsin", [(0, 0), (INF, "nan"), (NAN, "nan")])
    bad += check_special("cos", "__fcos", [(0, 0x3F800000), (INF, "nan"), (NAN, "nan")])

    print("ALL PASSED" if not bad else f"{bad} FAILURES")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
