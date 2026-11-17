"""Run-time support routines, written in RV32IM assembly.

Ripes implements RV32I + M (+ C) but no floating-point extension, so Optima's
floats are done in software. Every float is an IEEE-754 single-precision bit
pattern held in an ordinary integer register:

    bit 31      sign
    bits 30-23  exponent (biased by 127)
    bits 22-0   mantissa (with an implicit leading 1)

Calling convention (the standard RISC-V soft-float one): operands in a0, a1,
result in a0. The arithmetic routines are leaf functions that only touch
a0-a1 and t0-t6. The math routines (sqrt, exp, log, sin, cos) save and
restore the s-registers they use.

Simplifications compared with full IEEE-754: results are rounded to nearest
(ties to even), infinities and NaN are produced and propagated, but denormal
numbers (below about 1.2e-38) are treated as zero.

Only the routines a program actually uses are added to its output.
"""

import struct


def fbits(v):
    """IEEE-754 single-precision bit pattern of a Python float."""
    try:
        return struct.unpack("<I", struct.pack("<f", v))[0]
    except OverflowError:
        return 0xFF800000 if v < 0 else 0x7F800000


def fhex(v):
    return f"0x{fbits(v):08x}"


def _horner(var_reg, coeffs):
    """Assembly evaluating a polynomial in `var_reg` by Horner's rule.

    coeffs are highest power first; the running value is kept in a0.
    """
    lines = [f"    li   a0, {fhex(coeffs[0])}        # {coeffs[0]!r}"]
    for c in coeffs[1:]:
        lines += [
            f"    mv   a1, {var_reg}",
            "    jal  __fmul",
            f"    li   a1, {fhex(c)}        # {c!r}",
            "    jal  __fadd",
        ]
    return "\n".join(lines)


# name -> (routines it calls, assembly text)
ROUTINES = {}


def _routine(name, deps, text):
    ROUTINES[name] = (deps, text.strip("\n"))


# ------------------------------------------------------------------ packing
_routine("__fpack", [], """
# __fpack: round and assemble a float (shared tail of the arithmetic routines).
#   t6 = sign (0 or 0x80000000)   t0 = biased exponent
#   t2 = 24-bit mantissa (bit 23 set)
#   t1 = the bits below the mantissa (bit 31 = one half, the rest = sticky)
__fpack:
    bgez t1, __fpack_done        # less than half: round down
    slli t3, t1, 1
    bnez t3, __fpack_up          # more than half: round up
    andi t3, t2, 1               # exactly half: round to even
    beqz t3, __fpack_done
__fpack_up:
    addi t2, t2, 1
    srli t3, t2, 24
    beqz t3, __fpack_done
    srli t2, t2, 1               # mantissa overflowed into bit 24
    addi t0, t0, 1
__fpack_done:
    blez t0, __fpack_zero        # underflow: flush to zero
    li   t3, 255
    bge  t0, t3, __fpack_inf     # overflow: infinity
    slli t2, t2, 9               # drop the implicit leading 1
    srli t2, t2, 9
    slli t0, t0, 23
    or   a0, t6, t0
    or   a0, a0, t2
    ret
__fpack_zero:
    mv   a0, t6
    ret
__fpack_inf:
    li   a0, 0x7f800000
    or   a0, a0, t6
    ret
__fnan:
    li   a0, 0x7fc00000
    ret
""")

# ----------------------------------------------------------------- add / sub
_routine("__fadd", ["__fpack"], """
# __fsub: a0 = a0 - a1        __fadd: a0 = a0 + a1
__fsub:
    li   t0, 0x80000000
    xor  a1, a1, t0              # a - b = a + (-b)
__fadd:
    slli t0, a0, 1               # magnitudes (sign shifted out)
    slli t1, a1, 1
    bgeu t0, t1, __fadd_ordered
    mv   t2, a0                  # swap so that |a0| >= |a1|
    mv   a0, a1
    mv   a1, t2
    mv   t2, t0
    mv   t0, t1
    mv   t1, t2
__fadd_ordered:
    srli t2, t0, 24              # exponent of a (the larger)
    srli t3, t1, 24              # exponent of b
    li   t4, 255
    beq  t2, t4, __fadd_special  # a is inf or nan
    beqz t3, __fadd_b_zero       # b is zero: the answer is a
    sub  t4, t2, t3              # exponent difference
    sltiu t5, t4, 32
    beqz t5, __fadd_ret          # b is too small to matter
    srli t6, a0, 31              # result takes the sign of a
    slli t6, t6, 31
    li   t5, 0x800000
    slli t0, a0, 9               # mantissa of a, with 3 extra low bits
    srli t0, t0, 9
    or   t0, t0, t5
    slli t0, t0, 3
    slli t1, a1, 9               # mantissa of b, same layout
    srli t1, t1, 9
    or   t1, t1, t5
    slli t1, t1, 3
    srl  t5, t1, t4              # line b up with a
    sll  t3, t5, t4
    beq  t3, t1, __fadd_aligned
    ori  t5, t5, 1               # remember that bits were shifted out (sticky)
__fadd_aligned:
    xor  t3, a0, a1
    bltz t3, __fadd_diff         # different signs: subtract magnitudes
    add  t0, t0, t5
    srli t3, t0, 27
    beqz t3, __fadd_round
    andi t3, t0, 1               # carry out: shift right one place
    srli t0, t0, 1
    or   t0, t0, t3
    addi t2, t2, 1
    j    __fadd_round
__fadd_diff:
    sub  t0, t0, t5
    beqz t0, __fadd_zero         # exact cancellation
__fadd_norm:
    srli t3, t0, 26
    bnez t3, __fadd_round
    slli t0, t0, 1               # shift left until the leading 1 is back in place
    addi t2, t2, -1
    j    __fadd_norm
__fadd_round:
    slli t1, t0, 29              # the 3 extra bits become the rounding bits
    srli t3, t0, 3
    mv   t0, t2
    mv   t2, t3
    j    __fpack
__fadd_zero:
    li   a0, 0
__fadd_ret:
    ret
__fadd_b_zero:
    bnez t2, __fadd_ret
    and  a0, a0, a1              # 0 + 0: negative only if both are -0
    srli a0, a0, 31
    slli a0, a0, 31
    ret
__fadd_special:
    slli t5, a0, 9
    bnez t5, __fnan              # nan + anything
    bne  t3, t4, __fadd_ret      # inf + finite = inf
    xor  t5, a0, a1
    bltz t5, __fnan              # inf - inf
    ret
""")

# ------------------------------------------------------------------ multiply
_routine("__fmul", ["__fpack"], """
# __fmul: a0 = a0 * a1
__fmul:
    xor  t6, a0, a1              # sign of the result
    srli t6, t6, 31
    slli t6, t6, 31
    slli t0, a0, 1               # exponents
    srli t0, t0, 24
    slli t1, a1, 1
    srli t1, t1, 24
    li   t5, 255
    beq  t0, t5, __fmul_special
    beq  t1, t5, __fmul_special
    beqz t0, __fpack_zero        # 0 * x = 0
    beqz t1, __fpack_zero
    li   t4, 0x800000
    slli t2, a0, 9               # 24-bit mantissas
    srli t2, t2, 9
    or   t2, t2, t4
    slli t3, a1, 9
    srli t3, t3, 9
    or   t3, t3, t4
    add  t0, t0, t1              # exponents add
    addi t0, t0, -126
    mul  t1, t2, t3              # 48-bit product: low word
    mulhu t2, t2, t3             #                 high word
    srli t3, t2, 15
    bnez t3, __fmul_norm         # product in [2,4): exponent already right
    slli t2, t2, 1               # product in [1,2): shift left one place
    srli t3, t1, 31
    or   t2, t2, t3
    slli t1, t1, 1
    addi t0, t0, -1
__fmul_norm:
    slli t2, t2, 8               # top 24 bits -> mantissa
    srli t3, t1, 24
    or   t2, t2, t3
    slli t1, t1, 8               # the rest -> rounding bits
    j    __fpack
__fmul_special:
    li   t4, 0xff000000
    slli t2, a0, 1
    slli t3, a1, 1
    bgtu t2, t4, __fnan          # nan * anything
    bgtu t3, t4, __fnan
    beqz t2, __fnan              # 0 * inf
    beqz t3, __fnan
    j    __fpack_inf
""")

# -------------------------------------------------------------------- divide
_routine("__fdiv", ["__fpack"], """
# __fdiv: a0 = a0 / a1
__fdiv:
    xor  t6, a0, a1              # sign of the result
    srli t6, t6, 31
    slli t6, t6, 31
    slli t0, a0, 1               # exponents
    srli t0, t0, 24
    slli t1, a1, 1
    srli t1, t1, 24
    li   t5, 255
    beq  t0, t5, __fdiv_a_special
    beq  t1, t5, __fdiv_b_special
    beqz t1, __fdiv_by_zero
    beqz t0, __fpack_zero        # 0 / x = 0
    li   t4, 0x800000
    slli t2, a0, 9               # 24-bit mantissas
    srli t2, t2, 9
    or   t2, t2, t4
    slli t3, a1, 9
    srli t3, t3, 9
    or   t3, t3, t4
    sub  t0, t0, t1              # exponents subtract
    addi t0, t0, 127
    bgeu t2, t3, __fdiv_go
    slli t2, t2, 1               # make the quotient land in [1,2)
    addi t0, t0, -1
__fdiv_go:
    slli t2, t2, 7               # long division, 8 quotient bits at a time
    divu t1, t2, t3
    remu t2, t2, t3
    slli t2, t2, 8
    divu t4, t2, t3
    remu t2, t2, t3
    slli t1, t1, 8
    or   t1, t1, t4
    slli t2, t2, 8
    divu t4, t2, t3
    remu t2, t2, t3
    slli t1, t1, 8
    or   t1, t1, t4              # t1 = 24-bit quotient
    slli t2, t2, 1
    divu t4, t2, t3              # one more bit, for rounding
    remu t2, t2, t3
    snez t2, t2                  # anything left over is sticky
    slli t4, t4, 31
    or   t4, t4, t2
    mv   t2, t1
    mv   t1, t4
    j    __fpack
__fdiv_a_special:
    slli t2, a0, 9
    bnez t2, __fnan              # nan / x
    beq  t1, t5, __fnan          # inf / inf, inf / nan
    j    __fpack_inf             # inf / x
__fdiv_b_special:
    slli t2, a1, 9
    bnez t2, __fnan              # x / nan
    j    __fpack_zero            # x / inf = 0
__fdiv_by_zero:
    beqz t0, __fnan              # 0 / 0
    j    __fpack_inf             # x / 0 = inf
""")

# ------------------------------------------------------------------- compare
_routine("__flt", [], """
# __flt: a0 = 1 if a0 < a1, else 0 (also 0 if either is nan)
__flt:
    li   t2, 0xff000000
    slli t0, a0, 1
    bgtu t0, t2, __flt_false
    slli t1, a1, 1
    bgtu t1, t2, __flt_false
    srli t0, t0, 1               # turn sign+magnitude into ordinary integers
    bgez a0, __flt_a_ok
    neg  t0, t0
__flt_a_ok:
    srli t1, t1, 1
    bgez a1, __flt_b_ok
    neg  t1, t1
__flt_b_ok:
    slt  a0, t0, t1
    ret
__flt_false:
    li   a0, 0
    ret
""")

# --------------------------------------------------------------- conversions
_routine("__itof", ["__fpack"], """
# __itof: a0 = float(a0), a0 a signed integer
__itof:
    beqz a0, __itof_ret
    srli t6, a0, 31              # sign
    slli t6, t6, 31
    bgez a0, __itof_pos
    neg  a0, a0
__itof_pos:
    li   t0, 158                 # exponent if the leading 1 is at bit 31
__itof_norm:
    bltz a0, __itof_pack
    slli a0, a0, 1
    addi t0, t0, -1
    j    __itof_norm
__itof_pack:
    srli t2, a0, 8
    slli t1, a0, 24
    j    __fpack
__itof_ret:
    ret
""")

_routine("__ftoi", [], """
# __ftoi: a0 = nearest integer to the float in a0
__ftoi:
    slli t0, a0, 1
    srli t0, t0, 24              # exponent
    li   t1, 126
    blt  t0, t1, __ftoi_zero     # |x| < 0.5
    li   t1, 158
    bge  t0, t1, __ftoi_big
    li   t1, 0x800000
    slli t2, a0, 9
    srli t2, t2, 9
    or   t2, t2, t1              # mantissa
    li   t1, 150
    sub  t1, t1, t0              # how far to shift right
    blez t1, __ftoi_left
    addi t3, t1, -1
    li   t4, 1
    sll  t4, t4, t3
    add  t2, t2, t4              # add one half, then truncate
    srl  t2, t2, t1
    j    __ftoi_sign
__ftoi_left:
    neg  t1, t1
    sll  t2, t2, t1
__ftoi_sign:
    bgez a0, __ftoi_done
    neg  t2, t2
__ftoi_done:
    mv   a0, t2
    ret
__ftoi_zero:
    li   a0, 0
    ret
__ftoi_big:
    srli t0, a0, 31              # out of range: clamp
    li   a0, 0x7fffffff
    add  a0, a0, t0
    ret
""")

# ---------------------------------------------------------------------- sqrt
_routine("__fsqrt", ["__fdiv", "__fadd"], """
# __fsqrt: a0 = sqrt(a0), by Newton's iteration  y <- (y + x/y) / 2
__fsqrt:
    slli t0, a0, 1
    beqz t0, __fsqrt_ret         # sqrt(0) = 0
    bltz a0, __fnan              # negative
    srli t0, t0, 24
    beqz t0, __fpack_zero_pos
    li   t1, 255
    beq  t0, t1, __fsqrt_ret     # inf or nan
    addi sp, sp, -16
    sw   ra, 12(sp)
    sw   s1, 8(sp)
    sw   s2, 4(sp)
    sw   s3, 0(sp)
    mv   s1, a0                  # x
    srli s2, a0, 1               # first guess: halve the exponent
    li   t0, 0x1fbd1df5
    add  s2, s2, t0
    li   s3, 3
__fsqrt_loop:
    mv   a0, s1
    mv   a1, s2
    jal  __fdiv                  # x / y
    mv   a1, s2
    jal  __fadd                  # y + x/y
    li   t0, 0x800000
    sub  s2, a0, t0              # divide by 2 (exponent - 1)
    addi s3, s3, -1
    bnez s3, __fsqrt_loop
    mv   a0, s2
    lw   ra, 12(sp)
    lw   s1, 8(sp)
    lw   s2, 4(sp)
    lw   s3, 0(sp)
    addi sp, sp, 16
__fsqrt_ret:
    ret
__fpack_zero_pos:
    li   a0, 0
    ret
""")

# ----------------------------------------------------------------------- exp
_EXP_POLY = [1.9875691500e-4, 1.3981999507e-3, 8.3334519073e-3,
             4.1665795894e-2, 1.6666665459e-1, 5.0000001201e-1]

_routine("__fexp", ["__fmul", "__fadd", "__ftoi", "__itof"], f"""
# __fexp: a0 = exp(a0)
#   x = k*ln2 + r with |r| <= ln2/2, so exp(x) = 2^k * exp(r);
#   exp(r) = 1 + r + r^2 * P(r) with a short polynomial P.
__fexp:
    li   t1, 0xff000000
    slli t0, a0, 1
    bgtu t0, t1, __fnan
    srli t0, t0, 1               # |x|
    bltz a0, __fexp_neg
    li   t1, {fhex(88.72283)}          # exp overflows above 88.72
    bgt  t0, t1, __fexp_inf
    j    __fexp_go
__fexp_neg:
    li   t1, {fhex(87.33654)}          # exp underflows below -87.34
    bgt  t0, t1, __fexp_zero
__fexp_go:
    addi sp, sp, -16
    sw   ra, 12(sp)
    sw   s1, 8(sp)
    sw   s2, 4(sp)
    sw   s3, 0(sp)
    mv   s1, a0                  # x
    li   a1, {fhex(1.4426950408889634)}          # 1/ln2
    jal  __fmul
    jal  __ftoi
    mv   s2, a0                  # k = round(x / ln2)
    jal  __itof
    mv   s3, a0                  # k as a float
    li   a1, {fhex(0.693359375)}          # ln2, high part
    jal  __fmul
    mv   a1, a0
    mv   a0, s1
    jal  __fsub
    mv   s1, a0
    mv   a0, s3
    li   a1, {fhex(-2.12194440e-4)}          # ln2, low part
    jal  __fmul
    mv   a1, a0
    mv   a0, s1
    jal  __fsub
    mv   s1, a0                  # r = x - k*ln2
{_horner("s1", _EXP_POLY)}
    mv   a1, s1
    jal  __fmul
    mv   a1, s1
    jal  __fmul                  # r^2 * P(r)
    mv   a1, s1
    jal  __fadd                  # + r
    li   a1, 0x3f800000
    jal  __fadd                  # + 1
    srli t0, a0, 23              # multiply by 2^k: add k to the exponent
    add  t0, t0, s2
    slli t1, s2, 23
    add  a0, a0, t1
    lw   ra, 12(sp)
    lw   s1, 8(sp)
    lw   s2, 4(sp)
    lw   s3, 0(sp)
    addi sp, sp, 16
    blez t0, __fexp_zero
    li   t1, 255
    bge  t0, t1, __fexp_inf
    ret
__fexp_zero:
    li   a0, 0
    ret
__fexp_inf:
    li   a0, 0x7f800000
    ret
""")

# ----------------------------------------------------------------------- log
_LOG_POLY = [2.0 / 9, 2.0 / 7, 2.0 / 5, 2.0 / 3, 2.0]

_routine("__flog", ["__fmul", "__fadd", "__fdiv", "__itof"], f"""
# __flog: a0 = natural log of a0
#   x = m * 2^e with m in [0.707, 1.414], so log(x) = e*ln2 + log(m);
#   log(m) = 2s(1 + s^2/3 + s^4/5 + ...) with s = (m-1)/(m+1).
__flog:
    slli t0, a0, 1
    beqz t0, __flog_minf         # log(0) = -inf
    bltz a0, __fnan              # log of a negative number
    srli t0, t0, 24
    beqz t0, __flog_minf
    li   t1, 255
    beq  t0, t1, __flog_ret      # inf or nan
    addi sp, sp, -16
    sw   ra, 12(sp)
    sw   s1, 8(sp)
    sw   s2, 4(sp)
    sw   s3, 0(sp)
    addi s2, t0, -127            # e
    slli t1, a0, 9
    srli t1, t1, 9
    li   t2, 0x3f800000
    or   s1, t1, t2              # m in [1, 2)
    li   t2, 0x3fb504f3          # sqrt(2)
    bge  t2, s1, __flog_m_ok
    li   t2, 0x800000
    sub  s1, s1, t2              # m = m / 2
    addi s2, s2, 1
__flog_m_ok:
    mv   a0, s1
    li   a1, 0x3f800000
    jal  __fsub
    mv   s3, a0                  # m - 1
    mv   a0, s1
    li   a1, 0x3f800000
    jal  __fadd                  # m + 1
    mv   a1, a0
    mv   a0, s3
    jal  __fdiv
    mv   s1, a0                  # s
    mv   a1, a0
    jal  __fmul
    mv   s3, a0                  # s^2
{_horner("s3", _LOG_POLY)}
    mv   a1, s1
    jal  __fmul
    mv   s1, a0                  # log(m)
    mv   a0, s2
    jal  __itof
    li   a1, {fhex(0.6931471805599453)}          # ln2
    jal  __fmul
    mv   a1, s1
    jal  __fadd                  # e*ln2 + log(m)
    lw   ra, 12(sp)
    lw   s1, 8(sp)
    lw   s2, 4(sp)
    lw   s3, 0(sp)
    addi sp, sp, 16
__flog_ret:
    ret
__flog_minf:
    li   a0, 0xff800000
    ret
""")

# ----------------------------------------------------------------- sin / cos
_SIN_POLY = [-1.9515295891e-4, 8.3321608736e-3, -1.6666654611e-1]
_COS_POLY = [2.443315711809948e-5, -1.388731625493765e-3, 4.166664568298827e-2, -0.5, 1.0]

_routine("__fsin", ["__fmul", "__fadd", "__ftoi", "__itof"], f"""
# __fsin / __fcos: a0 = sin(a0) / cos(a0)
#   x = k*(pi/2) + r with |r| <= pi/4; the quadrant k picks sin(r) or cos(r)
#   and the sign. cos(x) is sin(x + pi/2), i.e. one quadrant further on.
__fcos:
    li   a2, 1
    j    __ftrig
__fsin:
    li   a2, 0
__ftrig:
    slli t0, a0, 1
    srli t0, t0, 24
    li   t1, 255
    beq  t0, t1, __fnan          # sin/cos of inf or nan
    addi sp, sp, -16
    sw   ra, 12(sp)
    sw   s1, 8(sp)
    sw   s2, 4(sp)
    sw   s3, 0(sp)
    mv   s1, a0                  # x
    mv   s3, a2
    li   a1, {fhex(0.6366197723675814)}          # 2/pi
    jal  __fmul
    jal  __ftoi
    add  s2, a0, s3              # quadrant = k (+1 for cos)
    jal  __itof
    mv   s3, a0                  # k as a float
    li   a1, {fhex(1.5703125)}          # pi/2, high part
    jal  __fmul
    mv   a1, a0
    mv   a0, s1
    jal  __fsub
    mv   s1, a0
    mv   a0, s3
    li   a1, {fhex(4.838267948966e-4)}          # pi/2, low part
    jal  __fmul
    mv   a1, a0
    mv   a0, s1
    jal  __fsub
    mv   s1, a0                  # r
    mv   a1, a0
    jal  __fmul
    mv   s3, a0                  # z = r^2
    andi t0, s2, 1
    bnez t0, __ftrig_cos
{_horner("s3", _SIN_POLY)}
    mv   a1, s3
    jal  __fmul
    mv   a1, s1
    jal  __fmul
    mv   a1, s1
    jal  __fadd                  # sin(r) = r + r*z*S(z)
    j    __ftrig_sign
__ftrig_cos:
{_horner("s3", _COS_POLY)}
__ftrig_sign:
    andi t0, s2, 2
    beqz t0, __ftrig_done
    li   t0, 0x80000000
    xor  a0, a0, t0              # quadrants 2 and 3: negate
__ftrig_done:
    lw   ra, 12(sp)
    lw   s1, 8(sp)
    lw   s2, 4(sp)
    lw   s3, 0(sp)
    addi sp, sp, 16
    ret
""")

# ------------------------------------------------------------------ printing
_routine("__print_point", [], """
# __print_point: print a0[0 .. a1-1] as "(x, y, ...)" using Ripes ecalls
__print_point:
    mv   t0, a0
    mv   t1, a1
    li   t2, 0
    li   a0, 40                  # '('
    li   a7, 11
    ecall
__print_point_loop:
    lw   a0, 0(t0)
    li   a7, 2                   # print_float
    ecall
    addi t0, t0, 4
    addi t2, t2, 1
    bge  t2, t1, __print_point_end
    li   a0, 44                  # ','
    li   a7, 11
    ecall
    li   a0, 32                  # ' '
    ecall
    j    __print_point_loop
__print_point_end:
    li   a0, 41                  # ')'
    li   a7, 11
    ecall
    ret
""")

_routine("__print_str", [], """
# __print_str: print the zero-terminated string at a0, one character at a time
__print_str:
    mv   t0, a0
    li   a7, 11                  # print_char
__print_str_loop:
    lbu  a0, 0(t0)
    beqz a0, __print_str_end
    ecall
    addi t0, t0, 1
    j    __print_str_loop
__print_str_end:
    ret
""")

# aliases: several entry points live inside one routine
ALIASES = {
    "__fsub": "__fadd",
    "__fcos": "__fsin",
    "__fnan": "__fpack",
    "__fpack_zero": "__fpack",
    "__fpack_inf": "__fpack",
}

# routines that jump to labels defined in __fpack without listing it above
for _name in ("__fsqrt", "__fexp", "__flog", "__fsin", "__flt"):
    _deps, _text = ROUTINES[_name]
    if "__fpack" not in _deps and ("__fnan" in _text or "__fpack_" in _text):
        ROUTINES[_name] = (_deps + ["__fpack"], _text)

ORDER = ["__fadd", "__fmul", "__fdiv", "__fpack", "__flt", "__itof", "__ftoi",
         "__fsqrt", "__fexp", "__flog", "__fsin", "__print_point", "__print_str"]


def resolve(used):
    """Names of all routines needed for the set of called labels `used`."""
    needed = set()
    todo = [ALIASES.get(u, u) for u in used]
    while todo:
        name = todo.pop()
        if name in needed:
            continue
        needed.add(name)
        todo += [ALIASES.get(d, d) for d in ROUTINES[name][0]]
    return [n for n in ORDER if n in needed]


def emit(used):
    """Assembly text for the routines needed by `used`."""
    return "\n\n".join(ROUTINES[n][1] for n in resolve(used))
