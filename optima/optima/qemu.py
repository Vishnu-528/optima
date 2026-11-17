"""QEMU back end: RV32IMFD with hardware floating point, bare metal.

The program runs directly on QEMU's "virt" machine with no operating system:

    0x80000000   start of RAM; QEMU jumps here, so _start is placed here
    0x10000000   serial port (16550 UART): writing a byte prints a character
    0x00100000   "test finisher" device: writing 0x5555 powers the machine off

Differences from the Ripes back end (codegen.py), which this one inherits
everything else from:

* A float lives in a float register (fa0 = accumulator, fa1 = second
  operand), and + - * / sqrt are single instructions instead of calls.
* exp, log, sin and cos have no instruction, so short routines are added.
  They compute in double precision internally, which makes the single-
  precision result accurate to the last digit in nearly every case.
* Printing writes characters to the serial port; turning a number into text
  is done by the __print_float / __print_int routines below.
* The program starts at _start, which sets up the stack and switches the
  floating-point unit on, and ends by powering the machine off.

Building needs an assembler and linker (LLVM's clang and ld.lld); see
build_elf() and run_elf() at the bottom of this file.
"""

import math
import os
import shutil
import subprocess
import tempfile

from . import ir
from .codegen import CodeGen
from .runtime import fbits

UART = 0x10000000
FINISHER = 0x00100000
RAM_BASE = 0x80000000
STACK_BYTES = 1 << 20

MARCH = "rv32imafd_zicsr"
MABI = "ilp32d"

LINKER_SCRIPT = f"""\
/* Memory layout for QEMU's virt machine: everything goes into RAM, code first. */
ENTRY(_start)
SECTIONS {{
  . = 0x{RAM_BASE:08x};
  .text   : {{ *(.text*) }}
  .rodata : {{ *(.rodata*) *(.srodata*) }}
  .data   : {{ *(.data*) *(.sdata*) }}
  .bss    : {{ *(.bss*) *(.sbss*) *(COMMON) }}
}}
"""


class QemuCodeGen(CodeGen):
    TARGET_NOTE = ("QEMU virt machine, RV32IMFD, hardware floating point "
                   "(build with: optc.py --target qemu)")
    RUNTIME_NOTE = "printing over the serial port, exp/log/sin/cos"
    FA, FB = "fa0", "fa1"
    FLR, FTOL, FBETA, FT = "fs0", "fs1", "fs2", "fs3"

    # ------------------------------------------------------------- floats
    def fmv(self, dst, src, comment=None):
        self.ins(f"fmv.s {dst}, {src}", comment)

    def fconst(self, reg, e):
        bits = fbits(e.args[0])
        if bits == 0:
            self.ins(f"fmv.w.x {reg}, zero", ir.show(e))
        else:
            self.ins(f"li t0, 0x{bits:08x}")
            self.ins(f"fmv.w.x {reg}, t0", ir.show(e))

    def fop(self, op):
        self.ins(f"f{op}.s fa0, fa0, fa1")

    def flt(self):
        self.ins("flt.s a0, fa0, fa1")

    def fneg(self):
        self.ins("fneg.s fa0, fa0")

    def ffn(self, name):
        if name == "sqrt":
            self.ins("fsqrt.s fa0, fa0")
        else:
            self.call(f"__f{name}")

    def itof(self):
        self.ins("fcvt.s.w fa0, a0")

    def store_and_diff(self, off):
        self.ins(f"fsw fa0, {off}(s2)")
        self.ins(f"lw t1, {off}(s2)")
        self.ins("sub t0, t1, t0")

    # ------------------------------------------------------------ printing
    def print_char(self, ch):
        self.ins(f"li a0, {ord(ch)}")
        self.call("__putc", f"print {ch!r}")

    def print_float(self):
        self.call("__print_float", "print float")

    def print_int(self):
        self.call("__print_int", "print integer")

    # ------------------------------------------------------ start and end
    def emit_entry(self):
        self.out.append("    .globl _start")
        self.label("_start")
        self.ins("la sp, __stack_top", "set up the stack")
        self.ins("la t0, __trap")
        self.ins("csrw mtvec, t0", "where to go if the processor faults")
        self.ins("li t0, 0x6000")
        self.ins("csrs mstatus, t0", "switch the floating-point unit on")
        self.ins("csrw fcsr, zero", "round to nearest, clear the flags")
        self.label("main")

    def emit_exit(self):
        self.ins(f"li t0, 0x{FINISHER:x}")
        self.ins("li t1, 0x5555")
        self.ins("sw t1, 0(t0)", "power off: tells QEMU to exit")
        self.label("__halt")
        self.ins("j __halt", "never reached")
        self.out.append("")
        self.out.append("# a processor fault (should never happen) ends the run with exit code 1")
        self.out.append("    .balign 4")
        self.label("__trap")
        self.ins(f"li t0, 0x{FINISHER:x}")
        self.ins("li t1, 0x13333")
        self.ins("sw t1, 0(t0)")
        self.ins("j __trap")

    def runtime_text(self):
        return emit(self.used_runtime)


# ===================================================================== runtime
# Register use: the routines below change only a0-a1, t0-t4, ft0-ft4 and fa0
# (the temporaries); they save and restore any s-register they use and never
# touch fs0-fs3, where the solver loops keep lr, tol, beta and a scratch value.

ROUTINES = {}       # name -> (dependencies, code, read-only data)


def _routine(name, deps, text, rodata=""):
    ROUTINES[name] = (deps, text.strip("\n"), rodata.strip("\n"))


def _doubles(label, values):
    lines = [f"{label}:"]
    lines += [f"    .double {v!r}" for v in values]
    return "\n".join(lines)


def _horner(acc, x, tmp, base, first, count):
    """acc = polynomial in x; coefficients are doubles at first, first+8, ...
    from the address in `base`, highest power first."""
    lines = [f"    fld  {acc}, {first}({base})"]
    for i in range(1, count):
        lines.append(f"    fld  {tmp}, {first + 8 * i}({base})")
        lines.append(f"    fmadd.d {acc}, {acc}, {x}, {tmp}")
    return "\n".join(lines)


_routine("__putc", [], f"""
# __putc: send the character in a0 to the serial port (changes t0, t1)
__putc:
    li   t0, 0x{UART:x}
__putc_wait:
    lbu  t1, 5(t0)               # line status register
    andi t1, t1, 0x20            # bit 5: ready for another character?
    beqz t1, __putc_wait
    sb   a0, 0(t0)
    ret
""")

_routine("__print_str", ["__putc"], """
# __print_str: print the zero-terminated string at a0
__print_str:
    addi sp, sp, -16
    sw   ra, 12(sp)
    sw   s1, 8(sp)
    mv   s1, a0
__print_str_loop:
    lbu  a0, 0(s1)
    beqz a0, __print_str_end
    jal  __putc
    addi s1, s1, 1
    j    __print_str_loop
__print_str_end:
    lw   ra, 12(sp)
    lw   s1, 8(sp)
    addi sp, sp, 16
    ret
""")

_routine("__print_int", ["__putc"], """
# __print_int: print the integer in a0 in decimal
__print_int:
    addi sp, sp, -32
    sw   ra, 28(sp)
    sw   s1, 24(sp)
    sw   s2, 20(sp)
    mv   s1, a0
    bgez s1, __print_int_digits
    li   a0, 45                  # '-'
    jal  __putc
    neg  s1, s1
__print_int_digits:
    addi s2, sp, 16              # digits are written backwards, ending here
    li   t2, 10
__print_int_next:
    remu t3, s1, t2              # last digit
    divu s1, s1, t2
    addi t3, t3, 48              # to a character
    addi s2, s2, -1
    sb   t3, 0(s2)
    bnez s1, __print_int_next
__print_int_out:
    lbu  a0, 0(s2)
    jal  __putc
    addi s2, s2, 1
    addi t3, sp, 16
    bltu s2, t3, __print_int_out
    lw   ra, 28(sp)
    lw   s1, 24(sp)
    lw   s2, 20(sp)
    addi sp, sp, 32
    ret
""")

_routine("__print_float", ["__putc", "__print_str"], """
# __print_float: print the float in fa0 the way C's "%g" does: 6 significant
# digits, trailing zeros dropped, and the form 1.5e+10 for very large or very
# small numbers.
#   s1 = bit pattern of |v|     s2 = decimal exponent e (v = d.ddddd * 10^e)
#   s3 = how many digits to print     s4 = loop counter
#   The 6 digit characters are kept at sp+0 .. sp+5.
__print_float:
    addi sp, sp, -48
    sw   ra, 44(sp)
    sw   s1, 40(sp)
    sw   s2, 36(sp)
    sw   s3, 32(sp)
    sw   s4, 28(sp)
    fmv.x.w s2, fa0
    slli s1, s2, 1
    srli s1, s1, 1               # |v|
    li   t0, 0x7f800000
    bleu s1, t0, __pf_number
    la   a0, __str_nan
    jal  __print_str
    j    __pf_done
__pf_number:
    bgez s2, __pf_positive
    li   a0, 45                  # '-'
    jal  __putc
__pf_positive:
    bnez s1, __pf_nonzero
    li   a0, 48                  # '0'
    jal  __putc
    j    __pf_done
__pf_nonzero:
    li   t0, 0x7f800000
    bne  s1, t0, __pf_finite
    la   a0, __str_inf
    jal  __print_str
    j    __pf_done
__pf_finite:
    fmv.w.x ft0, s1
    fcvt.d.s ft0, ft0            # d = |v| as a double
    la   t0, __pf_consts
    fld  ft1, 0(t0)              # 10.0
    fld  ft2, 8(t0)              # 1.0
    li   s2, 0
__pf_down:                       # while d >= 10: d /= 10, e += 1
    flt.d t1, ft0, ft1
    bnez t1, __pf_up
    fdiv.d ft0, ft0, ft1
    addi s2, s2, 1
    j    __pf_down
__pf_up:                         # while d < 1: d *= 10, e -= 1
    flt.d t1, ft0, ft2
    beqz t1, __pf_scaled
    fmul.d ft0, ft0, ft1
    addi s2, s2, -1
    j    __pf_up
__pf_scaled:                     # now 1 <= d < 10
    fld  ft1, 16(t0)             # 100000.0
    fmul.d ft0, ft0, ft1
    fcvt.w.d s3, ft0, rne        # the 6 digits as an integer 100000..1000000
    li   t1, 1000000
    blt  s3, t1, __pf_digits
    li   s3, 100000              # rounding went up to 10.0000: use 1.00000
    addi s2, s2, 1
__pf_digits:
    li   t2, 10
    addi t3, sp, 6
__pf_digit:
    remu t4, s3, t2
    divu s3, s3, t2
    addi t4, t4, 48
    addi t3, t3, -1
    sb   t4, 0(t3)
    bne  t3, sp, __pf_digit
    li   s3, 6                   # drop trailing zeros (keep at least 1 digit)
__pf_strip:
    li   t1, 1
    ble  s3, t1, __pf_stripped
    add  t3, sp, s3
    lbu  t4, -1(t3)
    li   t1, 48
    bne  t4, t1, __pf_stripped
    addi s3, s3, -1
    j    __pf_strip
__pf_stripped:
    li   t1, -4
    blt  s2, t1, __pf_exp        # smaller than 0.0001: exponent form
    li   t1, 6
    bge  s2, t1, __pf_exp        # 1000000 or more: exponent form
    bltz s2, __pf_small

    # ---- 1 <= |v| < 1000000: digits with the point after digit e
    addi t1, s2, 1
    bge  s3, t1, __pf_fixed
    mv   s3, t1                  # print at least e+1 digits
__pf_fixed:
    li   s4, 0
__pf_fixed_loop:
    addi t1, s2, 1
    bne  s4, t1, __pf_fixed_digit
    li   a0, 46                  # '.'
    jal  __putc
__pf_fixed_digit:
    add  t1, sp, s4
    lbu  a0, 0(t1)
    jal  __putc
    addi s4, s4, 1
    blt  s4, s3, __pf_fixed_loop
    j    __pf_done

    # ---- 0.0001 <= |v| < 1: "0." then -e-1 zeros, then the digits
__pf_small:
    li   a0, 48
    jal  __putc
    li   a0, 46
    jal  __putc
    addi s4, s2, 1
__pf_small_zeros:
    bgez s4, __pf_small_digits
    li   a0, 48
    jal  __putc
    addi s4, s4, 1
    j    __pf_small_zeros
__pf_small_digits:
    li   s4, 0
__pf_small_loop:
    add  t1, sp, s4
    lbu  a0, 0(t1)
    jal  __putc
    addi s4, s4, 1
    blt  s4, s3, __pf_small_loop
    j    __pf_done

    # ---- exponent form: d.ddddde+XX
__pf_exp:
    lbu  a0, 0(sp)
    jal  __putc
    li   t1, 1
    ble  s3, t1, __pf_exp_e
    li   a0, 46                  # '.'
    jal  __putc
    li   s4, 1
__pf_exp_loop:
    add  t1, sp, s4
    lbu  a0, 0(t1)
    jal  __putc
    addi s4, s4, 1
    blt  s4, s3, __pf_exp_loop
__pf_exp_e:
    li   a0, 101                 # 'e'
    jal  __putc
    li   a0, 43                  # '+'
    bgez s2, __pf_exp_sign
    li   a0, 45                  # '-'
    neg  s2, s2
__pf_exp_sign:
    jal  __putc
    li   t2, 10
    divu a0, s2, t2
    addi a0, a0, 48
    jal  __putc
    li   t2, 10
    remu a0, s2, t2
    addi a0, a0, 48
    jal  __putc
__pf_done:
    lw   ra, 44(sp)
    lw   s1, 40(sp)
    lw   s2, 36(sp)
    lw   s3, 32(sp)
    lw   s4, 28(sp)
    addi sp, sp, 48
    ret
""", """
__pf_consts:
    .double 10.0
    .double 1.0
    .double 100000.0
__str_inf: .string "inf"
__str_nan: .string "nan"
""")

_routine("__print_point", ["__putc", "__print_float"], """
# __print_point: print the a1 floats at a0 as "(x, y, ...)"
__print_point:
    addi sp, sp, -16
    sw   ra, 12(sp)
    sw   s1, 8(sp)
    sw   s2, 4(sp)
    mv   s1, a0
    mv   s2, a1
    li   a0, 40                  # '('
    jal  __putc
__print_point_loop:
    flw  fa0, 0(s1)
    jal  __print_float
    addi s1, s1, 4
    addi s2, s2, -1
    blez s2, __print_point_end
    li   a0, 44                  # ','
    jal  __putc
    li   a0, 32                  # ' '
    jal  __putc
    j    __print_point_loop
__print_point_end:
    li   a0, 41                  # ')'
    jal  __putc
    lw   ra, 12(sp)
    lw   s1, 8(sp)
    lw   s2, 4(sp)
    addi sp, sp, 16
    ret
""")

# ----------------------------------------------------------------------- exp
_EXP_COEFFS = [1.0 / math.factorial(n) for n in range(11, -1, -1)]

_routine("__fexp", [], f"""
# __fexp: fa0 = exp(fa0)
#   x = k*ln2 + r with |r| <= ln2/2, so exp(x) = 2^k * exp(r), and exp(r) is
#   the series 1 + r + r^2/2! + ... + r^11/11!, worked out in double precision.
__fexp:
    feq.s t0, fa0, fa0
    beqz t0, __fexp_ret          # exp(nan) = nan
    fcvt.d.s ft0, fa0
    la   t0, __exp_consts
    fld  ft1, 0(t0)
    fmin.d ft0, ft0, ft1         # beyond +-200 the answer is inf or 0 anyway
    fld  ft1, 8(t0)
    fmax.d ft0, ft0, ft1
    fld  ft1, 16(t0)             # 1/ln2
    fmul.d ft2, ft0, ft1
    fcvt.w.d t1, ft2, rne        # k = round(x / ln2)
    fcvt.d.w ft2, t1
    fld  ft1, 24(t0)             # ln2
    fnmsub.d ft0, ft2, ft1, ft0  # r = x - k*ln2
{_horner("ft3", "ft0", "ft1", "t0", 32, len(_EXP_COEFFS))}
    addi t1, t1, 1023            # build the double 2^k from its bit pattern
    slli t1, t1, 20
    addi sp, sp, -16
    sw   zero, 0(sp)
    sw   t1, 4(sp)
    fld  ft1, 0(sp)
    addi sp, sp, 16
    fmul.d ft3, ft3, ft1
    fcvt.s.d fa0, ft3            # back to single precision (gives inf or 0 if out of range)
__fexp_ret:
    ret
""", _doubles("__exp_consts", [200.0, -200.0, 1.0 / math.log(2.0), math.log(2.0)] + _EXP_COEFFS))

# ----------------------------------------------------------------------- log
_LOG_COEFFS = [2.0 / n for n in range(17, 0, -2)]

_routine("__flog", [], f"""
# __flog: fa0 = natural logarithm of fa0
#   x = 2^e * m with m between sqrt(1/2) and sqrt(2), so log(x) = e*ln2 + log(m);
#   with s = (m-1)/(m+1),  log(m) = 2*(s + s^3/3 + s^5/5 + ... + s^17/17).
__flog:
    fmv.x.w t0, fa0
    slli t1, t0, 1
    beqz t1, __flog_zero         # log(0) = -inf
    bltz t0, __flog_nan          # log of a negative number
    li   t1, 0x7f800000
    bgeu t0, t1, __flog_ret      # log(inf) = inf, log(nan) = nan
    fcvt.d.s ft0, fa0
    addi sp, sp, -16
    fsd  ft0, 0(sp)              # take the double apart through memory
    lw   t0, 4(sp)               # upper half: sign, exponent, top of mantissa
    srli t1, t0, 20
    addi t1, t1, -1023           # e
    slli t0, t0, 12
    srli t0, t0, 12
    li   t2, 0x3ff00000          # exponent field for 2^0
    or   t0, t0, t2
    sw   t0, 4(sp)
    fld  ft0, 0(sp)              # m, between 1 and 2
    addi sp, sp, 16
    la   t0, __log_consts
    fld  ft1, 0(t0)              # sqrt(2)
    flt.d t2, ft1, ft0
    beqz t2, __flog_series
    fld  ft1, 8(t0)              # m > sqrt(2): halve it and count one more power of 2
    fmul.d ft0, ft0, ft1
    addi t1, t1, 1
__flog_series:
    fld  ft1, 16(t0)             # 1.0
    fsub.d ft2, ft0, ft1
    fadd.d ft3, ft0, ft1
    fdiv.d ft2, ft2, ft3         # s = (m-1)/(m+1)
    fmul.d ft3, ft2, ft2         # s^2
{_horner("ft4", "ft3", "ft1", "t0", 32, len(_LOG_COEFFS))}
    fmul.d ft4, ft4, ft2         # log(m)
    fcvt.d.w ft1, t1
    fld  ft0, 24(t0)             # ln2
    fmadd.d ft4, ft1, ft0, ft4   # e*ln2 + log(m)
    fcvt.s.d fa0, ft4
__flog_ret:
    ret
__flog_zero:
    li   t0, 0xff800000          # -inf
    fmv.w.x fa0, t0
    ret
__flog_nan:
    li   t0, 0x7fc00000          # nan
    fmv.w.x fa0, t0
    ret
""", _doubles("__log_consts", [math.sqrt(2.0), 0.5, 1.0, math.log(2.0)] + _LOG_COEFFS))

# ------------------------------------------------------------------- sin, cos
_SIN_COEFFS = [(-1.0) ** k / math.factorial(2 * k + 1) for k in range(6, -1, -1)]
_COS_COEFFS = [(-1.0) ** k / math.factorial(2 * k) for k in range(7, -1, -1)]
_PIO2_HI = math.pi / 2
_PIO2_LO = 6.123233995736766e-17        # pi/2 minus its double approximation
TRIG_LIMIT = 1.0e9

_routine("__fsin", [], f"""
# __fsin / __fcos: fa0 = sin(fa0) / cos(fa0)
#   x = k*(pi/2) + r with |r| <= pi/4; the quadrant k picks the sin or the cos
#   series for r, and the sign. cos(x) is sin(x + pi/2): one quadrant further.
#   Arguments of 1e9 or more (where neighbouring floats are more than a full
#   turn apart) give nan.
__fcos:
    li   t3, 1
    j    __ftrig
__fsin:
    li   t3, 0
__ftrig:
    fmv.x.w t0, fa0
    slli t0, t0, 1
    srli t0, t0, 1               # |x|
    li   t1, 0x{fbits(TRIG_LIMIT):08x}          # 1e9
    bgeu t0, t1, __ftrig_nan     # too large, inf or nan
    fcvt.d.s ft0, fa0
    la   t0, __trig_consts
    fld  ft1, 0(t0)              # 2/pi
    fmul.d ft2, ft0, ft1
    fcvt.w.d t1, ft2, rne        # k = round(x / (pi/2))
    fcvt.d.w ft2, t1
    fld  ft1, 8(t0)              # pi/2
    fnmsub.d ft0, ft2, ft1, ft0
    fld  ft1, 16(t0)             # the part of pi/2 that does not fit in a double
    fnmsub.d ft0, ft2, ft1, ft0  # r = x - k*(pi/2)
    add  t1, t1, t3              # quadrant
    fmul.d ft3, ft0, ft0         # r^2
    andi t2, t1, 1
    bnez t2, __ftrig_cos
{_horner("ft4", "ft3", "ft1", "t0", 24, len(_SIN_COEFFS))}
    fmul.d ft4, ft4, ft0         # sin(r) = r - r^3/3! + r^5/5! - ...
    j    __ftrig_sign
__ftrig_cos:
{_horner("ft4", "ft3", "ft1", "t0", 24 + 8 * len(_SIN_COEFFS), len(_COS_COEFFS))}
                                 # cos(r) = 1 - r^2/2! + r^4/4! - ...
__ftrig_sign:
    andi t2, t1, 2
    beqz t2, __ftrig_done
    fneg.d ft4, ft4              # quadrants 2 and 3: negate
__ftrig_done:
    fcvt.s.d fa0, ft4
    ret
__ftrig_nan:
    li   t0, 0x7fc00000
    fmv.w.x fa0, t0
    ret
""", _doubles("__trig_consts", [2.0 / math.pi, _PIO2_HI, _PIO2_LO] + _SIN_COEFFS + _COS_COEFFS))

ALIASES = {"__fcos": "__fsin"}
ORDER = ["__putc", "__print_str", "__print_int", "__print_float", "__print_point",
         "__fexp", "__flog", "__fsin"]


def resolve(used):
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
    """Runtime routines needed by `used`, their constants, and the stack."""
    names = resolve(used)
    parts = [ROUTINES[n][1] for n in names]
    rodata = [ROUTINES[n][2] for n in names if ROUTINES[n][2]]
    if rodata:
        parts.append("# constants used by the routines above\n"
                     ".section .rodata\n.balign 8\n" + "\n".join(rodata))
    parts.append(f"# the stack: {STACK_BYTES // 1024} KiB, growing down from __stack_top\n"
                 f".bss\n.balign 16\n    .space {STACK_BYTES}\n__stack_top:")
    return "\n\n".join(parts)


# ============================================================ build and run
class ToolError(Exception):
    """A required external program is missing or failed."""


_WINDOWS_DIRS = {
    "clang": [r"C:\Program Files\LLVM\bin"],
    "ld.lld": [r"C:\Program Files\LLVM\bin"],
    "qemu-system-riscv32": [r"C:\Program Files\qemu", r"C:\msys64\ucrt64\bin",
                            r"C:\msys64\mingw64\bin"],
}
_ENV_NAMES = {"clang": "OPTIMA_CLANG", "ld.lld": "OPTIMA_LLD",
              "qemu-system-riscv32": "OPTIMA_QEMU"}
_INSTALL_HINT = {
    "clang": "install LLVM (Windows: winget install LLVM.LLVM; Ubuntu: sudo apt install clang lld)",
    "ld.lld": "install LLVM (Windows: winget install LLVM.LLVM; Ubuntu: sudo apt install clang lld)",
    "qemu-system-riscv32": "install QEMU (Windows: https://www.qemu.org/download/#windows; "
                           "Ubuntu: sudo apt install qemu-system-misc)",
}


def find_tool(name):
    """Full path of clang, ld.lld or qemu-system-riscv32.

    Looks at the OPTIMA_CLANG / OPTIMA_LLD / OPTIMA_QEMU environment
    variable, then PATH, then the usual Windows install folders.
    """
    env = _ENV_NAMES[name]
    path = os.environ.get(env)
    if path:
        if os.path.isfile(path):
            return path
        raise ToolError(f"{env} is set to {path!r}, which does not exist")
    path = shutil.which(name)
    if path:
        return path
    if os.name == "nt":
        for folder in _WINDOWS_DIRS[name]:
            candidate = os.path.join(folder, name + ".exe")
            if os.path.isfile(candidate):
                return candidate
    raise ToolError(f"cannot find {name}: {_INSTALL_HINT[name]}, "
                    f"or set {env} to its full path")


def _tool(args, what):
    try:
        res = subprocess.run(args, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ToolError(f"{what} could not be run: {e}")
    if res.returncode != 0:
        detail = (res.stderr or res.stdout).strip()
        raise ToolError(f"{what} failed:\n{detail}")


def build_elf(asm, out_path):
    """Assemble and link the assembly text `asm` into the ELF file out_path."""
    clang, lld = find_tool("clang"), find_tool("ld.lld")
    with tempfile.TemporaryDirectory(prefix="optima-") as tmp:
        src = os.path.join(tmp, "prog.s")
        obj = os.path.join(tmp, "prog.o")
        script = os.path.join(tmp, "link.ld")
        with open(src, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(asm)
        with open(script, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(LINKER_SCRIPT)
        _tool([clang, "--target=riscv32-unknown-elf", f"-march={MARCH}", f"-mabi={MABI}",
               "-c", src, "-o", obj], "the assembler (clang)")
        _tool([lld, "-m", "elf32lriscv", "-T", script, obj, "-o", out_path],
              "the linker (ld.lld)")


def run_elf(elf_path, timeout=60):
    """Run an ELF file on QEMU's virt machine; returns (output, exit code)."""
    qemu = find_tool("qemu-system-riscv32")
    args = [qemu, "-M", "virt", "-cpu", "rv32", "-m", "64M", "-bios", "none",
            "-display", "none", "-monitor", "none", "-serial", "stdio",
            "-kernel", elf_path]
    try:
        res = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True,
                             timeout=timeout)
    except subprocess.TimeoutExpired as e:
        partial = (e.stdout or b"").decode("utf-8", "replace")
        raise ToolError(f"the program was still running after {timeout} seconds and was stopped"
                        + (f"; output so far:\n{partial}" if partial else ""))
    except OSError as e:
        raise ToolError(f"QEMU could not be run: {e}")
    out = res.stdout.decode("utf-8", "replace").replace("\r\n", "\n")
    if res.returncode != 0 and not out:
        raise ToolError(f"QEMU failed:\n{res.stderr.decode('utf-8', 'replace').strip()}")
    return out, res.returncode


def run(asm, timeout=60):
    """Build and run assembly text; returns the program's output."""
    with tempfile.TemporaryDirectory(prefix="optima-") as tmp:
        elf = os.path.join(tmp, "prog.elf")
        build_elf(asm, elf)
        out, code = run_elf(elf, timeout)
    if code != 0:
        raise ToolError(f"the program stopped with a processor fault (exit code {code})"
                        + (f"; output so far:\n{out}" if out else ""))
    return out
