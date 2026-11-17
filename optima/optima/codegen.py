"""Code generator: checked program -> RV32IM assembly for Ripes.

Conventions in the generated code
---------------------------------
* A float is a 32-bit IEEE-754 pattern in an integer register; arithmetic is
  done by calling the soft-float routines in runtime.py (a0 op a1 -> a0).
* Every expression leaves its value in a0.
* s0 is the frame pointer. Temporaries live at negative offsets from s0, so
  sp never moves inside a function body.
* A user function  f(x, y) = ...  becomes three possible routines:
      fn_f    a0 = pointer to the arguments         -> a0 = f(x, y)
      grad_f  a0 = pointer to the arguments, a1 = pointer to n output words
      hess_f  same, with n*n output words (row by row)
  Inside them s1 holds the argument pointer and s2 the output pointer.
* Global variables are words in .data named g_<name>.
* Top-level statements run in order, starting at the first instruction, and
  the program ends with the Ripes exit ecall.
* Comments of the form "# ---- line N: ..." say which source line the code
  below them came from; any other "# ---- ..." comment starts code that
  belongs to no source line. The patched Ripes reads these to highlight the
  Optima line being executed.
"""

from . import ir, runtime
from .ir import E
from .runtime import fbits

LEAF_KINDS = ("const", "param", "global")
BIN_ROUTINE = {"add": "__fadd", "sub": "__fsub", "mul": "__fmul", "div": "__fdiv"}
FN_ROUTINE = {"sqrt": "__fsqrt", "exp": "__fexp", "log": "__flog",
              "sin": "__fsin", "cos": "__fcos"}

# Ripes ecall numbers
PRINT_INT, PRINT_FLOAT, PRINT_STRING, EXIT, PRINT_CHAR = 1, 2, 4, 10, 11


class Frame:
    """Bump allocator for words below the frame pointer s0."""

    def __init__(self, saved_words):
        self.used = saved_words
        self.peak = saved_words

    def alloc(self, n=1):
        """Reserve n words; returns the s0-relative offset of word 0."""
        self.used += n
        self.peak = max(self.peak, self.used)
        return -4 * self.used

    def release(self, n=1):
        self.used -= n

    @property
    def size(self):
        return (4 * self.peak + 15) // 16 * 16


def _children(e):
    k = e.kind
    if k in BIN_ROUTINE:
        return e.args
    if k == "neg":
        return (e.args[0],)
    if k == "fn":
        return (e.args[1],)
    return ()


def shared_subexpressions(roots):
    """Non-leaf nodes that are used more than once across `roots`."""
    refs = {}
    seen = set()

    def visit(e):
        if e in seen:
            return
        seen.add(e)
        for c in dict.fromkeys(_children(e)):
            refs[c] = refs.get(c, 0) + 1
            visit(c)

    for r in roots:
        refs[r] = refs.get(r, 0) + 1
        visit(r)
    return [e for e, count in refs.items() if count > 1 and e.kind not in LEAF_KINDS]


def _clip(text, limit=90):
    return text if len(text) <= limit else text[:limit - 3] + "..."


class CodeGen:
    def __init__(self, prog, source="", filename="<input>"):
        self.prog = prog
        self.src_lines = source.splitlines()
        self.filename = filename
        self.out = []            # lines of the routine being generated
        self.frame = None
        self.slots = {}          # shared subexpression -> frame offset
        self.done = set()        # shared subexpressions already computed
        self.used_runtime = set()
        self.strings = []        # string literals for print
        self.label_id = 0

    # ------------------------------------------------------------ emitting
    def ins(self, text, comment=None):
        op, _, rest = text.partition(" ")
        pad = f"{op:<5}" if len(op) < 5 or not rest else op + " "
        line = f"    {pad}{rest.strip()}".rstrip()
        if comment:
            line = f"{line:<32}# {comment}"
        self.out.append(line)

    def label(self, name):
        self.out.append(f"{name}:")

    def comment(self, text):
        self.out.append(f"    # {text}")

    def call(self, routine, comment=None):
        if routine.startswith("__"):
            self.used_runtime.add(routine)
        self.ins(f"jal {routine}", comment)

    def new_id(self):
        self.label_id += 1
        return self.label_id

    def ecall(self, number, comment=None):
        self.ins(f"li a7, {number}")
        self.ins("ecall", comment)

    def print_char(self, ch):
        self.ins(f"li a0, {ord(ch)}")
        self.ecall(PRINT_CHAR, f"print {ch!r}")

    # ------------------------------------------------- target primitives
    # Everything that depends on how floats are represented and how output
    # is done goes through the methods below; a back end for another target
    # (see qemu.py) overrides them and inherits the rest.
    TARGET_NOTE = "Ripes, RV32IM, software floating point (no F extension needed)"
    RUNTIME_NOTE = "software floating point and printing"
    FA, FB = "a0", "a1"                 # float accumulator and second operand
    FLR, FTOL, FBETA, FT = "s4", "s5", "s6", "s7"   # solver registers

    def ld(self, reg):
        """Instruction that loads a word from memory into `reg`."""
        return "flw" if reg[0] == "f" else "lw"

    def st(self, reg):
        return "fsw" if reg[0] == "f" else "sw"

    def fmv(self, dst, src, comment=None):
        self.ins(f"mv {dst}, {src}", comment)

    def fconst(self, reg, e):
        self.ins(f"li {reg}, 0x{fbits(e.args[0]):08x}", ir.show(e))

    def fop(self, op):
        """FA = FA <op> FB for op in add, sub, mul, div."""
        self.call(BIN_ROUTINE[op])

    def flt(self):
        """a0 = 1 if FA < FB else 0."""
        self.call("__flt")

    def fneg(self):
        self.ins("li t0, 0x80000000")
        self.ins("xor a0, a0, t0", "negate: flip the sign bit")

    def ffn(self, name):
        """FA = sqrt/exp/log/sin/cos(FA)."""
        self.call(FN_ROUTINE[name])

    def itof(self):
        """FA = float(integer in a0)."""
        self.call("__itof")

    def store_and_diff(self, off):
        """x[off] = FA;  t0 = new bit pattern - old bit pattern (old is in t0)."""
        self.ins(f"sw a0, {off}(s2)")
        self.ins("sub t0, a0, t0")

    def print_float(self):
        self.ecall(PRINT_FLOAT, "print float")

    def print_int(self):
        self.ecall(PRINT_INT, "print integer")

    def runtime_text(self):
        return runtime.emit(self.used_runtime)

    def emit_entry(self):
        self.label("main")

    def emit_exit(self):
        self.ins(f"li a7, {EXIT}")
        self.ins("ecall", "exit")
        self.ins("nop", "never reached")

    # --------------------------------------------------- frame slot access
    def slot_load(self, reg, off, comment=None):
        if -2048 <= off <= 2047:
            self.ins(f"{self.ld(reg)} {reg}, {off}(s0)", comment)
        else:
            self.ins(f"li t6, {off}")
            self.ins("add t6, s0, t6")
            self.ins(f"{self.ld(reg)} {reg}, 0(t6)", comment)

    def slot_store(self, reg, off):
        if -2048 <= off <= 2047:
            self.ins(f"{self.st(reg)} {reg}, {off}(s0)")
        else:
            self.ins(f"li t6, {off}")
            self.ins("add t6, s0, t6")
            self.ins(f"{self.st(reg)} {reg}, 0(t6)")

    def slot_addr(self, reg, off):
        if -2048 <= off <= 2047:
            self.ins(f"addi {reg}, s0, {off}")
        else:
            self.ins(f"li {reg}, {off}")
            self.ins(f"add {reg}, s0, {reg}")

    # A "place" is where a point lives: ("frame", offset) or ("global", name)
    def place_addr(self, reg, place):
        if place[0] == "frame":
            self.slot_addr(reg, place[1])
        else:
            self.ins(f"la {reg}, g_{place[1]}")

    def place_store(self, place, i):
        """a0 -> element i of the place."""
        if place[0] == "frame":
            self.slot_store(self.FA, place[1] + 4 * i)
        else:
            self.ins(f"la t0, g_{place[1]}")
            self.ins(f"{self.st(self.FA)} {self.FA}, {4 * i}(t0)")

    def place_load(self, reg, place, i):
        if place[0] == "frame":
            self.slot_load(reg, place[1] + 4 * i)
        else:
            self.ins(f"la t0, g_{place[1]}")
            self.ins(f"{self.ld(reg)} {reg}, {4 * i}(t0)")

    # ---------------------------------------------------- scalar expressions
    def is_simple(self, e):
        """Can e be loaded straight into a register without disturbing a0?"""
        return e.kind in LEAF_KINDS or (e in self.slots and e in self.done)

    def load_simple(self, e, reg):
        k = e.kind
        if e in self.slots and e in self.done:
            self.slot_load(reg, self.slots[e])
        elif k == "const":
            self.fconst(reg, e)
        elif k == "param":
            self.ins(f"{self.ld(reg)} {reg}, {4 * e.args[0]}(s1)", self.names[e.args[0]])
        elif k == "global":
            name, idx = e.args
            self.ins(f"la t0, g_{name}")
            self.ins(f"{self.ld(reg)} {reg}, {4 * max(idx, 0)}(t0)", ir.show(e))
        else:
            raise AssertionError(e.kind)

    def gen(self, e):
        """Emit code that leaves the value of scalar expression e in a0."""
        if self.is_simple(e):
            self.load_simple(e, self.FA)
            return
        k = e.kind
        if k in BIN_ROUTINE:
            a, b = e.args
            if a is b:
                self.gen(a)
                self.fmv(self.FB, self.FA)
            elif self.is_simple(b):
                self.gen(a)
                self.load_simple(b, self.FB)
            elif self.is_simple(a):
                self.gen(b)
                self.fmv(self.FB, self.FA)
                self.load_simple(a, self.FA)
            else:
                self.gen(a)
                if self.is_simple(b):          # b may have been computed by now
                    self.load_simple(b, self.FB)
                else:
                    tmp = self.frame.alloc()
                    self.slot_store(self.FA, tmp)
                    self.gen(b)
                    self.fmv(self.FB, self.FA)
                    self.slot_load(self.FA, tmp)
                    self.frame.release()
            self.fop(k)
        elif k == "neg":
            self.gen(e.args[0])
            self.fneg()
        elif k == "fn":
            self.gen(e.args[1])
            self.ffn(e.args[0])
        elif k == "call":
            fname, args = e.args
            base = self.frame.alloc(len(args))
            for i, a in enumerate(args):
                self.gen(a)
                self.slot_store(self.FA, base + 4 * i)
            self.slot_addr("a0", base)
            self.call(f"fn_{fname}")
            self.frame.release(len(args))
        elif k == "callp":
            fname, p = e.args
            place, words = self.place_of(p)
            self.place_addr("a0", place)
            self.call(f"fn_{fname}")
            self.frame.release(words)
        elif k == "elem":
            p, i = e.args
            place, words = self.place_of(p)
            self.place_load(self.FA, place, i)
            self.frame.release(words)
        elif k == "iters":
            self.ins("la t0, __iters")
            self.ins("lw a0, 0(t0)")
            self.itof()
        elif k == "pow":
            # Only reached in top-level statements (function bodies have their
            # powers expanded up front). Evaluate the base once into a
            # temporary, then multiply it out.
            base, n = e.args
            self.gen(base)
            tmp = self.frame.alloc()
            self.slot_store(self.FA, tmp)
            held = E("tmp", self.new_id())
            self.slots[held] = tmp
            self.done.add(held)
            tree = ir.expand_powers(E("pow", held, n))
            shared = shared_subexpressions([tree])
            for node in shared:
                self.slots[node] = self.frame.alloc()
            self.gen(tree)
            for node in shared + [held]:
                del self.slots[node]
                self.done.discard(node)
            self.frame.release(len(shared) + 1)
        else:
            raise AssertionError(k)
        if e in self.slots:
            self.slot_store(self.FA, self.slots[e])
            self.done.add(e)

    # ------------------------------------------------------ point expressions
    def point_size(self, p):
        k = p.kind
        if k == "plit":
            return len(p.args[0])
        if k == "pvar":
            return p.args[1]
        fname = p.args[0] if k == "pgrad" else p.args[1]
        return len(self.prog.functions[fname].params)

    def place_of(self, p):
        """Make point p available in memory. Returns (place, words to release)."""
        if p.kind == "pvar":
            return ("global", p.args[0]), 0
        n = self.point_size(p)
        place = ("frame", self.frame.alloc(n))
        self.gen_point(p, place)
        return place, n

    def gen_point(self, p, dest):
        """Emit code that writes point expression p into `dest`."""
        k = p.kind
        if k == "plit":
            for i, e in enumerate(p.args[0]):
                self.gen(e)
                self.place_store(dest, i)
        elif k == "pvar":
            if dest == ("global", p.args[0]):
                return
            self.ins(f"la t0, g_{p.args[0]}")
            self.place_addr("t1", dest)
            for i in range(p.args[1]):
                self.ins(f"lw t2, {4 * i}(t0)")
                self.ins(f"sw t2, {4 * i}(t1)")
        elif k == "pgrad":
            fname, at = p.args
            place, words = self.place_of(at)
            self.place_addr("a0", place)
            self.place_addr("a1", dest)
            self.call(f"grad_{fname}")
            self.frame.release(words)
        elif k == "popt":
            self.gen_optimize(p, dest)
        else:
            raise AssertionError(k)

    # ---------------------------------------------------------------- solvers
    def gen_optimize(self, p, dest):
        """minimize / maximize: iterate in place on the point stored at dest.

        Registers during the loop:
            s2 = address of x     s3 = iterations done    s4 = lr
            s5 = tol^2            s6 = beta               s7 = scratch
            s8 = non-zero if x moved during this step

        The loop stops when |gradient| < tol, when the step limit is reached,
        or when a step no longer changes x (it has converged as far as
        single-precision floats allow).
        """
        kind, fname, start, solver, opts = p.args
        opts = dict(opts)
        n = len(self.prog.functions[fname].params)
        k = self.new_id()
        loop, done = f"opt{k}_loop", f"opt{k}_done"
        step_op = "sub" if kind == "minimize" else "add"

        self.comment(f"{kind} {fname} using {solver}: x starts at {_clip(ir.show(start), 50)}")
        self.gen_point(start, dest)

        # evaluate the options first (they are arbitrary expressions)
        opt_slot = {}
        for name in ("lr", "beta", "tol"):
            if name in opts:
                self.gen(opts[name])
                if name == "tol":
                    self.fmv(self.FB, self.FA)
                    self.fop("mul")
                opt_slot[name] = self.frame.alloc()
                self.slot_store(self.FA, opt_slot[name])
        g = self.frame.alloc(n)
        v = self.frame.alloc(n) if solver == "momentum" else None
        h = self.frame.alloc(n * n) if solver == "newton" else None
        scratch = len(opt_slot) + n + (n if v is not None else 0) + (n * n if h is not None else 0)

        self.place_addr("s2", dest)
        for name, reg, what in (("lr", self.FLR, "learning rate"),
                                ("tol", self.FTOL, "tol squared"),
                                ("beta", self.FBETA, "momentum factor")):
            if name in opt_slot:
                self.slot_load(reg, opt_slot[name], f"{reg} = {what}")
        if v is not None:
            for i in range(n):
                self.slot_store("zero", v + 4 * i)
        self.ins("li s3, 0", "s3 = iteration counter")

        self.label(loop)
        self.ins("mv a0, s2")
        self.slot_addr("a1", g)
        self.call(f"grad_{fname}", "g = gradient at x")

        # stop when |g|^2 < tol^2
        for i in range(n):
            self.slot_load(self.FA, g + 4 * i)
            self.fmv(self.FB, self.FA)
            self.fop("mul")
            if i == 0:
                self.fmv(self.FT, self.FA)
            else:
                self.fmv(self.FB, self.FT)
                self.fop("add")
                self.fmv(self.FT, self.FA)
        self.fmv(self.FA, self.FT, "|g|^2")
        self.fmv(self.FB, self.FTOL)
        self.flt()
        self.ins(f"beqz a0, opt{k}_go")
        self.ins(f"j {done}", "gradient is small enough: converged")
        self.label(f"opt{k}_go")
        self.ins(f"li t0, {opts['steps']}")
        self.ins(f"blt s3, t0, opt{k}_step")
        self.ins(f"j {done}", "step limit reached")
        self.label(f"opt{k}_step")
        self.ins("li s8, 0", "s8 = has x moved this step?")

        if solver == "gradient_descent":
            sign = "-" if kind == "minimize" else "+"
            for i in range(n):
                self.comment(f"x[{i}] = x[{i}] {sign} lr * g[{i}]")
                self.slot_load(self.FA, g + 4 * i)
                self.fmv(self.FB, self.FLR)
                self.fop("mul")
                self.fmv(self.FB, self.FA)
                self.ins(f"{self.ld(self.FA)} {self.FA}, {4 * i}(s2)")
                self.fop(step_op)
                self.store_x(i)
        elif solver == "momentum":
            sign = "-" if kind == "minimize" else "+"
            for i in range(n):
                self.comment(f"v[{i}] = beta * v[{i}] + g[{i}];  x[{i}] = x[{i}] {sign} lr * v[{i}]")
                self.slot_load(self.FA, v + 4 * i)
                self.fmv(self.FB, self.FBETA)
                self.fop("mul")
                self.slot_load(self.FB, g + 4 * i)
                self.fop("add")
                self.slot_store(self.FA, v + 4 * i)
                self.fmv(self.FB, self.FLR)
                self.fop("mul")
                self.fmv(self.FB, self.FA)
                self.ins(f"{self.ld(self.FA)} {self.FA}, {4 * i}(s2)")
                self.fop(step_op)
                self.store_x(i)
        else:   # newton
            self.ins("mv a0, s2")
            self.slot_addr("a1", h)
            self.call(f"hess_{fname}", "H = second derivatives at x")
            self.gen_solve(n, h, g, done, k)
            for i in range(n):
                self.comment(f"x[{i}] = x[{i}] - d[{i}]")
                self.ins(f"{self.ld(self.FA)} {self.FA}, {4 * i}(s2)")
                self.slot_load(self.FB, g + 4 * i)
                self.fop("sub")
                self.store_x(i, slack=True)

        self.ins("addi s3, s3, 1")
        self.ins(f"beqz s8, {done}", "x did not move: converged to machine precision")
        self.ins(f"j {loop}")
        self.label(done)
        if self.prog.uses_iterations:
            self.ins("la t0, __iters")
            self.ins("sw s3, 0(t0)", "remember how many iterations were used")
        self.frame.release(scratch)

    def store_x(self, i, slack=False):
        """x[i] = a0, noting in s8 whether x[i] changed.

        With slack=True (Newton) a change of one unit in the last place does
        not count: neighbouring floats differ by 1 when read as integers, and
        Newton steps can flicker between two neighbours at the optimum.
        """
        self.ins(f"lw t0, {4 * i}(s2)", "old value")
        self.store_and_diff(4 * i)
        if slack:
            self.ins("srai t1, t0, 31")
            self.ins("xor t0, t0, t1")
            self.ins("sub t0, t0, t1", "|change| counted in float steps")
            self.ins("srli t0, t0, 1", "ignore a change of one step")
        self.ins("or s8, s8, t0")

    def gen_solve(self, n, h, g, fail, k):
        """Solve H d = g by Gaussian elimination with row pivoting.

        H is n*n words at frame offset h, g is n words at offset g; the
        solution d overwrites g. The loops are unrolled at compile time
        because n is known. Jumps to `fail` if H is singular.
        """
        def a(i, j):
            return h + 4 * (i * n + j)

        def b(i):
            return g + 4 * i

        self.comment("solve H d = g (Gaussian elimination); d replaces g")
        for c in range(n):
            for r in range(c + 1, n):
                skip = f"opt{k}_piv{c}_{r}"
                self.comment(f"pivoting: swap rows {r} and {c} if |H[{r}][{c}]| > |H[{c}][{c}]|")
                self.slot_load("t0", a(r, c))
                self.slot_load("t1", a(c, c))
                self.ins("slli t0, t0, 1")
                self.ins("slli t1, t1, 1")
                self.ins(f"bgeu t1, t0, {skip}")
                for j in range(c, n):
                    self.slot_load("t0", a(r, j))
                    self.slot_load("t1", a(c, j))
                    self.slot_store("t1", a(r, j))
                    self.slot_store("t0", a(c, j))
                self.slot_load("t0", b(r))
                self.slot_load("t1", b(c))
                self.slot_store("t1", b(r))
                self.slot_store("t0", b(c))
                self.label(skip)
            ok = f"opt{k}_ok{c}"
            self.slot_load("t0", a(c, c))
            self.ins("slli t0, t0, 1")
            self.ins(f"bnez t0, {ok}")
            self.ins(f"j {fail}", "zero pivot: H is singular, stop")
            self.label(ok)
            for r in range(c + 1, n):
                self.comment(f"row {r} -= (H[{r}][{c}] / H[{c}][{c}]) * row {c}")
                self.slot_load(self.FA, a(r, c))
                self.slot_load(self.FB, a(c, c))
                self.fop("div")
                self.fmv(self.FT, self.FA)
                for j in list(range(c + 1, n)) + [None]:
                    src = b(c) if j is None else a(c, j)
                    dst = b(r) if j is None else a(r, j)
                    self.fmv(self.FA, self.FT)
                    self.slot_load(self.FB, src)
                    self.fop("mul")
                    self.fmv(self.FB, self.FA)
                    self.slot_load(self.FA, dst)
                    self.fop("sub")
                    self.slot_store(self.FA, dst)
        for r in reversed(range(n)):
            self.comment(f"back-substitution: d[{r}]")
            self.slot_load(self.FA, b(r))
            for j in range(r + 1, n):
                self.fmv(self.FT, self.FA)
                self.slot_load(self.FA, a(r, j))
                self.slot_load(self.FB, b(j))
                self.fop("mul")
                self.fmv(self.FB, self.FA)
                self.fmv(self.FA, self.FT)
                self.fop("sub")
            self.slot_load(self.FB, a(r, r))
            self.fop("div")
            self.slot_store(self.FA, b(r))

    # ------------------------------------------------------------- statements
    def source_comment(self, line):
        if 1 <= line <= len(self.src_lines):
            text = self.src_lines[line - 1].strip()
            self.out.append("")
            self.out.append(f"    # ---- line {line}: {_clip(text, 70)}")

    def gen_statement(self, st):
        self.source_comment(st[1])
        if st[0] == "assign":
            _, _, name, e, t, is_new = st
            if t == "scalar":
                self.gen(e)
                self.ins(f"la t0, g_{name}")
                self.ins(f"{self.st(self.FA)} {self.FA}, 0(t0)", f"{name} = result")
            elif is_new or e.kind == "pvar":
                self.gen_point(e, ("global", name))
            else:
                # re-assignment: build the new value aside, then copy it over,
                # because the expression (or the function it uses) may read
                # the old value while it runs
                n = t[1]
                tmp = self.frame.alloc(n)
                self.gen_point(e, ("frame", tmp))
                self.ins(f"la t1, g_{name}")
                for i in range(n):
                    self.slot_load("t2", tmp + 4 * i)
                    self.ins(f"sw t2, {4 * i}(t1)")
                self.frame.release(n)
        elif st[0] == "print":
            items = st[2]
            for idx, item in enumerate(items):
                if idx:
                    self.print_char(" ")
                if item[0] == "str":
                    sid = len(self.strings)
                    self.strings.append(item[1])
                    self.ins(f"la a0, str_{sid}")
                    self.call("__print_str", f"print \"{_clip(item[1], 40)}\"")
                elif item[0] == "scalar":
                    self.gen(item[1])
                    self.print_float()
                elif item[0] == "int":
                    self.ins("la t0, __iters")
                    self.ins("lw a0, 0(t0)")
                    self.print_int()
                else:
                    place, words = self.place_of(item[1])
                    self.place_addr("a0", place)
                    self.ins(f"li a1, {item[2]}")
                    self.call("__print_point")
                    self.frame.release(words)
            self.print_char("\n")
        else:
            raise AssertionError(st)

    # --------------------------------------------------------------- routines
    def begin_routine(self, saved_words, names=()):
        self.out = []
        self.frame = Frame(saved_words)
        self.slots, self.done = {}, set()
        self.names = list(names)

    def gen_main(self):
        self.begin_routine(0)
        for st in self.prog.statements:
            self.gen_statement(st)
        assert self.frame.used == 0, "frame allocations out of balance"
        body = self.out
        size = self.frame.size
        self.out = []
        self.emit_entry()
        self.ins("mv s0, sp", "s0 = frame pointer")
        if size:
            if size <= 2047:
                self.ins(f"addi sp, sp, -{size}", "room for temporaries")
            else:
                self.ins(f"li t0, {size}")
                self.ins("sub sp, sp, t0", "room for temporaries")
        self.out += body
        self.out.append("")
        self.out.append("    # ---- end of program")
        self.emit_exit()
        return self.out

    def gen_function(self, label, line, header, names, roots, sinks, has_out):
        """One callable routine evaluating `roots`.

        sinks[i] is None (leave root i in a0 as the return value) or a list of
        byte offsets into the output array (s2) that receive root i.
        """
        saved = 4 if has_out else 3             # ra, s0, s1 (, s2)
        self.begin_routine(saved, names)
        roots = [ir.expand_powers(r) for r in roots]
        for e in shared_subexpressions(roots):
            self.slots[e] = self.frame.alloc()
        for (root, sink, note) in zip(roots, sinks, header[1:]):
            if note:
                self.comment(note)
            self.gen(root)
            for off in sink or ():
                self.ins(f"{self.st(self.FA)} {self.FA}, {off}(s2)")
        body = self.out
        size = self.frame.size
        regs = ["ra", "s0", "s1"] + (["s2"] if has_out else [])

        self.out = []
        self.out.append(f"# ---- line {line}: {header[0]}")
        self.label(label)
        if size <= 2032:
            self.ins(f"addi sp, sp, -{size}")
            for i, r in enumerate(regs):
                self.ins(f"sw {r}, {size - 4 * (i + 1)}(sp)")
            self.ins(f"addi s0, sp, {size}", "s0 = frame pointer")
        else:
            self.ins("mv t1, sp")
            self.ins(f"li t0, {size}")
            self.ins("sub sp, sp, t0")
            for i, r in enumerate(regs):
                self.ins(f"sw {r}, {-4 * (i + 1)}(t1)")
            self.ins("mv s0, t1", "s0 = frame pointer")
        self.ins("mv s1, a0", "s1 = pointer to the arguments")
        if has_out:
            self.ins("mv s2, a1", "s2 = pointer to the output")
        self.out += body
        if size <= 2032:
            for i, r in enumerate(regs):
                self.ins(f"lw {r}, {size - 4 * (i + 1)}(sp)")
            self.ins(f"addi sp, sp, {size}")
        else:
            self.ins("mv t0, s0")
            for i, r in enumerate(regs):
                if r != "s0":
                    self.ins(f"lw {r}, {-4 * (i + 1)}(t0)")
            self.ins("lw s0, -8(t0)")
            self.ins("mv sp, t0")
        self.ins("ret")
        return self.out

    def gen_functions(self):
        chunks = []
        for f in self.prog.functions.values():
            n = len(f.params)
            sig = f"{f.name}({', '.join(f.params)})"
            body_text = _clip(ir.show(f.body, f.params), 100)
            chunks.append(self.gen_function(
                f"fn_{f.name}", f.line, [f"{sig} = {body_text}", None],
                f.params, [f.body], [None], has_out=False))
            if f.needs_grad:
                grad = ir.gradient(f.body, n)
                notes = [f"d{f.name}/d{p} = {_clip(ir.show(g, f.params), 80)}"
                         for p, g in zip(f.params, grad)]
                chunks.append(self.gen_function(
                    f"grad_{f.name}", f.line,
                    [f"gradient of {sig}: a0 = arguments, a1 = where to store the {n} partial derivative"
                     f"{'s' if n != 1 else ''}"] + notes,
                    f.params, grad, [[4 * i] for i in range(n)], has_out=True))
            if f.needs_hess:
                hess = ir.hessian(f.body, n)
                roots, sinks, notes = [], [], []
                for i in range(n):
                    for j in range(i, n):
                        roots.append(hess[i][j])
                        offs = [4 * (i * n + j)]
                        if i != j:
                            offs.append(4 * (j * n + i))
                        sinks.append(offs)
                        notes.append(f"d2{f.name}/d{f.params[i]}d{f.params[j]} = "
                                     f"{_clip(ir.show(hess[i][j], f.params), 80)}")
                chunks.append(self.gen_function(
                    f"hess_{f.name}", f.line,
                    [f"second derivatives of {sig}: a0 = arguments, a1 = where to store the "
                     f"{n}x{n} matrix"] + notes,
                    f.params, roots, sinks, has_out=True))
        return chunks

    # ------------------------------------------------------------------ whole
    def generate(self):
        main = self.gen_main()
        functions = self.gen_functions()

        lines = [
            f"# Generated by the Optima compiler from {self.filename}",
            f"# Target: {self.TARGET_NOTE}",
            "",
        ]
        data = []
        for name, t in self.prog.globals.items():
            n = 1 if t == "scalar" else t[1]
            data.append(f"g_{name}: .word {', '.join(['0'] * n)}")
        if self.prog.uses_iterations:
            data.append("__iters: .word 0")
        for i, s in enumerate(self.strings):
            data.append(f"str_{i}: .string \"{s}\"")
        if data:
            lines.append(".data")
            lines += data
            lines.append("")
        lines.append(".text")
        lines += main
        for chunk in functions:
            lines.append("")
            lines += chunk
        rt = self.runtime_text()
        if rt:
            lines += ["", "# " + "=" * 70,
                      f"# ---- runtime library: {self.RUNTIME_NOTE}",
                      "# " + "=" * 70, "", rt]
        return "\n".join(lines) + "\n"


def generate(prog, source="", filename="<input>", target="ripes"):
    if target == "qemu":
        from .qemu import QemuCodeGen
        return QemuCodeGen(prog, source, filename).generate()
    if target != "ripes":
        raise ValueError(f"unknown target {target!r}")
    return CodeGen(prog, source, filename).generate()
