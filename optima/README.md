# Optima

A small language for numerical optimization, with a compiler written in
Python that produces RISC-V assembly. It has two back ends:

- **Version 1, Ripes**: RV32IM with floating point done in software, for the
  [Ripes](https://github.com/mortbopet/Ripes) simulator.
- **Version 2, QEMU**: RV32IMFD with hardware floating point, running bare
  metal on QEMU's `virt` machine. See [docs/QEMU.md](docs/QEMU.md).

```
target = 3
f(x, y) = (x - target)^2 + (y + 1)^2
r = minimize f from (0, 0) using gradient_descent(lr=0.1, steps=500)
print "minimum at", r, "after", iterations, "iterations"
```

```
minimum at (3, -1) after 71 iterations
```

You write the function; the compiler differentiates it symbolically and
generates the code for the function, its gradient, its second derivatives
(for Newton's method) and the solver loop.

The language is described in [docs/SYNTAX.md](docs/SYNTAX.md).

## Quick start

Needs Python 3.8 or newer. No packages to install.

```sh
python optc.py examples/quadratic.opt          # writes examples/quadratic.s
python optc.py examples/quadratic.opt --run    # compile and run in the test simulator
python optc.py examples/quadratic.opt -o -     # print the assembly
```

To run on QEMU instead (needs LLVM and QEMU installed, see
[docs/QEMU.md](docs/QEMU.md)):

```sh
python optc.py examples/quadratic.opt --target qemu --run
```

To run in Ripes, either

- open the `.s` file in Ripes (File -> Load Program, Source file) with an
  RV32 processor and the M extension enabled, or
- patch Ripes so it runs the compiler itself: see
  [ripes/README.md](ripes/README.md).

## Layout

| Path | What it is |
|---|---|
| `optc.py` | Command-line driver. |
| `optima/lexer.py` | Source text -> tokens. |
| `optima/parser.py`, `optima/syntax.py` | Tokens -> syntax tree (recursive descent). |
| `optima/sema.py` | Name, argument and point-size checks; builds the IR; inlines function calls. |
| `optima/ir.py` | Expression IR, simplification, symbolic differentiation. |
| `optima/codegen.py` | IR -> assembly, including the three solvers. Ripes back end. |
| `optima/runtime.py` | Ripes: software floating point and printing, in assembly. |
| `optima/qemu.py` | QEMU back end: hardware-float instructions, serial-port printing, `exp`/`log`/`sin`/`cos`, and the build-and-run steps. |
| `optima/sim.py` | A small RV32IM interpreter used by `--run` and the tests. |
| `examples/` | Example programs. |
| `tests/` | `test_softfloat.py`, `test_compiler.py` and `test_qemu.py`. |
| `ripes/` | Patch and instructions for running Optima inside Ripes. |

## How the compiler works

```
source -> lexer -> parser -> semantic analysis -> code generator -> prog.s
                                   |                    |
                             differentiation      runtime library
```

1. **Lexer and parser** turn the text into a syntax tree and report syntax
   errors with line and column.
2. **Semantic analysis** checks names, argument counts and point sizes, and
   converts each expression to a compact IR. A call from one function to
   another is inlined, so every function body is one expression.
3. **Differentiation** applies the usual rules (sum, product, quotient,
   chain, power) to that expression. The gradient is the list of partial
   derivatives; differentiating it again gives the second derivatives.
   Results are simplified as they are built (`x*1`, `x+0`, constant folding).
4. **Code generation** walks each expression and emits assembly that leaves
   the value in `a0`. Subexpressions shared between the partial derivatives
   are computed once and kept in the stack frame. Integer powers become
   repeated multiplication.
5. **Solvers** are generated inline for each `minimize`/`maximize`, unrolled
   over the number of variables. Newton's method solves its linear system by
   Gaussian elimination with row pivoting.

### Floating point (Ripes back end)

Ripes implements RV32I with the M and C extensions but not F, so there are
no floating-point instructions. Optima's floats are IEEE-754 single-precision
bit patterns held in integer registers, and `runtime.py` provides add,
subtract, multiply, divide, compare, conversions, `sqrt`, `exp`, `log`,
`sin` and `cos` in plain RV32IM assembly. Add, subtract, multiply and divide
round exactly as IEEE-754 requires; the math functions are accurate to about
2 units in the last place. Denormal numbers are treated as zero. Only the
routines a program uses are added to its output.

### Conventions in the generated assembly (Ripes back end)

- `fn_f`: `a0` = pointer to the arguments, returns `f` in `a0`.
- `grad_f`, `hess_f`: `a0` = pointer to the arguments, `a1` = where to write
  the result.
- Global variables are `g_<name>` in `.data`.
- Output uses the Ripes `print_int`, `print_float`, `print_char` ecalls, and
  the program ends with the `exit` ecall.

## Tests

```sh
python tests/test_softfloat.py     # floating-point routines against exact results
python tests/test_compiler.py      # programs, error messages, derivatives
```

Both run in the built-in simulator by default. To run them in real Ripes
(2.2.5 or newer, which has a command-line mode):

```sh
RIPES=/path/to/Ripes python tests/test_compiler.py
```

To run the same compiler tests through the QEMU back end, and to test its
own runtime routines:

```sh
OPTIMA_TARGET=qemu python tests/test_compiler.py
python tests/test_qemu.py
```

On Windows, set the variable first: `set OPTIMA_TARGET=qemu` (cmd) or
`$env:OPTIMA_TARGET="qemu"` (PowerShell).
