# Optima on QEMU (version 2)

The QEMU back end produces RV32IMFD code with hardware floating point and
runs it on QEMU's `virt` machine with no operating system underneath.

```
python optc.py examples/quadratic.opt --target qemu --run
```

The language is the same as in version 1; see [SYNTAX.md](SYNTAX.md). All
five example programs print exactly the same output on both back ends.

## What you need

| Tool | Used for | Install |
|---|---|---|
| Python 3.8+ | the compiler | python.org |
| LLVM (`clang`, `ld.lld`) | assembling and linking the generated `.s` | Windows: `winget install LLVM.LLVM` or the installer from the LLVM releases page. Ubuntu: `sudo apt install clang lld`. macOS: `brew install llvm lld` |
| QEMU (`qemu-system-riscv32`) | running the program | Windows: installer linked from qemu.org/download. Ubuntu: `sudo apt install qemu-system-misc`. macOS: `brew install qemu` |

Neither QEMU nor LLVM is modified or rebuilt.

The compiler looks for the tools on `PATH`, and on Windows also in
`C:\Program Files\LLVM\bin` and `C:\Program Files\qemu`. If they are
somewhere else, set `OPTIMA_CLANG`, `OPTIMA_LLD` and `OPTIMA_QEMU` to the
full paths of `clang`, `ld.lld` and `qemu-system-riscv32`.

This back end was built and tested on Linux (clang 18, QEMU 8.2). The
Windows tool lookup is written but has not been run on Windows.

## Commands

```
python optc.py prog.opt --target qemu --run        compile, build and run
python optc.py prog.opt --target qemu              write prog.s (assembly only, needs no tools)
python optc.py prog.opt --target qemu -o prog.elf  build the file QEMU loads
```

`-t qemu` is short for `--target qemu`. To do the build and run steps by
hand (the linker script is `LINKER_SCRIPT` in `optima/qemu.py`):

```
clang --target=riscv32-unknown-elf -march=rv32imafd_zicsr -mabi=ilp32d -c prog.s -o prog.o
ld.lld -m elf32lriscv -T link.ld prog.o -o prog.elf
qemu-system-riscv32 -M virt -cpu rv32 -bios none -display none -monitor none -serial stdio -kernel prog.elf
```

## How it differs from the Ripes back end

The lexer, parser, checks, differentiation and solver logic are shared.
`optima/qemu.py` replaces only the target-specific parts of `codegen.py`:

| | Ripes (version 1) | QEMU (version 2) |
|---|---|---|
| Floats | bit patterns in integer registers (`a0`, `a1`) | float registers (`fa0`, `fa1`) |
| `+ - * /`, `sqrt`, compare | calls to software routines | one instruction each (`fadd.s`, `fsqrt.s`, `flt.s`, ...) |
| `exp`, `log`, `sin`, `cos` | software routines, about 2 units in the last place | short routines that compute in double precision |
| Printing | Ripes system calls | characters written to the serial port at `0x10000000`; `__print_float` turns the number into text |
| Start | first instruction | `_start` at `0x80000000` sets the stack and switches the floating-point unit on |
| End | Ripes `exit` call | write `0x5555` to the power-off device at `0x100000` |
| Solver registers (lr, tol, beta) | `s4`-`s6` | `fs0`-`fs2` |

Results are still single precision (about 6 digits).

## Limits

- `sin` and `cos` of numbers at or above 1e9 in size give `nan`. At that
  size neighbouring single-precision numbers are more than a full turn
  apart, so the result would carry no information.
- Numbers print the way C's `%g` does. For a value exactly halfway between
  two 6-digit numbers, the last digit can differ from C by one.
- If the processor faults, the run stops with exit code 1. A program that
  runs longer than 60 seconds under `--run` is stopped.
