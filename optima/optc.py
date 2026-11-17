#!/usr/bin/env python3
"""Optima compiler driver.

Version 1 target, Ripes (RV32IM, software floats):
    python optc.py prog.opt              write prog.s (RISC-V assembly for Ripes)
    python optc.py prog.opt -o out.s     choose the output file
    python optc.py prog.opt -o -         write the assembly to standard output
    python optc.py prog.opt --run        compile and run in the built-in test simulator

Version 2 target, QEMU (RV32IMFD, hardware floats; needs LLVM and QEMU installed):
    python optc.py prog.opt --target qemu --run      compile, build and run in QEMU
    python optc.py prog.opt --target qemu            write prog.s
    python optc.py prog.opt --target qemu -o p.elf   build the program file QEMU loads
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optima import compile_source          # noqa: E402
from optima.errors import OptimaError      # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="optc", description="Compile an Optima program to RISC-V assembly "
                                 "for Ripes or for QEMU.")
    ap.add_argument("source", help="Optima source file (.opt), or - for standard input")
    ap.add_argument("-t", "--target", choices=("ripes", "qemu"), default="ripes",
                    help="ripes: RV32IM with software floats (default); "
                         "qemu: RV32IMFD with hardware floats")
    ap.add_argument("-o", "--output",
                    help="output file (default: <source>.s; - for stdout). With --target qemu, "
                         "a name ending in .elf builds the executable for QEMU")
    ap.add_argument("--run", action="store_true",
                    help="run the result and print its output: in the built-in simulator "
                         "for ripes, in QEMU for qemu")
    ap.add_argument("--stats", action="store_true",
                    help="with --run: also report the number of instructions executed")
    args = ap.parse_args(argv)

    if args.source == "-":
        source, filename = sys.stdin.read(), "<stdin>"
    else:
        try:
            with open(args.source, encoding="utf-8") as fh:
                source = fh.read()
        except OSError as e:
            print(f"optc: cannot read {args.source}: {e.strerror}", file=sys.stderr)
            return 2
        filename = os.path.basename(args.source)

    try:
        asm = compile_source(source, filename, target=args.target)
    except OptimaError as e:
        print(e.format(source, filename), file=sys.stderr)
        return 1

    if args.target == "qemu":
        return finish_qemu(args, asm)

    if args.run:
        from optima import sim
        try:
            output, steps = sim.run(asm)
        except sim.SimError as e:
            print(f"optc: simulator error: {e}", file=sys.stderr)
            return 3
        sys.stdout.write(output)
        if args.stats:
            print(f"[{steps} instructions executed]", file=sys.stderr)
        if not args.output:
            return 0

    out = args.output
    if out is None:
        out = "-" if args.source == "-" else os.path.splitext(args.source)[0] + ".s"
    if out == "-":
        sys.stdout.write(asm)
    else:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(asm)
    return 0


def finish_qemu(args, asm):
    from optima import qemu
    out = args.output
    try:
        if args.run:
            sys.stdout.write(qemu.run(asm))
            sys.stdout.flush()
            if not out:
                return 0
        if out is None:
            out = "-" if args.source == "-" else os.path.splitext(args.source)[0] + ".s"
        if out == "-":
            sys.stdout.write(asm)
        elif out.lower().endswith(".elf"):
            qemu.build_elf(asm, out)
        else:
            with open(out, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(asm)
    except qemu.ToolError as e:
        print(f"optc: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
