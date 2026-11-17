"""Optima: a small language for numerical optimization, compiled to RISC-V.

Pipeline:  source -> lexer -> parser -> sema (checks + IR) -> codegen -> assembly
"""

import sys

from . import codegen, sema
from .errors import OptimaError
from .parser import parse

__all__ = ["compile_source", "OptimaError"]
__version__ = "2.0"

sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))


def compile_source(source, filename="<input>", target="ripes"):
    """Compile Optima source text to RISC-V assembly text.

    target="ripes": RV32IM with software floats and Ripes system calls.
    target="qemu":  RV32IMFD with hardware floats, for QEMU's virt machine.
    """
    tree = parse(source)
    program = sema.analyze(tree)
    return codegen.generate(program, source, filename, target)
