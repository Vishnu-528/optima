"""A small RV32IM interpreter for testing compiler output without opening Ripes.

It understands the subset of assembly the Optima compiler emits (plus the
common pseudo-instructions) and the Ripes print/exit ecalls. It is a testing
aid, not a model of Ripes: every instruction, including pseudo-instructions,
counts as one step, and there is no pipeline or cache.

    python optc.py prog.opt --run
"""

import re
import struct

from .errors import OptimaError

REG_NAMES = (["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1"]
             + [f"a{i}" for i in range(8)]
             + [f"s{i}" for i in range(2, 12)]
             + [f"t{i}" for i in range(3, 7)])
REGS = {name: i for i, name in enumerate(REG_NAMES)}
REGS.update({f"x{i}": i for i in range(32)})
REGS["fp"] = 8

DATA_BASE = 0x10000000
STACK_TOP = 0x7FFFFFF0
STACK_SIZE = 1 << 20
MASK = 0xFFFFFFFF


class SimError(Exception):
    pass


def _signed(v):
    return v - 0x100000000 if v & 0x80000000 else v


def _parse_int(text):
    text = text.strip()
    try:
        return int(text, 0)
    except ValueError:
        raise SimError(f"bad number '{text}'")


class Program:
    def __init__(self, text):
        self.instrs = []          # (mnemonic, [operands], source line no.)
        self.labels = {}          # name -> address
        self.data = bytearray()
        self._assemble(text)

    def _assemble(self, text):
        section = "text"
        for lineno, raw in enumerate(text.splitlines(), 1):
            line = raw
            # strip comments (but not inside a string)
            if '"' not in line:
                line = line.split("#", 1)[0]
            else:
                q = line.find('"')
                h = line.find("#")
                if 0 <= h < q:
                    line = line[:h]
                else:
                    end = line.rfind('"')
                    line = line[:end + 1] + line[end + 1:].split("#", 1)[0]
            line = line.strip()
            while True:
                m = re.match(r"^([A-Za-z_.][\w.]*):\s*", line)
                if not m:
                    break
                addr = len(self.instrs) * 4 if section == "text" else DATA_BASE + len(self.data)
                if m.group(1) in self.labels:
                    raise SimError(f"line {lineno}: label '{m.group(1)}' defined twice")
                self.labels[m.group(1)] = addr
                line = line[m.end():]
            if not line:
                continue
            parts = line.split(None, 1)
            op = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            if op.startswith("."):
                if op == ".data":
                    section = "data"
                elif op == ".text":
                    section = "text"
                elif op in (".globl", ".global"):
                    pass
                elif op == ".word":
                    for v in rest.split(","):
                        self.data += struct.pack("<I", _parse_int(v) & MASK)
                elif op == ".zero":
                    self.data += bytes(_parse_int(rest))
                elif op in (".string", ".asciz"):
                    s = rest.strip()
                    if not (s.startswith('"') and s.endswith('"')):
                        raise SimError(f"line {lineno}: bad string")
                    self.data += s[1:-1].encode("ascii") + b"\0"
                    while len(self.data) % 4:
                        self.data += b"\0"
                else:
                    raise SimError(f"line {lineno}: unsupported directive {op}")
                continue
            if section != "text":
                raise SimError(f"line {lineno}: instruction in .data")
            ops = [o.strip() for o in rest.split(",")] if rest else []
            self.instrs.append((op, ops, lineno))


def run(asm_text, max_steps=200_000_000):
    """Assemble and run. Returns (console output, instructions executed)."""
    prog = Program(asm_text)
    labels = prog.labels
    data = bytearray(prog.data) + bytes(4096)
    stack = bytearray(STACK_SIZE)
    stack_base = 0x80000000 - STACK_SIZE
    x = [0] * 32
    x[2] = STACK_TOP
    x[3] = DATA_BASE
    out = []

    def seg(addr):
        if DATA_BASE <= addr < DATA_BASE + len(data):
            return data, addr - DATA_BASE
        if stack_base <= addr < 0x80000000:
            return stack, addr - stack_base
        raise SimError(f"memory access outside data/stack at 0x{addr:08x}")

    def load32(addr):
        b, o = seg(addr)
        return int.from_bytes(b[o:o + 4], "little")

    def store32(addr, v):
        b, o = seg(addr)
        b[o:o + 4] = (v & MASK).to_bytes(4, "little")

    def reg(name, lineno):
        try:
            return REGS[name]
        except KeyError:
            raise SimError(f"line {lineno}: unknown register '{name}'")

    def target(name, lineno):
        if name not in labels:
            raise SimError(f"line {lineno}: unknown label '{name}'")
        return labels[name]

    def mem_operand(text, lineno):
        m = re.match(r"^(-?\w*)\((\w+)\)$", text.replace(" ", ""))
        if not m:
            raise SimError(f"line {lineno}: bad memory operand '{text}'")
        return (_parse_int(m.group(1)) if m.group(1) else 0), reg(m.group(2), lineno)

    # pre-decode every instruction into (kind, a, b, c)
    R_OPS = {"add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and",
             "mul", "mulh", "mulhu", "mulhsu", "div", "divu", "rem", "remu"}
    I_OPS = {"addi", "andi", "ori", "xori", "slti", "sltiu", "slli", "srli", "srai"}
    B_OPS = {"beq", "bne", "blt", "bge", "bltu", "bgeu"}
    B_SWAP = {"bgt": "blt", "ble": "bge", "bgtu": "bltu", "bleu": "bgeu"}
    B_ZERO = {"beqz": ("beq", False), "bnez": ("bne", False), "bgez": ("bge", False),
              "bltz": ("blt", False), "blez": ("bge", True), "bgtz": ("blt", True)}
    code = []
    for op, ops, ln in prog.instrs:
        try:
            if op in R_OPS:
                code.append((op, reg(ops[0], ln), reg(ops[1], ln), reg(ops[2], ln)))
            elif op in I_OPS:
                code.append((op, reg(ops[0], ln), reg(ops[1], ln), _parse_int(ops[2])))
            elif op in ("lw", "sw", "lb", "lbu", "sb"):
                off, base = mem_operand(ops[1], ln)
                code.append((op, reg(ops[0], ln), base, off))
            elif op in B_OPS:
                code.append((op, reg(ops[0], ln), reg(ops[1], ln), target(ops[2], ln)))
            elif op in B_SWAP:
                code.append((B_SWAP[op], reg(ops[1], ln), reg(ops[0], ln), target(ops[2], ln)))
            elif op in B_ZERO:
                real, swap = B_ZERO[op]
                r = reg(ops[0], ln)
                a, b = (0, r) if swap else (r, 0)
                code.append((real, a, b, target(ops[1], ln)))
            elif op == "li":
                code.append(("li", reg(ops[0], ln), _parse_int(ops[1]) & MASK, 0))
            elif op == "la":
                code.append(("li", reg(ops[0], ln), target(ops[1], ln), 0))
            elif op == "lui":
                code.append(("li", reg(ops[0], ln), (_parse_int(ops[1]) << 12) & MASK, 0))
            elif op == "mv":
                code.append(("addi", reg(ops[0], ln), reg(ops[1], ln), 0))
            elif op == "neg":
                code.append(("sub", reg(ops[0], ln), 0, reg(ops[1], ln)))
            elif op == "not":
                code.append(("xori", reg(ops[0], ln), reg(ops[1], ln), -1))
            elif op == "seqz":
                code.append(("sltiu", reg(ops[0], ln), reg(ops[1], ln), 1))
            elif op == "snez":
                code.append(("sltu", reg(ops[0], ln), 0, reg(ops[1], ln)))
            elif op == "sltz":
                code.append(("slt", reg(ops[0], ln), reg(ops[1], ln), 0))
            elif op == "sgtz":
                code.append(("slt", reg(ops[0], ln), 0, reg(ops[1], ln)))
            elif op == "j":
                code.append(("jal", 0, target(ops[0], ln), 0))
            elif op in ("jal", "call"):
                if len(ops) == 2:
                    code.append(("jal", reg(ops[0], ln), target(ops[1], ln), 0))
                else:
                    code.append(("jal", 1, target(ops[0], ln), 0))
            elif op == "jalr":
                if len(ops) == 1:
                    code.append(("jalr", 1, reg(ops[0], ln), 0))
                else:
                    off, base = mem_operand(ops[1], ln)
                    code.append(("jalr", reg(ops[0], ln), base, off))
            elif op == "jr":
                code.append(("jalr", 0, reg(ops[0], ln), 0))
            elif op == "ret":
                code.append(("jalr", 0, 1, 0))
            elif op == "nop":
                code.append(("addi", 0, 0, 0))
            elif op == "ecall":
                code.append(("ecall", 0, 0, 0))
            else:
                raise SimError(f"line {ln}: unsupported instruction '{op}'")
        except IndexError:
            raise SimError(f"line {ln}: too few operands for '{op}'")

    pc = labels.get("main", 0)
    n = len(code)
    steps = 0
    while True:
        idx = pc >> 2
        if idx >= n or pc & 3:
            break                      # ran off the end of .text: stop, like Ripes
        op, a, b, c = code[idx]
        steps += 1
        if steps > max_steps:
            raise SimError(f"stopped after {max_steps} instructions (infinite loop?)")
        pc += 4
        if op == "addi":
            x[a] = (x[b] + c) & MASK
        elif op == "lw":
            x[a] = load32((x[b] + c) & MASK)
        elif op == "sw":
            store32((x[b] + c) & MASK, x[a])
        elif op == "li":
            x[a] = b
        elif op == "jal":
            x[a] = pc
            pc = b
        elif op == "jalr":
            t = (x[b] + c) & MASK & ~1
            x[a] = pc
            pc = t
        elif op == "slli":
            x[a] = (x[b] << c) & MASK
        elif op == "srli":
            x[a] = x[b] >> c
        elif op == "srai":
            x[a] = (_signed(x[b]) >> c) & MASK
        elif op == "or":
            x[a] = x[b] | x[c]
        elif op == "and":
            x[a] = x[b] & x[c]
        elif op == "xor":
            x[a] = x[b] ^ x[c]
        elif op == "add":
            x[a] = (x[b] + x[c]) & MASK
        elif op == "sub":
            x[a] = (x[b] - x[c]) & MASK
        elif op == "beq":
            if x[a] == x[b]:
                pc = c
        elif op == "bne":
            if x[a] != x[b]:
                pc = c
        elif op == "blt":
            if _signed(x[a]) < _signed(x[b]):
                pc = c
        elif op == "bge":
            if _signed(x[a]) >= _signed(x[b]):
                pc = c
        elif op == "bltu":
            if x[a] < x[b]:
                pc = c
        elif op == "bgeu":
            if x[a] >= x[b]:
                pc = c
        elif op == "andi":
            x[a] = x[b] & (c & MASK)
        elif op == "ori":
            x[a] = x[b] | (c & MASK)
        elif op == "xori":
            x[a] = x[b] ^ (c & MASK)
        elif op == "sll":
            x[a] = (x[b] << (x[c] & 31)) & MASK
        elif op == "srl":
            x[a] = x[b] >> (x[c] & 31)
        elif op == "sra":
            x[a] = (_signed(x[b]) >> (x[c] & 31)) & MASK
        elif op == "slt":
            x[a] = int(_signed(x[b]) < _signed(x[c]))
        elif op == "sltu":
            x[a] = int(x[b] < x[c])
        elif op == "slti":
            x[a] = int(_signed(x[b]) < c)
        elif op == "sltiu":
            x[a] = int(x[b] < (c & MASK))
        elif op == "mul":
            x[a] = (x[b] * x[c]) & MASK
        elif op == "mulhu":
            x[a] = (x[b] * x[c]) >> 32
        elif op == "mulh":
            x[a] = ((_signed(x[b]) * _signed(x[c])) >> 32) & MASK
        elif op == "mulhsu":
            x[a] = ((_signed(x[b]) * x[c]) >> 32) & MASK
        elif op == "divu":
            x[a] = MASK if x[c] == 0 else x[b] // x[c]
        elif op == "remu":
            x[a] = x[b] if x[c] == 0 else x[b] % x[c]
        elif op == "div":
            p, q = _signed(x[b]), _signed(x[c])
            x[a] = MASK if q == 0 else int(abs(p) // abs(q) * (1 if (p < 0) == (q < 0) else -1)) & MASK
        elif op == "rem":
            p, q = _signed(x[b]), _signed(x[c])
            x[a] = x[b] if q == 0 else (abs(p) % abs(q) * (1 if p >= 0 else -1)) & MASK
        elif op in ("lb", "lbu"):
            buf, o = seg((x[b] + c) & MASK)
            v = buf[o]
            x[a] = (v - 256) & MASK if (op == "lb" and v & 0x80) else v
        elif op == "sb":
            buf, o = seg((x[b] + c) & MASK)
            buf[o] = x[a] & 0xFF
        elif op == "ecall":
            call = x[17]
            if call == 1:
                out.append(str(_signed(x[10])))
            elif call == 2:
                out.append(format_float(x[10]))
            elif call == 4:
                addr = x[10]
                chars = []
                while True:
                    buf, o = seg(addr)
                    if buf[o] == 0:
                        break
                    chars.append(chr(buf[o]))
                    addr += 1
                out.append("".join(chars))
            elif call == 11:
                out.append(chr(x[10] & 0xFF))
            elif call == 34:
                out.append(f"0x{x[10]:08x}")
            elif call in (10, 93):
                break
            else:
                raise SimError(f"unsupported ecall {call}")
        else:
            raise SimError(f"internal: unhandled op {op}")
        x[0] = 0
    return "".join(out), steps


def format_float(bits):
    """Format a float32 bit pattern the way Ripes' print_float ecall does."""
    v = struct.unpack("<f", struct.pack("<I", bits & MASK))[0]
    return "%g" % v


def run_or_raise(asm_text, **kw):
    try:
        return run(asm_text, **kw)
    except SimError as e:
        raise OptimaError(f"simulator: {e}")
