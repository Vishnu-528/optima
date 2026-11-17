"""Syntax tree produced by the parser. Every node remembers where it came from."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    line: int
    col: int


# ---------------------------------------------------------------- expressions

@dataclass
class Number(Node):
    value: float
    is_int: bool = False


@dataclass
class Name(Node):
    name: str


@dataclass
class Call(Node):
    name: str
    args: list


@dataclass
class PointLit(Node):
    elems: list


@dataclass
class Index(Node):
    base: Node
    index: int


@dataclass
class Unary(Node):          # only '-'
    operand: Node


@dataclass
class Binary(Node):
    op: str                 # + - * /
    left: Node
    right: Node


@dataclass
class Power(Node):
    base: Node
    exponent: int


@dataclass
class Iterations(Node):
    pass


@dataclass
class Gradient(Node):
    func: str
    func_pos: tuple
    at: Node


@dataclass
class Optimize(Node):
    kind: str               # 'minimize' | 'maximize'
    func: str
    func_pos: tuple
    start: Node
    solver: str = "gradient_descent"
    solver_pos: Optional[tuple] = None
    options: dict = field(default_factory=dict)   # name -> (expr node, (line, col))


@dataclass
class Str(Node):
    text: str


# ----------------------------------------------------------------- statements

@dataclass
class Assign(Node):
    name: str
    value: Node


@dataclass
class FuncDef(Node):
    name: str
    params: list            # list of (name, line, col)
    body: Node


@dataclass
class Print(Node):
    items: list
