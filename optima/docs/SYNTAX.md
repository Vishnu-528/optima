# Optima language reference (version 1)

Optima is a small language for numerical optimization. You write a function
as a formula, and ask for its minimum or maximum. The compiler works out the
derivatives itself and produces RISC-V assembly.

```
# comments start with '#'
target = 3
f(x, y) = (x - target)^2 + (y + 1)^2

r = minimize f from (0, 0) using gradient_descent(lr=0.1, steps=500, tol=1e-6)

print "minimum at", r           # minimum at (3, -1)
print "value there:", f(r)
print "iterations used:", iterations
print gradient f at (1, 2)      # (-4, 6)
print r[0], r[1]
```

## 1. Program structure

A program is a list of statements, one per line, run from top to bottom.
There are three kinds of statement:

| Statement | Form | Example |
|---|---|---|
| Assignment | `name = expression` | `rate = 0.1` |
| Function definition | `name(p1, p2, ...) = expression` | `f(x, y) = x^2 + y^2` |
| Print | `print item, item, ...` | `print "x =", r[0]` |

- There is no `let` and there are no type names. Every number is a float.
- A name must be defined on an earlier line before it is used.
- A newline ends a statement, except inside `( )` or `[ ]`, so a long
  expression can be split across lines inside parentheses.
- `#` starts a comment that runs to the end of the line.

## 2. Values

| Kind | Written as | Notes |
|---|---|---|
| Number | `3`, `2.5`, `.5`, `1e-6` | Single-precision float (about 7 digits). |
| Point | `(a, b, c)` | Two or more numbers. Its size is fixed. |
| Coordinate | `p[0]`, `p[1]` | The index is a whole number, starting at 0, checked at compile time. |

A point with one coordinate behaves like a plain number: it can be used in
arithmetic and prints without parentheses. A variable keeps its kind: a
number cannot later be assigned a point, or a point one of a different size.

## 3. Expressions

Operators, from lowest to highest precedence:

| Operators | Meaning | Notes |
|---|---|---|
| `+` `-` | add, subtract | left to right |
| `*` `/` | multiply, divide | left to right |
| `-x` | negate | `-x^2` means `-(x^2)` |
| `x^n` | power | `n` must be a whole number written out, such as `2` or `-1` |
| `p[i]` `f(...)` `( )` | index, call, grouping | |

Built-in functions: `sqrt(x)`, `exp(x)`, `log(x)` (natural log), `sin(x)`,
`cos(x)`.

Invalid operations do not stop the program; they give `inf` or `nan`
(for example `1 / 0` is `inf`, `sqrt(-1)` is `nan`).

## 4. Functions

```
square(x) = x^2
dist(x, y) = sqrt(square(x) + square(y))
```

- The body is one expression. It may use the parameters, numbers, the
  built-in functions, functions defined earlier, and variables defined
  earlier (including coordinates of points, such as `c[0]`).
- A variable used in a function is read when the function runs, so
  assigning it a new value later changes the function.
- A function may not call itself, and its body may not contain `minimize`,
  `maximize`, `gradient`, `iterations` or a point.
- Call a function with one argument per parameter, `f(1, 2)`, or with a
  single point that has one coordinate per parameter, `f(r)`.
- Limits: at most 256 parameters.

## 5. Optimization

```
minimize FUNCTION from START [using SOLVER[(option=value, ...)]]
maximize FUNCTION from START [using SOLVER[(option=value, ...)]]
```

- `FUNCTION` is the name of a function you defined.
- `START` is the point to start from: a point in parentheses, a point
  variable, or a single number for a one-parameter function. It must have
  one coordinate per parameter. Write a computed start in parentheses:
  `from (2 * a)`.
- The result is the point that was found. Assign it or print it.
- `using` is optional; the default is `gradient_descent`.

| Solver | Each step | Options (default) |
|---|---|---|
| `gradient_descent` | `x = x - lr * gradient` | `lr` (0.1), `steps` (1000), `tol` (1e-6) |
| `momentum` | `v = beta * v + gradient`, then `x = x - lr * v` | `lr` (0.01), `beta` (0.9), `steps` (1000), `tol` (1e-6) |
| `newton` | `x = x - H⁻¹ * gradient`, with `H` the matrix of second derivatives | `steps` (50), `tol` (1e-6) |

`maximize` takes the same steps uphill (`+` instead of `-`). Newton's method
goes to the nearest point where the gradient is zero, whichever of
`minimize` or `maximize` is written, so start it near the answer you want.
It supports functions of up to 16 parameters.

Options:

- `lr`, `beta` and `tol` can be any numeric expression.
- `steps` must be a whole number of at least 1, written out.

A solver stops as soon as one of these is true:

1. the length of the gradient is below `tol` (converged);
2. `steps` steps have been taken;
3. a step no longer changes `x` (it has converged as far as single
   precision allows);
4. Newton only: the second-derivative matrix is singular.

`iterations` is the number of steps taken by the most recent
`minimize`/`maximize` (0 before the first one). It prints as a whole number.

## 6. Gradient

```
gradient FUNCTION at POINT
```

gives the point of partial derivatives, one per parameter:
`gradient f at (1, 2)` or, for one coordinate, `(gradient f at (1, 2))[0]`.

## 7. Print

`print` writes its items on one line, separated by spaces. An item is an
expression or a piece of text in double quotes. Numbers print with up to 6
significant digits and points as `(x, y)`. Text may contain any printable
characters except `"` and `\`, and can only appear directly in `print`.

## 8. Reserved words

`print minimize maximize from using gradient at iterations` cannot be used
as names. `sqrt exp log sin cos` and the solver names
`gradient_descent momentum newton` cannot be used as variable or function
names. A variable and a function cannot share a name.

## 9. Grammar

```
program      = { [ statement ] NEWLINE } ;
statement    = assignment | function_def | print_stmt ;
assignment   = IDENT "=" expr ;
function_def = IDENT "(" IDENT { "," IDENT } ")" "=" expr ;
print_stmt   = "print" item { "," item } ;
item         = STRING | expr ;

expr     = term  { ("+" | "-") term } ;
term     = unary { ("*" | "/") unary } ;
unary    = "-" unary | power ;
power    = postfix [ "^" [ "-" ] INT ] ;
postfix  = primary { "[" INT "]" } ;
primary  = NUMBER
         | IDENT
         | IDENT "(" expr { "," expr } ")"
         | "(" expr { "," expr } ")"            (* one: grouping; several: a point *)
         | "iterations"
         | "gradient" IDENT "at" unary
         | ("minimize" | "maximize") IDENT "from" unary [ "using" solver ] ;
solver   = IDENT [ "(" [ option { "," option } ] ")" ] ;
option   = IDENT "=" expr ;

NUMBER   = digits [ "." digits ] [ ("e" | "E") [ "+" | "-" ] digits ] | "." digits ... ;
INT      = digits ;
IDENT    = letter or "_" , then letters, digits or "_" ;
STRING   = '"' printable characters except '"' and '\' '"' ;
```

## 10. What version 1 does not have

`if`, loops, user-written algorithms, arithmetic on whole points, constraints
and bounds, data fitting, strings as values, and Adam. Floats are single
precision, so results are good to about 6 digits and a `tol` much below
`1e-6` is usually unreachable.
