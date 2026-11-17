# Running Optima directly inside Ripes

`optima-ripes.patch` changes the Ripes source so that Ripes runs the Optima
compiler (`optc.py`) itself. After patching and building you get:

- an **Optima** input type next to *Assembly* and *C* in the Editor tab;
- the program is compiled each time you stop typing, and loaded into the
  simulator;
- compile errors underline the offending line (hover for the message);
- while stepping, the Optima line being executed is highlighted;
- a **Generated assembly...** button that shows (and can save) the RISC-V
  assembly the compiler produced;
- **File -> Load Example -> Optima** with the example programs;
- `.opt` files in *Load Program* and *Save File*;
- command-line mode: `Ripes --mode cli --src prog.opt -t optima --proc RV32_SS --isaexts M`.

You do not need the patch to use the compiler: `python optc.py prog.opt`
writes `prog.s`, which loads into an unmodified Ripes as assembly.

## Build steps

The patch was made against Ripes commit
`dd0e2ebb7bfc373e5c161e82fec152a258751163` (master, 11 Nov 2026). Ripes at
that commit needs Qt 6.8 or newer (with Qt Charts and Qt SVG), CMake and a
C++20 compiler; see the Ripes README for its own build notes.

```sh
git clone --recursive https://github.com/mortbopet/Ripes.git
cd Ripes
git checkout dd0e2ebb7bfc373e5c161e82fec152a258751163
git submodule update --init --recursive
git apply /path/to/optima/ripes/optima-ripes.patch

cmake -B build -DCMAKE_BUILD_TYPE=Release -DOPTIMA_DIR=/path/to/optima
cmake --build build
./build/Ripes
```

`OPTIMA_DIR` is the folder that contains `optc.py`. Ripes runs the compiler
from that folder, so changes you make to the compiler take effect the next
time Ripes compiles, without rebuilding Ripes.

## How Ripes finds the compiler

In this order:

1. **Edit -> Settings -> Compiler -> Optima compiler** (Python interpreter
   and path to `optc.py`);
2. the `OPTIMA_COMPILER` environment variable (full path to `optc.py`);
3. the `OPTIMA_DIR` given when Ripes was built (default: an `optima` folder
   in the Ripes source tree);
4. an `optima` folder next to the Ripes executable.

Python is taken from the setting, or else the first of `python3`, `python`
(`python`, `py`, `python3` on Windows) that reports version 3.

## Using it

1. Select a 32-bit processor (any RV32 model) with the **M** extension
   ticked. This is the Ripes default.
2. Editor tab -> Input type: **Optima**.
3. Type a program or load an example, then press Run (or `>>` for full
   speed). Output appears in the Console.

## What the patch touches

| File | Change |
|---|---|
| `src/optimamanager.{h,cpp}` (new) | Finds Python and `optc.py`, runs the compiler (source on stdin, assembly on stdout), maps code addresses back to Optima lines. |
| `src/editor/optimasyntaxhighlighter.{h,cpp}` (new) | Syntax colours. |
| `src/edittab.{h,cpp,ui}` | The Optima input type, compile-as-you-type, error display, the Generated assembly window. |
| `src/assembler/program.h` | New `SourceType::Optima`. |
| `src/cli/clioptions.cpp`, `src/cli/clirunner.cpp` | `-t optima`. |
| `src/loaddialog.cpp`, `src/savedialog.{h,cpp}`, `src/mainwindow.cpp` | `.opt` files and the examples menu. |
| `src/settingsdialog.cpp`, `src/ripessettings.{h,cpp}` | The two settings. |
| `src/editor/codeeditor.cpp` | Picks the Optima highlighter. |
| `CMakeLists.txt`, `examples/` | `OPTIMA_DIR` option, example programs. |

Two small fixes to existing Ripes behaviour were needed and are part of the
patch: the input type of the previous session is now restored at start-up,
and opening the Settings window no longer switches the editor back to
Assembly unless C input was selected.

## Building for Windows on GitHub (no local tools needed)

`.github/workflows/build-ripes.yml` builds the patched Ripes on GitHub's
Windows machines and packages it with the Optima compiler.

1. Create a GitHub repository and upload the contents of this folder to it
   (`optc.py`, `optima/`, `ripes/`, `.github/`, `.gitattributes`, ...).
2. Open the repository's **Actions** tab, choose **Build patched Ripes
   (Windows)** and click **Run workflow**.
3. When the run finishes (about 15-25 minutes), download
   **Ripes-optima-windows** from the run's **Artifacts** section, unzip it and
   start `Ripes-optima\Ripes.exe`. Python 3 must be installed.
