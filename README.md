# Chess Bot

AlphaZero-style chess engine. 20 residual blocks × 256 channels, C++ board backend (Midnight move generator via pybind11), batched MCTS with GPU inference.

## Building the C++ extension

The `chessbot.cboard` extension must be built before running training or inference.

### Windows (MinGW / Strawberry Perl)

```bat
mkdir build_cboard
cd build_cboard

cmake ../chessbot/cboard ^
  -G "MinGW Makefiles" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -Dpybind11_DIR="%CD%/../.venv/Lib/site-packages/pybind11/share/cmake/pybind11" ^
  -DCMAKE_INSTALL_PREFIX="../chessbot" ^
  -DPYTHON_EXECUTABLE=".venv/Scripts/python.exe"

cmake --build . --config Release
cmake --install .

:: Copy the .pyd and the required runtime DLL into the venv
copy ..\chessbot\cboard.cp313-win_amd64.pyd ..\venv\Lib\site-packages\chessbot\
copy ..\chessbot\libwinpthread-1.dll ..\venv\Lib\site-packages\chessbot\
```

> **Note:** `libgcc` and `libstdc++` are statically linked via `-static-libgcc -static-libstdc++`.
> `libwinpthread-1.dll` still needs to be present — copy it from your Strawberry Perl installation
> (e.g. `C:\Strawberry\c\bin\libwinpthread-1.dll`).

Or, using `uv run` for the cmake variables:

```bash
# Run these in Git Bash / PowerShell with uv on PATH
mkdir -p build_cboard && cd build_cboard

cmake ../chessbot/cboard \
  -G "MinGW Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(uv run python -m pybind11 --cmakedir)" \
  -DCMAKE_INSTALL_PREFIX="../chessbot" \
  -DPYTHON_EXECUTABLE="$(uv run python -c 'import sys; print(sys.executable)')"

cmake --build . --config Release
cmake --install .

cp ../chessbot/cboard.cp313-win_amd64.pyd \
   ../.venv/Lib/site-packages/chessbot/
```

---

### WSL (Linux)

```bash
cd ~/projects/chess-bot

mkdir -p build_cboard && cd build_cboard

cmake ../chessbot/cboard \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(uv run python -m pybind11 --cmakedir)" \
  -DCMAKE_INSTALL_PREFIX="../chessbot" \
  -DPYTHON_EXECUTABLE="$(uv run python -c 'import sys; print(sys.executable)')"

cmake --build . --config Release
cmake --install .

# Copy the .so into the venv (glob handles the cpython tag and Python version)
cp ../chessbot/cboard*.so \
   ../.venv/lib/python3.*/site-packages/chessbot/
```

> **Note:** On Linux the `-static-libgcc -static-libstdc++` flags in CMakeLists.txt
> statically link the GCC runtime, so no extra DLLs are needed.

---

## Training

```bash
uv run python train.py --num-iters 2000 --num-eps 200
```

Key defaults: 800 MCTS sims/move, batch size 64, 20 res-blocks × 256 channels, checkpoints saved to `./checkpoints/`.

## Running / inference

```bash
uv run python bench.py          # benchmark MCTS throughput
uv run python -m chessbot.ui    # (requires [ui] extras: uv sync --extra ui)
```
