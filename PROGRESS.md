# AlphaZero Chess Bot — Progress

Reference: [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) | Paper: https://arxiv.org/pdf/1712.01815

---

## File Layout

```
chess-bot/
├── chessbot/               # renamed from chess/ (avoids shadowing python-chess)
│   ├── __init__.py
│   ├── ChessGame.py        # Game interface: canonical flip, _flip_move, getNextState
│   ├── ChessBoard.py       # 8×8×119 tensor, ACTION_SIZE=4672, move_to_action/action_to_move
│   └── ChessNNet.py        # ResNet (20 res blocks, 256 ch), policy + value heads
├── Game.py                 # abstract base class
├── NeuralNet.py            # abstract NN interface
├── MCTS.py                 # UCB tree search + Dirichlet noise at root
├── Arena.py                # model evaluation: (player1, player2, game) → (w, l, d)
├── Coach.py                # self-play training loop
├── utils.py                # AverageMeter, dotdict
├── train.py                # training entry point; CLI args for Colab use
├── train_colab.ipynb       # Google Colab notebook (clone → Drive mount → train)
├── tui.py                  # Textual TUI: replay PGN + play vs bot
├── uci_engine.py           # UCI protocol wrapper for cutechess-cli
├── elo.py                  # cutechess-cli ELO evaluation helper
├── bench.py                # [planned] timing harness: sims/s before/after optimisations
└── pyproject.toml          # uv: torch, python-chess, numpy, tqdm, textual
```

### Key Design Decisions
- **Input**: 8×8×119 — 8 history steps × 14 planes + 7 scalar planes
- **Network**: 20 residual blocks, 256 filters; policy → 4672 log-probs; value → tanh scalar
- **MCTS**: UCB with c_puct=1.0, Dirichlet α=0.3 noise at root, 800 sims/move
- **Loss**: `(z−v)² − π^T log(p) + c‖θ‖²`, c=1e-4
- **Canonical form**: spatial flip only (rows mirrored for black), piece colour planes unchanged
- **Action space**: 64×73=4672; always white-perspective; `_flip_move` converts black moves

---

Future consideration: full flat-array MCTS + Numba CUDA
Rewrite board state as int arrays, MCTS tree as flat numpy/cuda arrays, move gen as @cuda.jit(device=True). Integrate NN inference as a batched host call between CUDA kernel stages. Reference: https://github.com/pklesk/mcts_numba_cuda. Blocked by: incompatibility with AlphaZero NN-at-leaf evaluation model; chess move generation complexity (4672 actions vs 42 for Connect4); GPU memory cost of 4672-wide tree arrays. Revisit after Phase 5.

---

## Phase 5 — MidnightMoveGen C++ Extension (PLANNED)

**Goal:** Replace python-chess on the training hot path with a pybind11-wrapped C++ board that exposes a deep interface (action indices, tensors, game state) directly. Reduces python-chess to an optional UI-only dependency (`tui.py` only).

**Reference:** https://github.com/archishou/MidnightMoveGen — single-header legal move generator, ~420M NPS, has `has_repetition()`, incremental Zobrist, `set_fen`, make/unmake.

| # | File | Status | Description |
|---|------|--------|-------------|
| 20   | `chessbot/cboard/move_gen.h` | ✅ | Copy MidnightMoveGen `release/move_gen.h` verbatim |
| 20.5 | `chessbot/cboard/move_gen.h` | ✅ | Add runtime-dispatch wrappers to `Position` class (see Step 1.5 in guide) |
| 21   | `chessbot/cboard/board.cpp` | ✅ | C++ wrapper class + pybind11 bindings (see implementation guide below) |
| 22 | `chessbot/cboard/CMakeLists.txt` | ✅ | Build config: pybind11 + C++20, produces `cboard.cp313-win_amd64.pyd` |
| 23 | `chessbot/ChessBoard.py` | ✅ | Replace with thin shim that imports from `cboard`; `ChessBoardState = cboard.Position` |
| 24 | `chessbot/ChessGame.py` | ✅ | Remove `import chess`; adapted `getValidMoves`, `getNextState`, `getGameEnded`; removed `actionToMove` and `_flip_move` |
| 25 | `MCTS.py` | ✅ | Changed `board.apply(game.actionToMove(board, a))` → `board.apply(a)` |
| 26 | `uci_engine.py` | ✅ | Remove `import chess`; use `board_state.push_uci()`, `board_state.side_to_move()` |
| 27 | `pyproject.toml` | ✅ | Move `python-chess` to optional `[ui]` extra; `pybind11` already in deps |
| 28 | `bench.py` | ✅ | Run before/after to confirm speedup |

---

## Phase 5 — Implementation Guide

This section is written for a fresh coding agent implementing Phase 5 from scratch.

### Overview

The C++ extension module (`chessbot/cboard`) replaces `chess.Board` (python-chess) as the board backend for all training code. The module exposes a single class `Position` with a deep interface: it handles move-make/unmake, 8-step history, feature tensor construction, legal move enumeration in canonical action-index space, game termination, and FEN/UCI parsing — everything except PGN and SAN (those remain in `tui.py` with python-chess).

### Directory layout after Phase 5

```
chessbot/
├── cboard/
│   ├── move_gen.h          # MidnightMoveGen header, copied verbatim
│   ├── board.cpp           # wrapper class + pybind11 module definition
│   └── CMakeLists.txt      # build config
├── ChessBoard.py           # thin shim: imports from cboard, re-exports constants
├── ChessGame.py            # no chess imports; adapted for cboard API
├── ChessNNet.py            # unchanged
└── __init__.py             # unchanged
```

### Step 1 — Build infrastructure

Use `scikit-build-core` (preferred) or `setuptools` with `pybind11`. The module name must be `cboard` and land in `chessbot/` so `from chessbot.cboard import Position` works.

`CMakeLists.txt` (inside `chessbot/cboard/`):
```cmake
cmake_minimum_required(VERSION 3.15)
project(cboard)
set(CMAKE_CXX_STANDARD 17)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(cboard board.cpp)
target_include_directories(cboard PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(cboard PRIVATE -O3 -march=native)
```

`pyproject.toml` additions:
```toml
[build-system]
requires = ["scikit-build-core", "pybind11"]
build-backend = "scikit_build_core.build"

[project.optional-dependencies]
ui = ["python-chess"]
```

**Verify:** `python -c "from chessbot.cboard import Position, ACTION_SIZE; print(ACTION_SIZE)"` prints `4672` with no import errors. The `.so`/`.pyd` file exists under `chessbot/`.

### Step 1.5 — Modify move_gen.h

After copying `move_gen.h` verbatim, add the following methods to the `Position` class. All are purely additive — no existing code is changed, so Midnight's internal move generation and all its templates remain intact.

These additions eliminate every `if (pos_.side == WHITE) ... else ...` template-dispatch block from `board.cpp`, except for `valid_moves_mask()` where one dispatch is unavoidable (handled via a private template method on `BoardWrapper` — see Step 6).

#### 1. `play(Move)` and `undo(Move)` — runtime-dispatch make/unmake

```cpp
// Add inside the Position class, after the existing play<C>/undo<C> templates.

inline void play(Move m) {
    if (side == WHITE) play<WHITE>(m);
    else               play<BLACK>(m);
}

inline void undo(Move m) {
    // side has not yet been restored; it reflects the player TO MOVE (the one
    // who did NOT make the last move). Dispatch on the opponent who did.
    if (side == WHITE) undo<BLACK>(m);
    else               undo<WHITE>(m);
}
```

#### 2. `in_check()` — missing from Midnight, needed for `result()`

Midnight detects check internally inside `MoveList` but exposes no public predicate. Add:

```cpp
[[nodiscard]] inline bool in_check() const {
    if (side == WHITE) {
        Square king_sq = lsb(pieces[WHITE_KING]);
        return attackers_of(king_sq, occupancy()) & occupancy<BLACK>();
    } else {
        Square king_sq = lsb(pieces[BLACK_KING]);
        return attackers_of(king_sq, occupancy()) & occupancy<WHITE>();
    }
}
```

`attackers_of(Square, Bitboard)` already exists on `Position`. `pieces[WHITE_KING]` / `pieces[BLACK_KING]` index the piece bitboard array directly using the `Piece` enum values — verify the exact enum values in `types.h` before building. `occupancy<WHITE/BLACK>()` returns all pieces of that colour OR'd together.

#### 3. `legal_move_count()` — avoids exposing MoveList type to board.cpp

```cpp
[[nodiscard]] inline int legal_move_count() const {
    if (side == WHITE) return static_cast<int>(MoveList<WHITE, ALL>(const_cast<Position&>(*this)).size());
    return                    static_cast<int>(MoveList<BLACK, ALL>(const_cast<Position&>(*this)).size());
}
```

The `const_cast` is required because `MoveList` takes a non-const `Position&` (it caches shared data internally). This is safe since `MoveList` does not mutate the position.

#### 4. `piece_bb(Color, PieceType)` — runtime bitboard access for tensor construction

```cpp
[[nodiscard]] inline Bitboard piece_bb(Color c, PieceType pt) const {
    return pieces[make_piece(pt, c)];
}
```

If `make_piece(pt, c)` does not exist as a free function, substitute `Piece(c * 8 + pt)` — verify against the `Piece` enum in `types.h` (WHITE pieces are 0–5, BLACK pieces start at 8 in Midnight's layout).

#### 5. `castling_rights_ks(Color)` and `castling_rights_qs(Color)` — runtime castling queries

```cpp
[[nodiscard]] inline bool castling_rights_ks(Color c) const {
    return c == WHITE ? king_and_oo_rook_not_moved<WHITE>()
                      : king_and_oo_rook_not_moved<BLACK>();
}
[[nodiscard]] inline bool castling_rights_qs(Color c) const {
    return c == WHITE ? king_and_ooo_rook_not_moved<WHITE>()
                      : king_and_ooo_rook_not_moved<BLACK>();
}
```

**Verify:**
- `move_gen.h` still compiles cleanly as a standalone header (`g++ -std=c++17 -c move_gen.h` produces no errors or warnings).
- From a C++ test harness (can reuse Midnight's `perft.cpp` pattern): construct `Position` from startpos FEN; call `p.play(m)` and `p.undo(m)` for several moves and confirm `p.hash()` round-trips correctly.
- `p.in_check()` returns `false` at startpos; returns `true` after the sequence `1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6 4.Qxf7` (scholar's mate, black king is in check at the final position — confirm via python-chess).
- `p.legal_move_count()` returns 20 at startpos.
- `p.piece_bb(WHITE, PAWN)` at startpos returns the same bitboard as `p.occupancy<WHITE, PAWN>()` — assert equality.
- `p.castling_rights_ks(WHITE)` and `p.castling_rights_qs(WHITE)` both return `true` at startpos; both return `false` after the white king moves to e2 and back (once moved, `from_to` records it, rights are permanently lost even if king returns).
- Full perft(5) still passes: 4,865,609 nodes from startpos. This confirms the additions did not disturb move generation.

### Step 2 — C++ wrapper class

`board.cpp` defines class `BoardWrapper` that owns:
- `Midnight::Position pos_` — the board state
- `std::vector<Midnight::Move> move_stack_` — for no-argument `undo()` (Midnight's `undo<C>(Move)` requires the move back)
- A fixed-size ring buffer of 8 history frames for `to_tensor()`
- `int fullmove_number_` — tracked separately (Midnight exposes `game_ply` but not fullmove)

```cpp
#include "move_gen.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <vector>
namespace py = pybind11;
using namespace Midnight;

static constexpr int ACTION_SIZE = 4672;

// Built once at module import. Maps action index → (from_sq, to_sq, promo).
// from_sq/to_sq are in WHITE-perspective (canonical) coordinates.
struct ActionEntry { int from_sq, to_sq, promo; };  // promo: 0=none,N,B,R,Q
static ActionEntry IDX_TO_ENTRY[ACTION_SIZE];
// Reverse map: packed key → action index. Key = from_sq*64*8 + to_sq*8 + promo_code.
static int ENTRY_TO_IDX[64 * 64 * 8];  // -1 if invalid

void build_action_tables();  // mirrors Python _build_action_maps() — see below
```

**Action table construction** (`build_action_tables()`):

Mirror `_build_action_maps()` from `ChessBoard.py` exactly, but store `(from_sq, to_sq, promo_code)` tuples instead of `chess.Move` objects. Square convention is identical (sq = file + rank*8). Promo codes: 0=none, 1=knight, 2=bishop, 3=rook, 4=queen. Queen promotions (slots 0-55 from rank 6) use promo_code=4. Underpromo slots 64-72 use codes 1-3.

**Verify:** After `build_action_tables()`, assert:
- Exactly 4672 entries in `IDX_TO_ENTRY` with no duplicate `(from_sq, to_sq, promo)` triples.
- `ENTRY_TO_IDX` round-trips: for every valid idx, `ENTRY_TO_IDX[pack(IDX_TO_ENTRY[idx])] == idx`.
- Cross-check against Python: for each entry in Python's `MOVE_TO_IDX`, confirm the C++ table maps the same `(from_sq, to_sq, promo)` to the same action index.

### Step 3 — `apply(int action_idx)` and `undo()`

```cpp
void apply(int action_idx) {
    // 1. Decode canonical (white-perspective) from/to from table.
    const ActionEntry& e = IDX_TO_ENTRY[action_idx];
    int from_c = e.from_sq, to_c = e.to_sq;

    // 2. If black's turn, flip squares back to real board coordinates.
    int from_r = (pos_.side == BLACK) ? (from_c ^ 56) : from_c;
    int to_r   = (pos_.side == BLACK) ? (to_c   ^ 56) : to_c;

    // 3. Find matching move in legal move list (needed for MoveType flags).
    Move found = find_legal_move(from_r, to_r, e.promo);  // O(~30 iters)

    // 4. Push to internal stacks, play, record new history frame.
    move_stack_.push_back(found);
    push_undo_history_entry();   // saves displaced history frame if any
    pos_.play(found);            // Step 1.5: runtime-dispatch, no if/else needed
    if (pos_.side == WHITE) fullmove_number_++;  // side flipped after play; increment if now white (black just moved)
    push_history_frame();
}

void undo() {
    Move m = move_stack_.back(); move_stack_.pop_back();
    bool was_black_move = (pos_.side == WHITE);  // white to move = black made the last move
    pop_history_frame();         // restores ring buffer
    pos_.undo(m);                // Step 1.5: runtime-dispatch, no if/else needed
    if (was_black_move) fullmove_number_--;
}
```

`find_legal_move(from_r, to_r, promo)` iterates `MoveList<WHITE/BLACK, ALL>(pos_)` and returns the first move whose `from()==from_r && to()==to_r` and whose promotion piece matches `promo`. This is O(legal move count ≈ 30) and does NOT require a hash map. Note: `find_legal_move` itself still needs an internal if/else for `MoveList<C>` — this is the one unavoidable template dispatch in `board.cpp` besides `fill_mask_<C>()` (Step 6).

**Verify:**
- From startpos, `apply(idx_e2e4); undo()`: `string_representation()` matches initial hash.
- Apply 10 moves then undo 10 times: hash and `side_to_move()` match the original startpos values.
- Apply the Fool's mate sequence (4 moves) then undo all 4: hash matches startpos.
- `apply(idx_e2e4)` on startpos: `side_to_move()` returns -1 (black to move), `fullmove_number()` returns 1.
- `apply(idx_e7e5)` after e2e4: `side_to_move()` returns 1, `fullmove_number()` returns 2.

### Step 4 — History ring buffer

The ring buffer stores 8 `HistoryFrame` entries. Each frame holds:
- `float piece_planes[12][8][8]` — bit-unpacked occupancy at that ply
- `bool rep1, rep2` — repetition flags at that ply

Layout: `history_frames_[8]`, `history_start_` (index of oldest), `history_size_` (0–8).

`push_history_frame()` (called inside `apply()` AFTER `pos_.play()`):
```
slot = (history_start_ + history_size_) % 8
if history_size_ == 8: history_start_ = (history_start_ + 1) % 8  (evict oldest)
else: history_size_++
Compute frame at slot: 12 piece planes from bitboards + rep1/rep2 from pos_.has_repetition()
```

The undo stack entry stores the evicted frame (if any) so `pop_history_frame()` can restore it:
```cpp
struct UndoHistEntry { HistoryFrame evicted; bool had_eviction; };
std::vector<UndoHistEntry> undo_hist_stack_;
```

`pop_history_frame()` (called inside `undo()` BEFORE `pos_.undo()`):
```
history_size_--   (drop newest)
if (undo_hist_stack_.back().had_eviction):
    history_start_ = (history_start_ - 1 + 8) % 8
    history_frames_[history_start_] = evicted_frame
    history_size_++
undo_hist_stack_.pop_back()
```

**Verify:**
- After startpos construction: `history_size_ == 1`, `history_start_ == 0`.
- After 7 more `apply()` calls: `history_size_ == 8`.
- After one more `apply()` (9th total): `history_size_ == 8` still; the oldest frame is gone.
- After `undo()` from that 9th state: `history_size_ == 8` again and the oldest frame is restored (expose a test accessor or verify indirectly via `to_tensor()` nonzero plane count).
- Apply 9 moves, undo 9 times: `history_size_ == 1`, piece planes in frame 0 match startpos.

### Step 5 — `to_tensor(bool canonical)`

Returns a `py::array_t<float>` of shape `{119, 8, 8}` (CHW).

```
bool flip = canonical && (pos_.side == BLACK)
Planes 0..111: iterate history oldest→newest (t=0..7):
    offset = t * 14
    src = history_frames_[(history_start_ + t) % 8]  (or zeros if t >= history_size_)
    planes[offset..offset+11] = src.piece_planes  (flipped on rank axis if flip)
    if src.rep1: planes[offset+12] = 1.0 (broadcast to 8×8)
    if src.rep2: planes[offset+13] = 1.0
Planes 112..118 (scalar, broadcast to 8×8):
    [112] = (pos_.side == WHITE) ? 1.0 : 0.0
    [113] = fullmove_number_ / 500.0
    [114] = pos_.castling_rights_ks(WHITE)  // Step 1.5: replaces king_and_oo_rook_not_moved<WHITE>()
    [115] = pos_.castling_rights_qs(WHITE)  // Step 1.5: replaces king_and_ooo_rook_not_moved<WHITE>()
    [116] = pos_.castling_rights_ks(BLACK)
    [117] = pos_.castling_rights_qs(BLACK)
    [118] = pos_.fifty_move_rule() / 100.0
```

Piece planes from bitboards — mirrors `_push_history` in current `ChessBoard.py`. Uses `pos_.piece_bb(c, pt)` from Step 1.5 to avoid 12 separate template instantiations:
```cpp
// Plane ordering: white P=0, N=1, B=2, R=3, Q=4, K=5, black P=6…K=11
// Matches current PIECE_TO_PLANE dict exactly.
static constexpr PieceType PIECE_ORDER[6] = {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};
for (int c = 0; c < 2; c++) {
    for (int pt = 0; pt < 6; pt++) {
        Bitboard bb = pos_.piece_bb(Color(c), PIECE_ORDER[pt]);
        // Unpack 64 bits into float[8][8]: bit k → [k/8][k%8]
        // If flip: mirror rank axis ([r][f] → [7-r][f])
        unpack_bitboard(bb, frame.piece_planes[c * 6 + pt], flip);
    }
}
```

**Verify:**
- Shape is `(119, 8, 8)`, dtype `float32`.
- Startpos, white to move: `tensor[112]` is all 1s; `tensor[113]` is all `1/500`; `tensor[114:118]` all 1s; `tensor[118]` all 0s.
- Startpos plane 0 (white pawns): nonzero at rows 1, all 8 files — i.e. `tensor[0, 1, :].sum() == 8.0`, all other rows zero.
- Startpos plane 5 (white king): nonzero only at `[0, 0, 4]` (rank 0, file 4 = e1).
- After `apply(e2e4)` (black to move), `to_tensor(canonical=True)`: `tensor[112]` is all 0s (black's perspective); black pawn plane should show pawns at rank 6 from black's view (row 1 in flipped coordinates).
- **Cross-check against Python:** run 1000 random positions through both old Python `ChessBoardState.to_tensor()` and new C++ `Position.to_tensor()`; `np.max(np.abs(py_tensor - cpp_tensor)) < 1e-6` for every position.

### Step 6 — `valid_moves_mask()`

Returns `py::array_t<float>` shape `{ACTION_SIZE}`. Handles canonical flip for black internally.

`valid_moves_mask()` is the one place where the `MoveList<C>` template dispatch cannot be fully hidden behind Step 1.5 additions — the loop body must execute inside the branch where `C` is a compile-time constant so the `MoveList<C>` constructor is correctly instantiated. The solution (Option 2) is a **private template method** on `BoardWrapper` that factors the loop body into a single definition, keeping `flip` as a `constexpr` so the compiler fully specialises both instantiations.

```cpp
// Private template method — one definition, no code duplication.
// flip is constexpr: the compiler generates two fully-specialised loop bodies
// (one with ^0 for white, one with ^56 for black) and eliminates the XOR entirely.
template<Color C>
void fill_mask_(float* data) const {
    constexpr int flip = (C == BLACK) ? 56 : 0;
    for (Move m : MoveList<C, ALL>(const_cast<Position&>(pos_))) {
        int from_c = m.from() ^ flip;
        int to_c   = m.to()   ^ flip;
        int promo  = promo_code(m);
        int idx    = ENTRY_TO_IDX[from_c * 512 + to_c * 8 + promo];
        if (idx >= 0) data[idx] = 1.0f;
    }
}

// Public method exposed to Python.
py::array_t<float> valid_moves_mask() {
    auto mask = py::array_t<float>(ACTION_SIZE);
    std::fill(mask.mutable_data(), mask.mutable_data() + ACTION_SIZE, 0.0f);
    if (pos_.side == WHITE) fill_mask_<WHITE>(mask.mutable_data());
    else                    fill_mask_<BLACK>(mask.mutable_data());
    return mask;
}
```

`promo_code(Move m)` is a local helper that maps the move's `MoveType` to the promo integer used in `ENTRY_TO_IDX` (0=none, 1=knight, 2=bishop, 3=rook, 4=queen). Extract from `m.type()` using a switch or lookup table over Midnight's `MoveType` enum values.

After this, `ChessGame.getValidMoves()` becomes a single call: `return board.valid_moves_mask()` for both colors. The separate black-flip path in the current Python code is eliminated.

**Verify:**
- Startpos: `mask.sum() == 20.0` (exactly 20 legal moves).
- After `apply(e2e4)` (black to move): `mask.sum() == 20.0` (black's 20 replies in canonical space).
- Fool's mate position after 1.f3 e5 2.g4 (white to move): `mask.sum() == 29.0` (standard count at that position — verify against python-chess `len(list(board.legal_moves))`).
- **Cross-check against Python:** for 500 random positions, `cpp_mask.sum() == len(list(py_board.legal_moves))` and the set of nonzero indices matches the Python `valid_moves_mask()` output exactly.

### Step 7 — `is_game_over()` and `result()`

Midnight has no single `is_game_over()`. Compose using Step 1.5 additions (`pos_.legal_move_count()` and `pos_.in_check()`) — no template dispatch needed in `board.cpp`:

```cpp
bool is_game_over() {
    if (pos_.fifty_move_rule() >= 100) return true;
    if (pos_.has_repetition(THREE_FOLD)) return true;
    return pos_.legal_move_count() == 0;  // Step 1.5: no MoveList<C> boilerplate
}

float result() {
    // Returns value from current player's perspective.
    if (pos_.fifty_move_rule() >= 100) return 1e-4f;
    if (pos_.has_repetition(THREE_FOLD)) return 1e-4f;
    if (pos_.legal_move_count() > 0) return 0.0f;   // game ongoing; Step 1.5
    if (pos_.in_check())             return -1.0f;   // mated; Step 1.5
    return 1e-4f;                                     // stalemate
}
```

Note: `result()` calls `legal_move_count()` once (one `MoveList` construction), then `in_check()` only if there are no legal moves (cheap bitboard scan). The `is_game_over()` call in MCTS is redundant with `result()` in practice — MCTS uses `getGameEnded()` which calls `result()` directly, so `is_game_over()` is only exposed for completeness.

**Verify:**
- Startpos: `is_game_over() == False`, `result() == 0.0`.
- Fool's mate — load FEN `"rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"` (white is mated): `is_game_over() == True`, `result() == -1.0` (white to move, in check, no legal moves).
- Stalemate FEN `"k7/8/1Q6/8/8/8/8/7K b - - 0 1"` (black stalemated): `is_game_over() == True`, `result() == 1e-4`.
- Fifty-move FEN with halfmove clock = 100, e.g. `"k7/8/8/8/8/8/8/7K w - - 100 1"`: `is_game_over() == True`, `result() == 1e-4`.
- Threefold repetition: apply a sequence of 8 moves that repeats a position three times; after the third repetition `has_repetition(THREE_FOLD)` must return true and `result() == 1e-4`.

### Step 8 — `string_representation()`, `copy()`, FEN/UCI helpers

```cpp
py::tuple string_representation() {
    return py::make_tuple(pos_.hash(), (int)pos_.fifty_move_rule());
}

BoardWrapper copy() {
    BoardWrapper b;
    b.pos_ = pos_;                    // Midnight Position has value semantics
    b.move_stack_ = move_stack_;
    b.history_frames_ = history_frames_;
    b.history_start_ = history_start_;
    b.history_size_ = history_size_;
    b.undo_hist_stack_ = undo_hist_stack_;
    b.fullmove_number_ = fullmove_number_;
    return b;
}

void set_fen(const std::string& fen) {
    pos_.set_fen(fen);
    // Reset all stacks and history
    move_stack_.clear(); undo_hist_stack_.clear();
    history_size_ = 0; history_start_ = 0;
    // Parse fullmove from FEN (6th field)
    fullmove_number_ = parse_fullmove_from_fen(fen);
    push_history_frame();
}

void push_uci(const std::string& uci) {
    // Parse "e2e4", "e7e8q" etc. into (from_sq, to_sq, promo_code)
    int from_r = parse_sq(uci.substr(0, 2));
    int to_r   = parse_sq(uci.substr(2, 2));
    int promo  = (uci.size() == 5) ? char_to_promo(uci[4]) : 0;
    // Convert to canonical, look up action index, then apply
    int from_c = (pos_.side == BLACK) ? (from_r ^ 56) : from_r;
    int to_c   = (pos_.side == BLACK) ? (to_r   ^ 56) : to_r;
    int idx = ENTRY_TO_IDX[from_c * 64 * 8 + to_c * 8 + promo];
    apply(idx);
}
```

`parse_sq("e2")` → `('e'-'a') + (2-1)*8 = 4 + 8 = 12`. Standard formula.

**Verify:**
- `Position().string_representation()` returns a 2-tuple of ints; the hash is nonzero and matches `chess.polyglot.zobrist_hash(chess.Board())` from python-chess.
- `copy()`: apply e2e4 on original; the copy's `string_representation()` still matches startpos hash.
- `set_fen(kiwipete)` where kiwipete = `"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"`: `valid_moves_mask().sum() == 48` (known perft position).
- `push_uci("e2e4")`: `string_representation()` hash matches `apply(idx_e2e4)` hash on a fresh board.
- `push_uci("e7e8q")` from an appropriate position: promotion handled correctly, `valid_moves_mask()` does not include the pre-promotion pawn move.

### Step 9 — pybind11 module definition (bottom of board.cpp)

```cpp
PYBIND11_MODULE(cboard, m) {
    build_action_tables();  // called once at import

    m.attr("ACTION_SIZE") = ACTION_SIZE;

    py::class_<BoardWrapper>(m, "Position")
        .def(py::init<>())                           // startpos
        .def(py::init<const std::string&>())         // from FEN
        .def("apply",                &BoardWrapper::apply)
        .def("undo",                 &BoardWrapper::undo)
        .def("copy",                 &BoardWrapper::copy)
        .def("valid_moves_mask",     &BoardWrapper::valid_moves_mask)
        .def("to_tensor",            &BoardWrapper::to_tensor,
             py::arg("canonical") = true)
        .def("string_representation",&BoardWrapper::string_representation)
        .def("is_game_over",         &BoardWrapper::is_game_over)
        .def("result",               &BoardWrapper::result)
        .def("side_to_move",         &BoardWrapper::side_to_move)  // returns int: 1=W, -1=B
        .def("hash",                 &BoardWrapper::hash)
        .def("halfmove_clock",       &BoardWrapper::halfmove_clock)
        .def("fullmove_number",      &BoardWrapper::fullmove_number)
        .def("set_fen",              &BoardWrapper::set_fen)
        .def("push_uci",             &BoardWrapper::push_uci);
}
```

**Verify:**
- `python -c "import chessbot.cboard"` completes without error.
- `python -c "from chessbot.cboard import Position, ACTION_SIZE; p = Position(); print(p.side_to_move(), p.halfmove_clock(), p.fullmove_number())"` prints `1 0 1`.
- All 14 methods listed in the `.def()` calls are callable from Python without `AttributeError`.
- `Position("invalid fen string")` raises a Python exception rather than segfaulting.

### Step 10 — ChessBoard.py rewrite

Replace the entire file with:

```python
# chessbot/ChessBoard.py
import numpy as np
from chessbot.cboard import Position as ChessBoardState, ACTION_SIZE

# Legacy function aliases — delegates to the C++ action table
from chessbot.cboard import move_to_action, action_to_move  # if exposed
# If not exposed as module functions, action_to_move/move_to_action are no
# longer needed externally (ChessGame calls board.apply(idx) directly).
```

The `ChessBoardState` name is preserved so all other imports (`from chessbot.ChessBoard import ChessBoardState`) continue to work unchanged.

**Verify:**
- `from chessbot.ChessBoard import ChessBoardState, ACTION_SIZE` succeeds; `ACTION_SIZE == 4672`.
- `ChessBoardState()` constructs a startpos board.
- `ChessBoardState().string_representation()` returns a 2-tuple.
- `python -c "import chessbot.ChessBoard"` produces no `ImportError` and no reference to `chess` in the module's namespace.

### Step 11 — ChessGame.py changes

Remove `import chess`. Affected methods:

**`_flip_move`** — currently constructs `chess.Move`. After migration this is pure integer arithmetic and no longer constructs any move object. It's only used in `getValidMoves` for the black path, which is now handled inside C++ `valid_moves_mask()`. `_flip_move` can be deleted or kept as:
```python
def _flip_move(from_sq, to_sq, promo=None):
    return (from_sq ^ 56, to_sq ^ 56, promo)
```
(Only `tui.py` and `uci_engine.py` import it; after those files are updated it can be removed.)

**`getNextState`** — remove `chess.BLACK` check (use `board.side_to_move() == -1`) and remove `_flip_move`. The action-to-move conversion and flip now happen inside `board.apply(action)`:
```python
def getNextState(self, board, player, action):
    next_board = board.copy()
    next_board.apply(action)
    return next_board, -player
```

**`getValidMoves`** — collapse to single call (C++ handles color dispatch):
```python
def getValidMoves(self, board, player):
    return board.valid_moves_mask()
```

**`getGameEnded`** — replace `board.is_game_over()` / `board.board.result()` with C++ methods:
```python
def getGameEnded(self, board, player):
    r = board.result()   # 0.0=ongoing, 1.0=current wins, -1.0=mated, 1e-4=draw
    if r == 0.0:
        return 0
    # result() is from current player's perspective; adjust for player convention
    return r if board.side_to_move() == player else -r
```
Note: `board.result()` returns from the perspective of the player TO MOVE. `getGameEnded(board, player)` is always called with `player == side_to_move` in practice, so the branch `board.side_to_move() != player` should never fire; the guard is defensive.

**`actionToMove`** — delete. MCTS._walk calls `board.apply(a)` directly after Step 12.

**`stringRepresentation`** — unchanged:
```python
def stringRepresentation(self, board):
    return board.string_representation()
```

**Verify:**
- `python -c "import chessbot.ChessGame"` succeeds; `"chess"` does not appear in `chessbot.ChessGame.__dict__`.
- `ChessGame().getValidMoves(ChessGame().getInitBoard(), 1).sum() == 20.0`.
- `ChessGame().getGameEnded(ChessGame().getInitBoard(), 1) == 0`.
- Fool's mate: construct the mated position via `getNextState` calls; `getGameEnded(board, player) == -1` for the player that was mated (white).
- 10-move random rollout (as in Phase 1 test 4) completes without error; `getGameEnded` eventually returns nonzero.

### Step 12 — MCTS.py change

One line in `_walk()` (line 190):
```python
# Before:
board.apply(self.game.actionToMove(board, a))
# After:
board.apply(a)
```

**Verify:**
- `bench.py --sims 50 --episodes 3` runs to completion; no exceptions.
- Policy vector at each step sums to ≈ 1.0 (within 1e-5).
- No probability mass on illegal actions: `(policy * (1 - valid_mask)).sum() < 1e-6` for every position sampled.
- Run the Arena smoke test (2 games, random vs random): `nwins + lwins + draws == 2`.

### Step 13 — uci_engine.py changes

Remove `import chess`. Replace:
- `chess.Board()` + `board.set_fen()` + `board.push_uci()` → `ChessBoardState()` + `.set_fen()` + `.push_uci()`
- `board.turn == chess.WHITE` → `board_state.side_to_move() == 1`
- `board_state.board.legal_moves` fallback → `board_state.valid_moves_mask()` (pick first nonzero index)

**Verify:**
- `echo "uci\nquit" | python uci_engine.py` outputs `id name`, `id author`, and `uciok` with no errors.
- `echo "uci\nisready\nposition startpos\ngo\nquit" | python uci_engine.py` outputs a `bestmove` that is a legal UCI move from startpos (e.g. `bestmove e2e4`).
- `echo "uci\nisready\nposition startpos moves e2e4 e7e5\ngo\nquit" | python uci_engine.py` outputs a legal bestmove from the 1.e4 e5 position.
- `python -c "import uci_engine"` completes; `"chess"` does not appear in `uci_engine.__dict__`.

### Step 14 — tui.py (no changes needed)

`tui.py` keeps `import chess` and `import chess.pgn`. `PlayApp` maintains a parallel `chess.Board` for display/SAN generation (it already calls `board.push(real_move)` separately from `game.getNextState()`). No changes required; python-chess remains a required dep for this file only.

**Verify:**
- `python -c "import tui"` succeeds (smoke test only; Textual not launched).
- Replay mode: `python tui.py --mode replay --pgn <any_pgn_file>` opens without error and steps through moves correctly.
- Play mode: `python tui.py --mode play --side white --mcts-sims 5` launches, bot makes a legal first move when it is its turn, and UCI input is accepted.

### Final integration verification

After all 14 steps are complete:
1. **Perft:** Build and run MidnightMoveGen's own `perft.cpp` standalone; startpos `perft(5)` == 4,865,609 and kiwipete `perft(4)` == 4,085,603. This validates the C++ move generator before any Python involvement.
2. **`to_tensor` equivalence:** Run 1000 random positions through both old Python `ChessBoardState.to_tensor()` and new C++ `Position.to_tensor()`; `np.max(np.abs(py - cpp)) < 1e-6` for all.
3. **`valid_moves_mask` equivalence:** For 500 random positions confirm count and set of nonzero indices match python-chess exactly.
4. **Full self-play smoke test:** `python train.py --num-iters 1 --num-eps 2 --mcts-sims 20` completes one iteration (self-play → train → arena) without error.
5. **`bench.py`:** Record ms/sim before (python-chess baseline from Phase 4) and after; expect >5× improvement.

---

## Phase 4 — Hot-path Micro-optimisations (COMPLETE)

Baseline (CPU, 4×64ch net, 50 sims, 50 moves): **1.74 ms/sim** → after: **1.114 ms/sim** (**−36%**)

| # | Opt | File | Status | Description |
|---|-----|------|--------|-------------|
| 18 | C1 | `ChessBoard.py` | ✅ | Cache Zobrist hash — `string_representation()` O(n_pieces) → O(1) |
| 19 | C2 | `ChessBoard.py` | ✅ | Bitboard piece_planes — `piece_map()` Python loop → numpy unpackbits |

---

## Phase 3 — Performance Optimisation (COMPLETE)

| # | File | Status | Description |
|---|------|--------|-------------|
| 15 | `MCTS.py` | ✅ | Vectorised UCB: replaced `{(s,a): scalar}` dicts with `{s: ndarray(4672)}`; numpy argmax replaces Python for-loop over 4,672 actions |
| 16 | `Coach.py` + `train.py` | ✅ | Parallel self-play via `ProcessPoolExecutor` + `--num-workers` CLI arg; workers load shared checkpoint, return examples |
| 17 | `bench.py` | ✅ | Timing harness: measures ms/move and sims/s; accepts `--sims`, `--episodes`, `--checkpoint-*` args |

---

## Phase 2 — Strength Testing (COMPLETE)

| # | File | Status | Verified by |
|---|------|--------|-------------|
| 10 | `tui.py` | ✅ | `import tui` OK; replay + play modes implemented |
| 11 | `uci_engine.py` | ✅ | `uci\nquit` → `uciok`; `position startpos\ngo` → `bestmove g1f3` (legal) |
| 12 | `elo.py` | ✅ | `import elo` OK; `--dry-run` prints cutechess-cli command |

### Colab training adaptation

| # | File | Status | Description |
|---|------|--------|-------------|
| 13 | `train.py` (edit) | ✅ | Added `argparse`: `--checkpoint-dir`, `--resume`, `--num-iters`, `--num-eps`, `--mcts-sims`, `--num-channels`, `--num-res-blocks` |
| 14 | `train_colab.ipynb` | ✅ | Colab notebook: git clone → Drive mount → GPU check → train → download checkpoint |

---

## Phase 1 — AlphaZero Core (COMPLETE)

| # | File | Status | Verified by |
|---|------|--------|-------------|
| 1 | `pyproject.toml` (uv setup) | ✅ | `import chess, torch, numpy` → all OK |
| 2 | `Game.py`, `NeuralNet.py`, `utils.py` | ✅ | imports clean, AverageMeter avg=2.0 |
| 3 | `chessbot/ChessBoard.py` | ✅ | shape (8,8,119), ACTION_SIZE=4672, 20 legal moves at start, Fool's mate −1 |
| 4 | `chessbot/ChessGame.py` | ✅ | 10-move random rollout, checkmate detection, string repr |
| 5 | `chessbot/ChessNNet.py` | ✅ | forward shapes, train step, checkpoint round-trip |
| 6 | `MCTS.py` | ✅ | 50 sims, policy sums to 1, zero mass on illegal moves |
| 7 | `Arena.py` | ✅ | 2-game smoke test, w+l+d==2 |
| 8 | `Coach.py` | ✅ | 1-iter loop (self-play→train→arena→accept/reject) |
| 9 | `train.py` | ✅ | 1000×100=100k config verified; NOT run |

---
## Hyperparameters (train.py)

| Param | Value | Note |
|-------|-------|------|
| numIters | 1000 | training iterations |
| numEps | 100 | self-play episodes per iter |
| tempThreshold | 30 | moves before τ→0 |
| updateThreshold | 0.55 | win-rate to accept new model |
| maxlenOfQueue | 200000 | max training examples kept |
| numMCTSSims | 800 | MCTS simulations per move |
| arenaCompare | 40 | arena games per evaluation |
| cpuct | 1.0 | UCB exploration constant |
| epochs | 10 | gradient steps per train call |
| batch_size | 512 | |
| num_channels | 256 | ResNet filter count |
| num_res_blocks | 20 | ResNet depth |
| l2_reg | 1e-4 | weight decay |
