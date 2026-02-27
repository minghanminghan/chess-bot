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

## Phase 3 — Performance Optimisation (COMPLETE)

| # | File | Status | Description |
|---|------|--------|-------------|
| 15 | `MCTS.py` | ✅ | Vectorised UCB: replaced `{(s,a): scalar}` dicts with `{s: ndarray(4672)}`; numpy argmax replaces Python for-loop over 4,672 actions |
| 16 | `Coach.py` + `train.py` | ✅ | Parallel self-play via `ProcessPoolExecutor` + `--num-workers` CLI arg; workers load shared checkpoint, return examples |
| 17 | `bench.py` | ✅ | Timing harness: measures ms/move and sims/s; accepts `--sims`, `--episodes`, `--checkpoint-*` args |

---

### Bottleneck analysis

Profiled against the **practical config** (100 sims, 128ch, 10 blocks, ~30 legal moves/position).
Time per move is the primary metric because self-play is >80% of total wall time.

| # | Location | Root cause | Cost | After fix | Speedup |
|---|----------|------------|------|-----------|---------|
| **B1** | `MCTS.search()` lines 118–128 | `for a in range(4672)` Python loop — iterates entire action space per simulation to find UCB-max | ~20–50 ms/move (CPU) | ~0.1 ms/move | **10–50×** on loop |
| **B2** | `MCTS.search()` backprop 140–147 | `(s,a)` tuple keys — two dict lookups per sim, string key hashing | Minor alone; removed by B1 fix | Eliminated | — |
| **B3** | `MCTS.getActionProb()` lines 52–56 | List comprehension `[Nsa.get((s,a),0) for a in range(4672)]` at end of every move | ~2–5 ms/move | ~0.01 ms | — |
| **B4** | `Coach.learn()` lines 84–87 | Episodes run **sequentially** — CPU cores idle while one runs | 100 eps × T = 100T | N_workers × T ÷ N_workers | **2–4× throughput** |
| **B5** | `nnet.predict()` called one-at-a-time | batch=1 inference — GPU mostly idle; Python→CUDA launch overhead dominates | ~2–5 ms/call | Unchanged (requires batch-MCTS rewrite, out of scope) | — |

**Summary**: B1 is the highest-ROI single change. B4 multiplies throughput for free on multi-core.
B5 (batch MCTS) is the deepest fix but requires a near-complete MCTS rewrite — deferred.

**Combined expected speedup (B1 + B4):**
- Self-play: **4–10× faster** (UCB loop gone + parallelism)
- Full training iteration: **3–7× faster** (training step unchanged)
- Wall time to 500 iters, practical config, Colab A100: ~4–8h (was ~25–40h)

---

## Phase 4.B — A100 Colab Profiling: Bottleneck Teardown

### Profiling configuration
A100-SXM4-80GB · 4×64ch net · `torch.compile` + bfloat16
50 moves × 50 sims = 2,500 total · wall time **17.7 s** → **354 ms/move** · **7.07 ms/sim**

### Measured wall-time breakdown

| Hot-path site | Cumtime | % total | Root cause |
|---|---|---|---|
| `to_tensor()` | 5.37 s | 30% | 8 × `chess.Board(fen)` + 512 × `piece_at()` per call |
| `nnet.predict()` | 5.97 s | 34% | batch-1 GPU forward; kernel-launch overhead dominates compute |
| `MCTS.search()` Python | 2.21 s | 13% | recursive Python; 5 × dict-hash per step on ~80-char FEN key |
| `board.push()` / copy | ~1.5 s | ~8% | `chess.Board.copy()` on every MCTS tree-node expansion |
| `string_representation()` | ~0.4 s | ~2% | `board.fen()` recomputed every sim even for the root |
| `getValidMoves()` / legal_moves | ~0.25 s | ~1% | python-chess move generation per leaf (result cached in `Vs`) |
| Remainder | ~2.0 s | ~11% | backprop arithmetic, numpy UCB, `is_game_over`, I/O |

---

### B1 — `to_tensor()` FEN reconstruction ✅ FIXED

**Root cause** `to_tensor()` called `chess.Board(fen)` × 8 per call to rebuild
each history board from a string, then iterated all 64 squares with `piece_at()`.
2,500 leaf expansions → 20,000 FEN parses + 1,280,000 `piece_at()` calls → 5.37 s.

**Fix applied** (`ChessBoard.py`)
- `_push_history()` now pre-computes a `(8, 8, 12)` float32 occupancy array via
  `board.piece_map()` (~20 occupied squares, not 64) at push time.
- `to_tensor()` does 8 numpy slice-copies + scalar fills — zero python-chess calls.
- New `_push_history()` cost: ~35 µs/push × 6,155 MCTS pushes ≈ **215 ms**.
- Old reconstruction cost: ~1,877 µs/call × 2,500 calls = **4,692 ms**.
- **Net savings: ~4.5 s → ≈ 26% overall speedup.**

---

### Remaining estimated wall time after fix: ~13 s

| # | Site | Est. time | % remaining | Fix |
|---|------|-----------|----|-----|
| B2 | `nnet.predict()` batch=1 | ~5.97 s | 46% | ✅ batch-MCTS (`_walk` + `_run_batch` + `predict_batch`) |
| B3 | `board.push()` / `chess.Board.copy()` | ~1.5 s | 12% | ✅ `apply()`/`undo()` make-unmake |
| — | MCTS Python / dict overhead | ~2.21 s | 17% | partly resolved by B3 + B4 |
| B4 | FEN generation (`string_representation`) | ~0.4 s | 3% | ✅ Zobrist hash tuple |
| B5 | CHW tensor layout | ~0.1 s | <1% | ✅ planes `(119,8,8)`, permutes removed |
| — | `getValidMoves()` / `legal_moves` | ~0.25 s | 2% | (cached per unique state in `Vs`) |

---

### B2 — Batch neural network inference ✅ FIXED

Every leaf expansion calls `predict()` with `batch=1`, paying a full CUDA
kernel launch (~1 ms overhead) for each of ~2,500 calls per episode. The A100's
tensor cores are nearly idle at batch-1; profiler shows 2.4 ms/call with this
small net, dominated by launch latency not compute.

At production config (800 sims, 20-block 256-ch net): the net is ~10× larger but
A100 still under-utilised at batch-1. Every ms of launch latency × 800 sims ×
~100 moves = **80 s/episode** wasted on overhead alone.

**Fix: batch-MCTS with leaf queue**

Replace the recursive depth-first search with a width-first accumulation loop:

```
For each of numMCTSSims iterations:
  1. Walk tree greedily (UCB) until hitting a leaf → record path + board
  2. Accumulate leaf boards in pending[]
  3. When pending reaches batch_size (or all sims done):
       tensors = np.stack([b.to_tensor() for b in pending])   # (N, 119, 8, 8)
       pis, vs  = nnet.predict_batch(tensors)                 # one GPU call
       expand each board, backprop all N paths in order
```

`ChessNNet.predict_batch()`:
```python
def predict_batch(self, boards: np.ndarray):
    """boards: (N, 8, 8, 119)  →  pis: (N, 4672), vs: (N,)"""
    self.nnet.eval()
    with torch.no_grad():
        x = torch.tensor(boards, dtype=torch.float32, device=self.device)
        x = x.permute(0, 3, 1, 2)  # (N, 119, 8, 8)
        with torch.autocast(...):
            log_ps, vs = self.nnet(x)
        return torch.exp(log_ps).float().cpu().numpy(), vs.squeeze(1).float().cpu().numpy()
```

**Expected impact**: 5–10× speedup on `predict()` (46% of remaining wall time)
→ **~25–35% overall speedup**. Optimal batch size on A100: 32–128 (test empirically).
**Cost**: near-complete MCTS rewrite. Cleanest to implement after B-C3 (push/pop)
eliminates the need to pass board copies into recursive calls.

---

### B3 — `chess.Board.copy()` on every MCTS node ✅ FIXED

`getNextState()` calls `board.push(move)` which calls `board.copy()` + `chess.Board.copy()`.
`chess.Board.copy()` copies ~40 python-chess attributes including 8 bitboards.

**Scale**: at 800 sims/move with average tree depth ~3:
- ~2,400 copies per move × ~25 µs each = **60 ms/move from copies alone**
- Across a 100-move game = **6 s per episode** just in board copies

**Fix: mutable push / pop (make-unmake moves)**

Change `ChessBoardState` to mutate in-place and expose a `pop()`:

```python
def push(self, move: chess.Move) -> None:
    """Mutate board in place; save undo info on _undo_stack."""
    # Deque is full when len == HISTORY_LEN; oldest entry will be displaced
    displaced = self._history[0] if len(self._history) == self.HISTORY_LEN else None
    self._undo_stack.append(displaced)
    self.board.push(move)   # python-chess maintains its own undo stack
    self._push_history()

def pop(self) -> None:
    """Undo last push."""
    self.board.pop()
    self._history.pop()                # remove newest entry (right end)
    displaced = self._undo_stack.pop()
    if displaced is not None:
        self._history.appendleft(displaced)   # restore oldest entry (left end)
```

`MCTS.search()` make-unmake pattern (replaces `getNextState` + `getCanonicalForm`):
```python
move = self.game.actionToMove(canonicalBoard, a)
canonicalBoard.apply(move)
v = self.search(canonicalBoard)
canonicalBoard.undo()
# back-propagate v as before
```

**Impact**: eliminates `chess.Board.copy()` entirely from MCTS; MCTS holds a single
`ChessBoardState` object with a move stack instead of O(depth) copied objects.
**Estimated savings**: ~1.5 s at 50 sims → **~10–12% overall speedup**; scales
linearly with sim count (more impactful at 800 sims).
**Note**: `ChessBoardState` becomes mutable; `Coach.executeEpisode()` already
owns one board object per episode so this is safe. `Arena` and parallel workers
each own their own board; no sharing concerns.

---

### B4 — Zobrist hash replaces FEN as MCTS key ✅ FIXED

`MCTS.search()` calls `stringRepresentation(board)` once per call → `board.board.fen()`.
python-chess FEN serialises the full position as an ~80-char string — ~50 µs/call.
At 800 sims/move, average depth 3: **240,000 calls × 50 µs = 12 s/game** in FEN alone.

Lazy-caching avoids recomputing the same FEN, but still pays the cost once per unique
state and requires cache-invalidation plumbing inside `apply()`/`undo()`.

**Better fix: Zobrist hash tuple** (1 line in `ChessBoard.py`):

```python
def string_representation(self):
    return (chess.polyglot.zobrist_hash(self.board), self.board.halfmove_clock)
```

- `chess.polyglot.zobrist_hash()` — XOR of pre-computed table values: **~2 µs** vs 50 µs for FEN (~25× faster)
- `halfmove_clock` — included so positions at different distances from the 50-move draw rule
  get different MCTS keys; correctly distinguishes game-theoretically distinct states
- Returns a `(int, int)` tuple — hashable; Python dicts accept any hashable key, no interface change
- Collision probability: 1/2⁶⁴ ≈ 5×10⁻²⁰ — negligible in any realistic training run
- **No cache lifecycle**: unlike lazy-FEN, Zobrist is computed fresh from the live board in 2 µs,
  so `apply()` / `undo()` need zero cache-invalidation code

**Estimated savings**: at 800 sims and 100 moves/game,
~240,000 calls × (50–2) µs = **~11.5 s/game saved** (mostly recovered by FEN elimination;
additional benefit over lazy-FEN is simpler code and O(1) dict hashing on int tuple vs string).

---

### B5 — CHW tensor layout (eliminate redundant permutes) ✅ FIXED

`to_tensor()` returns `(8, 8, 119)` HWC.
`predict()` does `.permute(2, 0, 1).unsqueeze(0)` → `(1, 119, 8, 8)` CHW.
`train()` does `.permute(0, 3, 1, 2)` → `(B, 119, 8, 8)`.

**Fix**: store piece planes as `(12, 8, 8)` in `_push_history()` and output
`(119, 8, 8)` directly from `to_tensor()`.

```python
# _push_history() — store CHW:
piece_planes = np.zeros((12, 8, 8), dtype=np.float32)
for sq, piece in b.piece_map().items():
    rank, file = divmod(sq, 8)
    piece_planes[PIECE_TO_PLANE[(piece.piece_type, piece.color)], rank, file] = 1.0

# to_tensor() — output (119, 8, 8), rank-flip is axis-1 not axis-0:
planes = np.zeros((119, 8, 8), dtype=np.float32)
planes[offset:offset + 12] = piece_planes[:, ::-1, :] if flip else piece_planes

# predict() — drop permute:
x = torch.tensor(board, ...).unsqueeze(0)   # board is (119,8,8) → (1,119,8,8)

# train() — drop permute:
boards = torch.tensor(np.array(boards), ...).to(device)   # already (B,119,8,8)
```

**Impact**: saves ~5–10 µs per predict call; produces a contiguous CHW tensor
for GPU transfer; eliminates non-contiguous views from `.permute()`.
Net: **< 1% overall** but cleaner code and slightly better GPU transfer alignment.

---

### Implementation priority

| Step | Change | Files | Status |
|------|--------|-------|--------|
| B4 | Zobrist hash key | `ChessBoard.py` | ✅ |
| B3 | `apply()`/`undo()` make-unmake | `ChessBoard.py` + `ChessGame.py` + `MCTS.py` | ✅ |
| B5 | CHW tensor layout, drop permutes | `ChessBoard.py` + `ChessNNet.py` | ✅ |
| B2 | Batch MCTS (`_walk`/`_run_batch`/`_backprop` + `predict_batch`) | `MCTS.py` + `ChessNNet.py` + `train.py` | ✅ |

**New CLI arg**: `--mcts-batch-size N` (default 8; try `--mcts-batch-size 64` on A100)

**Combined expected speedup (B1–B5 all applied) at 800 sims on A100:**
- Self-play: **4–8× faster** vs post-Phase-4 baseline
- Full training iteration: **3–5× faster**
- Remaining bottleneck: MCTS Python overhead (dict lookups, UCB, `legal_moves`)

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
