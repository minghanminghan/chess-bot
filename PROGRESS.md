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

## Phase 4 — Hot-path Micro-optimisations (COMPLETE)

Baseline (CPU, 4×64ch net, 50 sims, 50 moves): **1.74 ms/sim** → after: **1.114 ms/sim** (**−36%**)

| # | Opt | File | Status | Description |
|---|-----|------|--------|-------------|
| 18 | C1 | `ChessBoard.py` | ✅ | Cache Zobrist hash — `string_representation()` O(n_pieces) → O(1) |
| 19 | C2 | `ChessBoard.py` | ✅ | Bitboard piece_planes — `piece_map()` Python loop → numpy unpackbits |

---



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
