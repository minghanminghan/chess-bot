"""
Bottleneck profiler for the chess-bot MCTS pipeline.

Mirrors the profiling cell from train_colab.ipynb but runs locally
with the Phase 5 C++ cboard extension.

Usage:
    uv run python profiler.py
"""

import cProfile, pstats, io, time, sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from chessbot.ChessGame import ChessGame
from chessbot.ChessNNet import ChessNNet
from alphazero_general.MCTS import MCTS
from utils import dotdict

PROFILE_SIMS  = 50   # MCTS sims per move
PROFILE_MOVES = 50   # moves to profile (caps episode length)
NET_CH, NET_BL = 64, 4  # tiny network so compile is fast
BATCH_SIZE = 8

cfg = dotdict(dict(numMCTSSims=PROFILE_SIMS, cpuct=1.0,
                   dirichlet_alpha=0.3, dirichlet_eps=0.25,
                   tempThreshold=30, num_channels=NET_CH, num_res_blocks=NET_BL,
                   mcts_batch_size=BATCH_SIZE))

game  = ChessGame()
nnet  = ChessNNet(game, cfg)
board = game.getInitBoard()

HAS_BATCH     = hasattr(nnet, 'predict_batch')
HAS_APPLYUNDO = hasattr(board, 'apply')
print(f"Code version:  predict_batch={HAS_BATCH}  apply/undo={HAS_APPLYUNDO}")
print(f"Hot path:      {'batch MCTS (new)' if HAS_BATCH else 'per-sim MCTS (old)'}")
if not HAS_BATCH:
    print("  ⚠ predict_batch missing")
print()

# ── 1. Manual hot-path timing ──────────────────────────────────────────────────
tensor = board.to_tensor(canonical=True)
# Pick the first legal action (e2e4 at startpos) for apply/undo timing
action = int(np.argmax(board.valid_moves_mask()))
N = 2000

print(f"Timing hot paths ({N} calls each) ...")

# ── Leaf-expansion costs ───────────────────────────────────────────────────────
if HAS_BATCH:
    batch = np.stack([tensor] * BATCH_SIZE)
    t = time.perf_counter()
    for _ in range(N): nnet.predict_batch(batch)
    t_infer = (time.perf_counter() - t) / N * 1000 / BATCH_SIZE  # amortised per leaf
    infer_label = f"predict_batch(N={BATCH_SIZE}) / {BATCH_SIZE}  [per leaf]"
else:
    t = time.perf_counter()
    for _ in range(N): nnet.predict(tensor)
    t_infer = (time.perf_counter() - t) / N * 1000
    infer_label = "nnet.predict()  [per leaf, batch=1]"

t = time.perf_counter()
for _ in range(N): board.to_tensor(canonical=True)
t_tensor = (time.perf_counter() - t) / N * 1000

t = time.perf_counter()
for _ in range(N): game.getValidMoves(board, 1)
t_valids = (time.perf_counter() - t) / N * 1000

# ── Walk-step costs ────────────────────────────────────────────────────────────
t = time.perf_counter()
for _ in range(N): game.stringRepresentation(board)
t_strrepr = (time.perf_counter() - t) / N * 1000

t = time.perf_counter()
for _ in range(N): game.getGameEnded(board, 1)
t_ended = (time.perf_counter() - t) / N * 1000

if HAS_APPLYUNDO:
    t = time.perf_counter()
    for _ in range(N):
        board.apply(action)
        board.undo()
    t_walk = (time.perf_counter() - t) / N * 1000
    walk_label = "apply() + undo()  [per step]"
else:
    t = time.perf_counter()
    for _ in range(N): game.getNextState(board, 1, action)
    t_walk = (time.perf_counter() - t) / N * 1000
    walk_label = "getNextState()  [per step, old]"

# ── 2. cProfile over a full short episode ─────────────────────────────────────
mcts  = MCTS(game, nnet, cfg)
board = game.getInitBoard()
cur   = 1

pr = cProfile.Profile()
pr.enable()
t0 = time.perf_counter()

moves = 0
while True:
    pi         = mcts.getActionProb(board, temp=1)
    action     = np.random.choice(len(pi), p=pi)
    board, cur = game.getNextState(board, cur, action)
    moves += 1
    if game.getGameEnded(board, cur) != 0 or moves >= PROFILE_MOVES:
        break

elapsed = time.perf_counter() - t0
pr.disable()

total_sims  = moves * PROFILE_SIMS
ms_per_move = elapsed / moves * 1000
ms_per_sim  = elapsed / total_sims * 1000

# ── 3. Report ─────────────────────────────────────────────────────────────────
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
print(f"\n{'='*66}")
print(f"  GPU: {gpu_name}   net: {NET_BL}×{NET_CH}ch   "
      f"{moves} moves × {PROFILE_SIMS} sims = {total_sims:,} total")
print(f"{'='*66}")
print(f"  Wall time : {elapsed:.1f}s  →  {ms_per_move:.0f} ms/move  |  {ms_per_sim:.2f} ms/sim")

print(f"\n  ── Leaf-expansion costs (paid per new unvisited node) ───────────────")
for name, cost in [(infer_label, t_infer), ("board.to_tensor()", t_tensor), ("getValidMoves()", t_valids)]:
    bar = '█' * int(cost / ms_per_sim * 30)
    print(f"  {name:44s}  {cost:7.3f} ms  {bar}")

print(f"\n  ── Walk-step costs (paid at every tree-traversal step) ──────────────")
for name, cost in [(walk_label, t_walk), ("stringRepresentation()", t_strrepr), ("getGameEnded()", t_ended)]:
    bar = '█' * int(cost / ms_per_sim * 30)
    print(f"  {name:44s}  {cost:7.3f} ms  {bar}")

print(f"\n  ── cProfile top-20 by cumulative time ──────────────────────────────")
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(20)
for line in s.getvalue().splitlines():
    if line.strip():
        print(' ', line)
