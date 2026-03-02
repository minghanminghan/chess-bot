"""
AlphaZero Chess — 100k self-play training run.

Approximate step count:
  numIters=1000 iterations × numEps=100 episodes/iter
  = 100,000 self-play episodes  (~100k "steps")

Usage:
  uv run python train.py                          # fresh run, local checkpoints
  uv run python train.py --resume                 # resume from best.pth.tar
  uv run python train.py --checkpoint-dir /path   # e.g. Google Drive path in Colab
  uv run python train.py --num-iters 5 --num-eps 5 --mcts-sims 25 \
      --num-channels 64 --num-res-blocks 4        # fast scale-down test

Checkpoints are saved to ./checkpoints/ (or --checkpoint-dir) after each accepted
model update. Training examples are saved as checkpoint_<iter>.examples.
"""

import argparse
import os
import sys
import torch
sys.path.insert(0, os.path.dirname(__file__))

from utils import dotdict
from chessbot.ChessGame import ChessGame
from chessbot.ChessNNet import ChessNNet
from alphazero_general.Coach import Coach

# ── Hyperparameters ──────────────────────────────────────────────────────────

args = dotdict({
    # ── Self-play ────────────────────────────────────────────────────────────
    'numIters':     1000,   # training iterations
    'numEps':       100,    # self-play episodes per iteration  → 100k total

    # ── MCTS ─────────────────────────────────────────────────────────────────
    'numMCTSSims':  800,    # simulations per move (AlphaZero: 800)
    'cpuct':        1.0,    # UCB exploration constant
    'dirichlet_alpha': 0.3, # Dirichlet noise concentration (AlphaZero chess: 0.3)
    'dirichlet_eps':   0.25,# Dirichlet noise weight at root

    # ── Temperature ──────────────────────────────────────────────────────────
    'tempThreshold': 30,    # moves before switching τ → 0

    # ── Training ─────────────────────────────────────────────────────────────
    'lr':           0.001,  # Adam learning rate
    'l2_reg':       1e-4,   # L2 weight regularisation (c in the paper)
    'epochs':       10,     # gradient epochs per training call
    'batch_size':   512,    # mini-batch size

    # ── Network architecture ─────────────────────────────────────────────────
    'num_channels':   256,  # ResNet filter count (AlphaZero: 256)
    'num_res_blocks':  20,  # residual tower depth (AlphaZero: 20)

    # ── Arena / model acceptance ─────────────────────────────────────────────
    'arenaCompare':      40,    # games per evaluation round
    'updateThreshold':   0.55,  # min win-rate to accept new model

    # ── Example history ──────────────────────────────────────────────────────
    'numItersForTrainExamplesHistory': 20,  # keep last 20 iters of examples
    'maxlenOfQueue': 200_000,               # (informational; enforced via history)

    # ── Parallelism ───────────────────────────────────────────────────────────
    'num_workers':    1,        # parallel self-play workers (1 = sequential)
    'mcts_batch_size': 8,       # leaves per GPU call inside MCTS (B2 opt)

    # ── Persistence ──────────────────────────────────────────────────────────
    'checkpoint': './checkpoints',
})

# ── CLI overrides (for Colab / command-line use) ─────────────────────────────
_cli = argparse.ArgumentParser(add_help=True)
_cli.add_argument('--checkpoint-dir',  default=None,
                  help='Checkpoint directory (e.g. /content/drive/MyDrive/chess-bot)')
_cli.add_argument('--resume',          action='store_true',
                  help='Resume from best.pth.tar and saved examples')
_cli.add_argument('--num-iters',       type=int, default=None)
_cli.add_argument('--num-eps',         type=int, default=None)
_cli.add_argument('--mcts-sims',       type=int, default=None)
_cli.add_argument('--num-channels',    type=int, default=None)
_cli.add_argument('--num-res-blocks',  type=int, default=None)
_cli.add_argument('--num-workers',      type=int, default=1,
                  help='Parallel self-play workers (1 = sequential)')
_cli.add_argument('--mcts-batch-size', type=int, default=None,
                  help='Leaves per GPU call in MCTS (default 8; try 32–128 on A100)')
_parsed = _cli.parse_args()

if _parsed.checkpoint_dir:  args.checkpoint     = _parsed.checkpoint_dir
if _parsed.num_iters:       args.numIters        = _parsed.num_iters
if _parsed.num_eps:         args.numEps          = _parsed.num_eps
if _parsed.mcts_sims:       args.numMCTSSims     = _parsed.mcts_sims
if _parsed.num_channels:    args.num_channels    = _parsed.num_channels
if _parsed.num_res_blocks:  args.num_res_blocks  = _parsed.num_res_blocks
if _parsed.num_workers:       args.num_workers      = _parsed.num_workers
if _parsed.mcts_batch_size:   args.mcts_batch_size  = _parsed.mcts_batch_size

RESUME = _parsed.resume

# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    torch.set_float32_matmul_precision('high')   # enable TF32 for H100

    print("AlphaZero Chess — training run")
    print(f"  Iterations : {args.numIters}")
    print(f"  Episodes   : {args.numEps} per iteration  ({args.numIters * args.numEps:,} total)")
    print(f"  MCTS sims  : {args.numMCTSSims}")
    print(f"  Network    : {args.num_res_blocks} res-blocks × {args.num_channels} channels")
    print(f"  Workers    : {args.num_workers}")
    print(f"  MCTS batch : {args.mcts_batch_size}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print()

    game = ChessGame()
    nnet = ChessNNet(game, args)

    if RESUME:
        ckpt = os.path.join(args.checkpoint, 'best.pth.tar')
        if os.path.isfile(ckpt):
            print(f"Resuming from {ckpt}")
            nnet.load_checkpoint(args.checkpoint, 'best.pth.tar')
        else:
            print(f"Warning: no checkpoint found at {ckpt}, starting fresh.")

    coach = Coach(game, nnet, args)

    if RESUME:
        coach.loadTrainExamples()

    coach.learn()


if __name__ == '__main__':
    main()
