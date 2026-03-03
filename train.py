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
model update. Training examples are saved every --save-examples-every iterations.

WSL note: if your project lives on /mnt/c/..., pass --checkpoint-dir ~/chess-bot-checkpoints
to avoid writing through the slow Windows FS layer.
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
    'lr':           0.01,   # initial learning rate (first lr_schedule tier)
    'l2_reg':       1e-4,   # L2 weight regularisation (c in the paper)
    'epochs':       10,     # gradient epochs per training call
    'batch_size':   1024,   # mini-batch size

    # ── LR schedule — iteration → lr (applied at start of that iteration) ──
    # Mirrors AlphaZero paper: high lr early, decay as training matures.
    # Keys are iteration numbers (1-based); the highest key ≤ current iter wins.
    'lr_schedule':  {1: 0.01, 300: 0.001, 700: 0.0001},

    # ── Network architecture ─────────────────────────────────────────────────
    'num_channels':   256,  # ResNet filter count (AlphaZero: 256)
    'num_res_blocks':  20,  # residual tower depth (AlphaZero: 20)

    # ── Arena / model acceptance ─────────────────────────────────────────────
    # 40 games: σ≈7.9% at p=0.5, sufficient to detect consistent improvement.
    # 100 games adds little signal vs ~2.5× the wall-clock cost.
    'arenaCompare':      40,    # games per evaluation round
    'updateThreshold':   0.55,  # min win-rate to accept new model

    # ── Example history ──────────────────────────────────────────────────────
    # Memory budget: maxlenOfQueue × 48 KB ≈ peak RAM for the replay buffer.
    #   20 000 examples →  ~1 GB    (tight, minimum viable)
    #   50 000 examples →  ~2.4 GB  (comfortable on 8 GB systems)
    #   200 000 examples → ~9.6 GB  (original default — only for 32+ GB RAM)
    # numItersForTrainExamplesHistory is a secondary cap (oldest iteration dropped
    # first); maxlenOfQueue is the hard ceiling enforced by total example count.
    'numItersForTrainExamplesHistory': 10,
    'maxlenOfQueue': 50_000,

    # Save the full examples pickle every N iterations (not every iteration).
    # Avoids a 500 MB–3 GB disk write on every iteration; crash recovery still
    # works because the model checkpoint is saved every iteration.
    'save_examples_every': 5,

    # ── Parallelism ───────────────────────────────────────────────────────────
    'num_workers':    1,        # parallel self-play workers (1 = sequential)
    'mcts_batch_size': 64,      # leaves per GPU call inside MCTS

    # ── Persistence ──────────────────────────────────────────────────────────
    'checkpoint': './checkpoints',
})

# ── CLI overrides (for Colab / command-line use) ─────────────────────────────
_cli = argparse.ArgumentParser(add_help=True)
_cli.add_argument('--checkpoint-dir',  default=None,
                  help='Checkpoint directory (default: ./checkpoints). '
                       'In WSL, use ~/chess-bot-checkpoints to avoid slow /mnt/ I/O.')
_cli.add_argument('--resume',          action='store_true',
                  help='Resume from best.pth.tar and saved examples')
_cli.add_argument('--num-iters',       type=int, default=None)
_cli.add_argument('--num-eps',         type=int, default=None)
_cli.add_argument('--mcts-sims',       type=int, default=None)
_cli.add_argument('--num-channels',    type=int, default=None)
_cli.add_argument('--num-res-blocks',  type=int, default=None)
_cli.add_argument('--num-workers',     type=int, default=None,
                  help='Parallel self-play workers (1 = sequential)')
_cli.add_argument('--mcts-batch-size', type=int, default=None,
                  help='Leaves per GPU call in MCTS (default 64; try 128–256 on A100)')
_cli.add_argument('--save-examples-every', type=int, default=None,
                  help='Write training-examples pickle every N iterations (default 5)')
_cli.add_argument('--max-queue',           type=int, default=None,
                  help='Hard cap on total training examples kept in RAM '
                       '(default 50000 ≈ 2.4 GB; ~48 KB per example)')
_parsed = _cli.parse_args()

if _parsed.checkpoint_dir:      args.checkpoint          = _parsed.checkpoint_dir
if _parsed.num_iters:            args.numIters            = _parsed.num_iters
if _parsed.num_eps:              args.numEps              = _parsed.num_eps
if _parsed.mcts_sims:            args.numMCTSSims         = _parsed.mcts_sims
if _parsed.num_channels:         args.num_channels        = _parsed.num_channels
if _parsed.num_res_blocks:       args.num_res_blocks      = _parsed.num_res_blocks
if _parsed.num_workers is not None:     args.num_workers      = _parsed.num_workers
if _parsed.mcts_batch_size:      args.mcts_batch_size     = _parsed.mcts_batch_size
if _parsed.save_examples_every:  args.save_examples_every = _parsed.save_examples_every
if _parsed.max_queue:            args.maxlenOfQueue       = _parsed.max_queue

RESUME = _parsed.resume

# ── Entry point ──────────────────────────────────────────────────────────────

def _warn_if_slow_checkpoint():
    """Warn when running in WSL with checkpoints on the Windows FS (/mnt/)."""
    try:
        with open('/proc/version') as f:
            if 'microsoft' not in f.read().lower():
                return
    except OSError:
        return  # not Linux / not WSL
    resolved = os.path.abspath(args.checkpoint)
    if resolved.startswith('/mnt/'):
        print(f"WARNING: checkpoint dir '{resolved}' is on the Windows FS (/mnt/).")
        print("         Writes go through the NTFS driver and will be 3-5× slower.")
        print("         Pass --checkpoint-dir ~/chess-bot-checkpoints for native speed.")
        print()


def main():
    torch.set_float32_matmul_precision('high')   # enable TF32 on Ampere+

    _warn_if_slow_checkpoint()

    print("AlphaZero Chess — training run")
    print(f"  Iterations : {args.numIters}")
    print(f"  Episodes   : {args.numEps} per iteration  ({args.numIters * args.numEps:,} total)")
    print(f"  MCTS sims  : {args.numMCTSSims}")
    print(f"  Network    : {args.num_res_blocks} res-blocks × {args.num_channels} channels")
    print(f"  Workers    : {args.num_workers}")
    print(f"  MCTS batch : {args.mcts_batch_size}")
    print(f"  Arena      : {args.arenaCompare} games")
    print(f"  Replay buf : {args.numItersForTrainExamplesHistory} iters / "
          f"{args.maxlenOfQueue:,} examples max  "
          f"(≈{args.maxlenOfQueue * 48 // 1024} MB peak, "
          f"save every {args.save_examples_every} iters)")
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
