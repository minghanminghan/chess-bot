"""
AlphaZero Chess — 100k self-play training run.

Approximate step count:
  numIters=1000 iterations × numEps=100 episodes/iter
  = 100,000 self-play episodes  (~100k "steps")

Usage:
  uv run python train.py

Checkpoints are saved to ./checkpoints/ after each accepted model update.
Training examples are saved to ./checkpoints/checkpoint_<iter>.examples.

To resume from a checkpoint:
  Set RESUME=True below, ensure best.pth.tar exists in ./checkpoints/.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from utils import dotdict
from chessbot.ChessGame import ChessGame
from chessbot.ChessNNet import ChessNNet
from Coach import Coach

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

    # ── Persistence ──────────────────────────────────────────────────────────
    'checkpoint': './checkpoints',
})

# ── Resume flag ──────────────────────────────────────────────────────────────
RESUME = False   # set to True to load best.pth.tar and continue training

# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    print("AlphaZero Chess — training run")
    print(f"  Iterations : {args.numIters}")
    print(f"  Episodes   : {args.numEps} per iteration  ({args.numIters * args.numEps:,} total)")
    print(f"  MCTS sims  : {args.numMCTSSims}")
    print(f"  Network    : {args.num_res_blocks} res-blocks × {args.num_channels} channels")
    print(f"  Checkpoint : {args.checkpoint}")
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
