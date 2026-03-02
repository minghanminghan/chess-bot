"""
Timing benchmark for MCTS + self-play.

Usage:
    uv run python bench.py                     # default: 3 episodes, 100 sims
    uv run python bench.py --episodes 5 --sims 200
    uv run python bench.py --checkpoint-dir ./checkpoints --checkpoint-file best.pth.tar
"""
import argparse
import os
import time
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes',        type=int, default=3)
    p.add_argument('--sims',            type=int, default=100)
    p.add_argument('--num-channels',    type=int, default=64)
    p.add_argument('--num-res-blocks',  type=int, default=4)
    p.add_argument('--checkpoint-dir',  default=None)
    p.add_argument('--checkpoint-file', default='best.pth.tar')
    args = p.parse_args()

    from chessbot.ChessGame import ChessGame
    from chessbot.ChessNNet import ChessNNet
    from alphazero_general.MCTS import MCTS
    from utils import dotdict

    cfg = dotdict(dict(numMCTSSims=args.sims, cpuct=1.0,
                       dirichlet_alpha=0.0, tempThreshold=30,
                       num_channels=args.num_channels, num_res_blocks=args.num_res_blocks))
    game = ChessGame()
    nnet = ChessNNet(game, cfg)
    if args.checkpoint_dir:
        path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
        if os.path.isfile(path):
            nnet.load_checkpoint(args.checkpoint_dir, args.checkpoint_file)

    total_moves = total_sims = 0
    t0 = time.perf_counter()

    for ep in range(args.episodes):
        mcts  = MCTS(game, nnet, cfg)
        board = game.getInitBoard()
        cur   = 1
        moves = 0
        while True:
            canonical = game.getCanonicalForm(board, cur)
            pi        = mcts.getActionProb(canonical, temp=1)
            action    = np.random.choice(len(pi), p=pi)
            board, cur = game.getNextState(board, cur, action)
            moves += 1
            total_sims += args.sims
            if game.getGameEnded(board, cur) != 0 or moves > 200:
                break
        total_moves += moves
        print(f"  Episode {ep+1}: {moves} moves")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal : {total_moves} moves | {total_sims:,} sims | {elapsed:.1f}s")
    print(f"Speed : {total_moves/elapsed:.1f} moves/s  |  {total_sims/elapsed:,.0f} sims/s")
    print(f"        {elapsed/total_moves*1000:.1f} ms/move  |  {elapsed/total_sims*1000:.2f} ms/sim")


if __name__ == '__main__':
    main()
