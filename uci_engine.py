"""
UCI stdin/stdout engine wrapper for ChessBot-AZ.

Implements the UCI protocol so the bot can be used with cutechess-cli
or any UCI-compatible chess GUI.

Usage (pipe test):
    echo -e "uci\\nquit" | uv run python uci_engine.py --mcts-sims 5
    echo -e "uci\\nisready\\nposition startpos\\ngo movetime 100\\nquit" \\
        | uv run python uci_engine.py --mcts-sims 5

UCI flow:
    Host → uci           Engine ← id name / id author / uciok
    Host → isready       Engine ← readyok
    Host → position ...  Engine   (updates internal board)
    Host → go ...        Engine ← bestmove <uci>
    Host → quit          Engine   sys.exit(0)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chessbot.ChessBoard import ChessBoardState
from chessbot.ChessGame import ChessGame
from chessbot.ChessNNet import ChessNNet
from chessbot.ui_utils import action_to_uci
from alphazero_general.MCTS import MCTS
from utils import dotdict


def _log(msg: str) -> None:
    """Send an info string to the GUI (visible in engine output logs)."""
    print(f"info string {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="UCI engine wrapper for ChessBot-AZ")
    p.add_argument("--checkpoint-dir",   default="./checkpoints")
    p.add_argument("--checkpoint-file",  default="best.pth.tar")
    p.add_argument("--mcts-sims",        type=int, default=200)
    p.add_argument("--num-channels",     type=int, default=256)
    p.add_argument("--num-res-blocks",   type=int, default=20)
    cli = p.parse_args()

    args = dotdict({
        "numMCTSSims":     cli.mcts_sims,
        "cpuct":           1.0,
        "dirichlet_alpha": 0.0,
        "num_channels":    cli.num_channels,
        "num_res_blocks":  cli.num_res_blocks,
    })

    # Load model once at startup
    game = ChessGame()
    nnet = ChessNNet(game, args)
    ckpt = os.path.join(cli.checkpoint_dir, cli.checkpoint_file)
    if os.path.isfile(ckpt):
        nnet.load_checkpoint(cli.checkpoint_dir, cli.checkpoint_file)
    else:
        _log(f"Warning: checkpoint not found at {ckpt}; using random weights")

    # Mutable game state — reset on each 'position' command
    board_state = ChessBoardState()

    # ── Handlers ─────────────────────────────────────────────────────────────

    def handle_uci() -> None:
        print("id name ChessBot-AZ", flush=True)
        print("id author ChessBot", flush=True)
        print("uciok", flush=True)

    def handle_position(line: str) -> None:
        nonlocal board_state
        tokens = line.split()
        i = 1
        new_state = ChessBoardState()  # default: startpos
        if i < len(tokens) and tokens[i] == "fen":
            fen_parts = []
            i += 1
            while i < len(tokens) and tokens[i] != "moves":
                fen_parts.append(tokens[i])
                i += 1
            new_state.set_fen(" ".join(fen_parts))
        elif i < len(tokens) and tokens[i] == "startpos":
            i += 1

        if i < len(tokens) and tokens[i] == "moves":
            i += 1
            while i < len(tokens):
                new_state.push_uci(tokens[i])
                i += 1

        board_state = new_state

    def handle_go(_line: str) -> None:
        is_black = board_state.side_to_move() == -1
        cur_player = board_state.side_to_move()
        canonical = game.getCanonicalForm(board_state, cur_player)
        mcts = MCTS(game, nnet, args)
        probs = mcts.getActionProb(canonical, temp=0)
        action = int(np.argmax(probs))

        uci = action_to_uci(action, is_black=is_black)
        if uci is None:
            # Fallback: first valid action
            valid = board_state.valid_moves_mask()
            fallback = int(np.argmax(valid))
            uci = action_to_uci(fallback, is_black=is_black) or "0000"
            _log("Warning: top action not in table, using fallback")

        print(f"bestmove {uci}", flush=True)

    # ── Main UCI loop ─────────────────────────────────────────────────────────

    while True:
        try:
            line = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        if line == "uci":
            handle_uci()
        elif line == "isready":
            print("readyok", flush=True)
        elif line.startswith("position"):
            handle_position(line)
        elif line.startswith("go"):
            handle_go(line)
        elif line == "quit":
            sys.exit(0)
        # setoption / stop / ponderhit: silently ignored


if __name__ == "__main__":
    main()
