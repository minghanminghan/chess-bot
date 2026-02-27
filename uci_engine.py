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

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chessbot.ChessBoard import ChessBoardState, action_to_move
from chessbot.ChessGame import ChessGame, _flip_move
from chessbot.ChessNNet import ChessNNet
from MCTS import MCTS
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
    cur_player = 1  # 1=white, -1=black

    # ── Handlers ─────────────────────────────────────────────────────────────

    def handle_uci() -> None:
        print("id name ChessBot-AZ", flush=True)
        print("id author ChessBot", flush=True)
        print("uciok", flush=True)

    def handle_position(line: str) -> None:
        nonlocal board_state, cur_player
        tokens = line.split()
        board = chess.Board()
        i = 1
        if i < len(tokens) and tokens[i] == "fen":
            fen_parts = []
            i += 1
            while i < len(tokens) and tokens[i] != "moves":
                fen_parts.append(tokens[i])
                i += 1
            board.set_fen(" ".join(fen_parts))
        elif i < len(tokens) and tokens[i] == "startpos":
            i += 1  # startpos — board already at starting position

        if i < len(tokens) and tokens[i] == "moves":
            i += 1
            while i < len(tokens):
                board.push_uci(tokens[i])
                i += 1

        board_state = ChessBoardState(board)
        cur_player = 1 if board.turn == chess.WHITE else -1

    def handle_go(_line: str) -> None:
        # Fresh MCTS per move (no tree reuse — simple and correct)
        canonical = game.getCanonicalForm(board_state, cur_player)
        mcts = MCTS(game, nnet, args)
        probs = mcts.getActionProb(canonical, temp=0)
        action = int(np.argmax(probs))

        raw_move = action_to_move(action)
        if raw_move is None:
            # Fallback: pick first legal move
            raw_move = next(iter(board_state.board.legal_moves))
            _log("Warning: action_to_move returned None, using first legal move")
            print(f"bestmove {raw_move.uci()}", flush=True)
            return

        real_move = _flip_move(raw_move) if board_state.board.turn == chess.BLACK else raw_move
        print(f"bestmove {real_move.uci()}", flush=True)

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
