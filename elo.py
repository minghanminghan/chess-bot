"""
ELO evaluator — runs cutechess-cli matches between ChessBot-AZ and an opponent.

Streams live match output and prints results. Checkpoints go to --pgn-out.

Stockfish UCI_Elo presets (approximate skill level):
    1320  Beginner
    1500  Casual club player
    1800  Intermediate club player
    2000  Strong amateur
    2500  Master level
    3190  Full Stockfish strength

Usage:
    uv run python elo.py --dry-run                      # print command, no execution
    uv run python elo.py --engine2-elo 1320 --games 20  # vs Stockfish@1320, 20 games
    uv run python elo.py --games 100 --sprt             # full run with SPRT stopping
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def build_command(args: argparse.Namespace, pgn_out: str) -> list[str]:
    """Construct the cutechess-cli command as a list of strings."""
    cmd = [args.cutechess_path]

    # Engine 1: our bot
    cmd += [
        "-engine",
        f"cmd={args.engine1_cmd}",
        "dir=.",
        "proto=uci",
        "name=ChessBot-AZ",
    ]

    # Engine 2: opponent
    e2 = [
        f"cmd={args.engine2_cmd}",
        "proto=uci",
        f"name={args.engine2_name}",
    ]
    if args.engine2_elo:
        e2 += [
            "option.UCI_LimitStrength=true",
            f"option.UCI_Elo={args.engine2_elo}",
        ]
    cmd += ["-engine"] + e2

    # Match settings
    cmd += ["-each", f"tc={args.tc}"]
    cmd += ["-games", str(args.games)]
    cmd += ["-concurrency", str(args.concurrency)]
    cmd += ["-pgnout", pgn_out]

    if args.sprt:
        cmd += ["-sprt", "elo0=0", "elo1=5", "alpha=0.05", "beta=0.05"]

    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description="Run ELO evaluation via cutechess-cli")
    p.add_argument("--engine1-cmd",    default="uv run python uci_engine.py",
                   help="Command to launch ChessBot-AZ")
    p.add_argument("--engine2-cmd",    default="stockfish",
                   help="Opponent engine command")
    p.add_argument("--engine2-name",   default="Stockfish",
                   help="Opponent display name")
    p.add_argument("--engine2-elo",    type=int, default=None,
                   help="Stockfish UCI_Elo (1320–3190); omit for full strength")
    p.add_argument("--games",          type=int, default=100)
    p.add_argument("--tc",             default="40/60",
                   help="Time control (e.g. '40/60' = 40 moves in 60 s)")
    p.add_argument("--concurrency",    type=int, default=1)
    p.add_argument("--pgn-out",        default=None,
                   help="PGN output path (auto-timestamped if omitted)")
    p.add_argument("--sprt",           action="store_true",
                   help="Enable SPRT early-stopping rule")
    p.add_argument("--cutechess-path", default="cutechess-cli",
                   help="cutechess-cli binary name or full path")
    p.add_argument("--dry-run",        action="store_true",
                   help="Print the command without executing it")
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pgn_out = args.pgn_out or os.path.join("matches", f"match_{ts}.pgn")

    cmd = build_command(args, pgn_out)
    print("Command:", " ".join(cmd))

    if args.dry_run:
        return

    pgn_dir = os.path.dirname(pgn_out)
    if pgn_dir:
        os.makedirs(pgn_dir, exist_ok=True)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        print(f"\nMatch finished. Exit code: {proc.returncode}")
    except FileNotFoundError:
        print(
            f"Error: '{args.cutechess_path}' not found.\n"
            "Install cutechess-cli and ensure it is on your PATH.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
