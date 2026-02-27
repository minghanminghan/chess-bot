"""
ChessBot Textual TUI

Modes:
    replay  --pgn path/to/game.pgn        Step through a PGN file (← → arrows)
    play    [--side white|black] [opts]   Play live vs the AlphaZero bot

Usage:
    uv run python tui.py --mode replay --pgn game.pgn
    uv run python tui.py --mode play --mcts-sims 100 --side white
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

import chess
import chess.pgn
import numpy as np
from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Header, Footer, Static, Input, Button

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chessbot.ChessBoard import ChessBoardState, move_to_action, action_to_move
from chessbot.ChessGame import ChessGame, _flip_move
from chessbot.ChessNNet import ChessNNet
from MCTS import MCTS
from utils import dotdict


# ── Piece glyphs ─────────────────────────────────────────────────────────────

GLYPHS = {
    (chess.KING,   chess.WHITE): "♔",
    (chess.QUEEN,  chess.WHITE): "♕",
    (chess.ROOK,   chess.WHITE): "♖",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.PAWN,   chess.WHITE): "♙",
    (chess.KING,   chess.BLACK): "♚",
    (chess.QUEEN,  chess.BLACK): "♛",
    (chess.ROOK,   chess.BLACK): "♜",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.PAWN,   chess.BLACK): "♟",
}


def _board_text(
    board: chess.Board,
    last_move: "chess.Move | None" = None,
    flip: bool = False,
) -> Text:
    """Return a Rich Text object rendering the 8×8 board with coloured squares."""
    text = Text()
    rank_iter = range(7, -1, -1) if not flip else range(8)
    for rank in rank_iter:
        text.append(f" {rank + 1} ")
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            is_last = last_move is not None and sq in (
                last_move.from_square, last_move.to_square
            )
            is_light = (rank + file) % 2 == 1
            if is_last:
                style = "bold on dark_green"
            elif is_light:
                style = "on grey82"
            else:
                style = "on dark_goldenrod"
            glyph = GLYPHS.get((piece.piece_type, piece.color), "·") if piece else "·"
            text.append(f" {glyph} ", style=style)
        text.append("\n")
    file_row = "   " + "  ".join("abcdefgh" if not flip else "hgfedcba")
    text.append(file_row, style="dim")
    return text


# ── Shared board widget ───────────────────────────────────────────────────────

class BoardWidget(Static):
    DEFAULT_CSS = "BoardWidget { width: 30; padding: 0 1; }"

    def __init__(self, board: chess.Board, flip: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._board = board
        self._last_move: "chess.Move | None" = None
        self._flip = flip

    def on_mount(self) -> None:
        self._redraw()

    def set_board(self, board: chess.Board, last_move: "chess.Move | None" = None) -> None:
        self._board = board
        self._last_move = last_move
        self._redraw()

    def _redraw(self) -> None:
        self.update(_board_text(self._board, self._last_move, self._flip))


# ── Replay App ────────────────────────────────────────────────────────────────

class ReplayApp(App):
    CSS = """
    Screen { layout: vertical; }
    #main { height: 1fr; layout: horizontal; }
    BoardWidget { width: 30; padding: 0 1; }
    #info { width: 1fr; padding: 1; }
    #controls { height: 3; layout: horizontal; padding: 0 1; }
    #controls Button { margin: 0 1; }
    """
    BINDINGS = [
        Binding("left",  "prev_move", "Prev"),
        Binding("right", "next_move", "Next"),
        Binding("q",     "quit",      "Quit"),
    ]

    def __init__(self, pgn_path: str, **kwargs):
        super().__init__(**kwargs)
        with open(pgn_path) as f:
            game = chess.pgn.read_game(f)
        if game is None:
            raise ValueError(f"No game found in {pgn_path}")
        self._pgn_name = os.path.basename(pgn_path)
        self._headers = dict(game.headers)
        self._board = chess.Board()
        self._moves: list[chess.Move] = list(game.mainline_moves())
        self._move_idx = 0
        # Pre-compute SAN once
        tmp = chess.Board()
        self._sans: list[str] = []
        for m in self._moves:
            self._sans.append(tmp.san(m))
            tmp.push(m)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="main"):
            yield BoardWidget(self._board, id="board_widget")
            yield Static("", id="info")
        with Horizontal(id="controls"):
            yield Button("◀ Prev", id="btn_prev")
            yield Button("Next ▶", id="btn_next", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"Replay — {self._pgn_name}"
        self._update()

    def action_next_move(self) -> None:
        if self._move_idx < len(self._moves):
            self._board.push(self._moves[self._move_idx])
            self._move_idx += 1
            self._update()

    def action_prev_move(self) -> None:
        if self._move_idx > 0:
            self._board.pop()
            self._move_idx -= 1
            self._update()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_next":
            self.action_next_move()
        elif event.button.id == "btn_prev":
            self.action_prev_move()

    def _update(self) -> None:
        last = self._board.move_stack[-1] if self._board.move_stack else None
        self.query_one(BoardWidget).set_board(self._board, last)
        self.query_one("#info", Static).update(self._make_info())

    def _make_info(self) -> Text:
        w = self._headers.get("White", "?")
        b = self._headers.get("Black", "?")
        result = self._headers.get("Result", "*")
        idx = self._move_idx
        total = len(self._moves)

        t = Text()
        t.append("White:  ", style="bold"); t.append(f"{w}\n")
        t.append("Black:  ", style="bold"); t.append(f"{b}\n")
        t.append("Result: ", style="bold"); t.append(f"{result}\n\n")
        t.append(f"Move {idx} / {total}\n\n", style="bold")

        # Pair moves: "N. white_san black_san"
        pairs: list[str] = []
        for i in range(0, len(self._sans), 2):
            w_san = self._sans[i]
            b_san = self._sans[i + 1] if i + 1 < len(self._sans) else ""
            pairs.append(f"{i // 2 + 1}. {w_san} {b_san}".rstrip())

        cur_pair = max(0, (idx - 1) // 2)
        start = max(0, cur_pair - 4)
        for j, pair in enumerate(pairs[start: start + 10]):
            is_active = (start + j == cur_pair) and idx > 0
            if is_active:
                t.append("▶ ", style="bold cyan")
                t.append(f"{pair}\n", style="bold")
            else:
                t.append(f"  {pair}\n")
        return t


# ── Play App ──────────────────────────────────────────────────────────────────

class PlayApp(App):
    CSS = """
    Screen { layout: vertical; }
    #main { height: 1fr; layout: horizontal; }
    BoardWidget { width: 30; padding: 0 1; }
    #info { width: 1fr; padding: 1; }
    #input_row { height: 3; layout: horizontal; padding: 0 1; }
    #input_row Input { width: 22; }
    #input_row Button { margin: 0 1; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        game: ChessGame,
        nnet,
        mcts_args: dotdict,
        human_is_white: bool,
        pgn_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._game = game
        self._nnet = nnet
        self._mcts_args = mcts_args
        self._human_is_white = human_is_white
        self._pgn_dir = pgn_dir
        self._board_state = game.getInitBoard()
        self._cur_player = 1
        self._moves_played: list[chess.Move] = []
        self._game_over = False
        self._status = "Your turn" if human_is_white else "Bot thinking…"
        self._flip = not human_is_white

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="main"):
            yield BoardWidget(self._board_state.board, flip=self._flip, id="board_widget")
            yield Static("", id="info")
        with Horizontal(id="input_row"):
            yield Input(placeholder="UCI move (e.g. e2e4)", id="move_input")
            yield Button("Submit", id="btn_submit", variant="primary")
            yield Button("Resign", id="btn_resign", variant="error")
        yield Footer()

    async def on_mount(self) -> None:
        self.title = (
            f"ChessBot Play — You: {'White' if self._human_is_white else 'Black'}"
            f" | Sims: {self._mcts_args.numMCTSSims}"
        )
        self._update_display()
        if not self._human_is_white:
            await self._do_bot_move()

    # ── Bot move (async, MCTS in thread pool) ─────────────────────────────────

    async def _do_bot_move(self) -> None:
        self._status = "Bot thinking…"
        self._set_input_enabled(False)
        self._update_display()
        loop = asyncio.get_running_loop()
        move = await loop.run_in_executor(None, self._run_mcts_sync)
        if move is not None:
            self._moves_played.append(move)
        self._check_game_over()
        if not self._game_over:
            self._status = "Your turn"
        self._set_input_enabled(True)
        self._update_display()

    def _run_mcts_sync(self) -> "chess.Move | None":
        """Blocking — runs in thread pool. Returns the real-coordinate move."""
        canonical = self._game.getCanonicalForm(self._board_state, self._cur_player)
        mcts = MCTS(self._game, self._nnet, self._mcts_args)
        probs = mcts.getActionProb(canonical, temp=0)
        action = int(np.argmax(probs))
        raw_move = action_to_move(action)
        if raw_move is None:
            return None
        was_black = self._board_state.board.turn == chess.BLACK
        real_move = _flip_move(raw_move) if was_black else raw_move
        self._board_state, self._cur_player = self._game.getNextState(
            self._board_state, self._cur_player, action
        )
        return real_move

    # ── Human move ────────────────────────────────────────────────────────────

    def _submit_move(self, uci_str: str) -> None:
        if self._game_over:
            return
        uci_str = uci_str.strip().lower()
        if not uci_str:
            return

        try:
            move = chess.Move.from_uci(uci_str)
        except ValueError:
            self._status = f"Invalid UCI: {uci_str!r}"
            self._update_display()
            return

        board = self._board_state.board
        if move not in board.legal_moves:
            promo = chess.Move.from_uci(uci_str + "q")
            if promo in board.legal_moves:
                move = promo
            else:
                self._status = f"Illegal move: {uci_str}"
                self._update_display()
                return

        was_black = board.turn == chess.BLACK
        canonical_move = _flip_move(move) if was_black else move
        action = move_to_action(canonical_move)
        if action is None:
            self._status = f"Move not in action table: {uci_str}"
            self._update_display()
            return

        self._board_state, self._cur_player = self._game.getNextState(
            self._board_state, self._cur_player, action
        )
        self._moves_played.append(move)
        self._check_game_over()
        if not self._game_over:
            asyncio.ensure_future(self._do_bot_move())
        self._update_display()

    # ── Game end ──────────────────────────────────────────────────────────────

    def _check_game_over(self) -> None:
        board = self._board_state.board
        if not board.is_game_over():
            return
        self._game_over = True
        result = board.result()
        label = {"1-0": "White wins", "0-1": "Black wins"}.get(result, "Draw")
        self._status = f"Game over: {label} ({result})"
        pgn_path = self._save_pgn(result)
        self.notify(f"Saved: {pgn_path}")

    def _save_pgn(self, result: str) -> str:
        os.makedirs(self._pgn_dir, exist_ok=True)
        pgn_game = chess.pgn.Game()
        pgn_game.headers.update({
            "White": "Human" if self._human_is_white else "ChessBot-AZ",
            "Black": "ChessBot-AZ" if self._human_is_white else "Human",
            "Result": result,
        })
        node = pgn_game
        for m in self._moves_played:
            node = node.add_variation(m)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._pgn_dir, f"game_{ts}.pgn")
        with open(path, "w") as f:
            print(pgn_game, file=f)
        return path

    # ── UI events ─────────────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_submit":
            inp = self.query_one("#move_input", Input)
            self._submit_move(inp.value)
            inp.value = ""
        elif event.button.id == "btn_resign":
            if not self._game_over:
                self._game_over = True
                result = "0-1" if self._human_is_white else "1-0"
                winner = "Black" if self._human_is_white else "White"
                self._status = f"You resigned. {winner} wins."
                pgn_path = self._save_pgn(result)
                self.notify(f"Saved: {pgn_path}")
                self._update_display()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit_move(event.value)
        self.query_one("#move_input", Input).value = ""

    # ── Display helpers ───────────────────────────────────────────────────────

    def _set_input_enabled(self, enabled: bool) -> None:
        self.query_one("#move_input", Input).disabled = not enabled
        self.query_one("#btn_submit", Button).disabled = not enabled

    def _update_display(self) -> None:
        board = self._board_state.board
        last = board.move_stack[-1] if board.move_stack else None
        self.query_one(BoardWidget).set_board(board, last)
        self.query_one("#info", Static).update(self._make_info())

    def _make_info(self) -> Text:
        board = self._board_state.board
        t = Text()
        t.append("You:    ", style="bold")
        t.append(f"{'White' if self._human_is_white else 'Black'}\n")
        t.append("Bot:    ", style="bold")
        t.append(f"{'Black' if self._human_is_white else 'White'}\n")
        t.append("Sims:   ", style="bold")
        t.append(f"{self._mcts_args.numMCTSSims}\n\n")
        t.append("Status: ", style="bold")
        t.append(f"{self._status}\n\n")

        is_human_turn = (board.turn == chess.WHITE) == self._human_is_white
        if is_human_turn and not self._game_over:
            legal = list(board.legal_moves)
            ucis = [m.uci() for m in legal[:8]]
            suffix = " …" if len(legal) > 8 else ""
            t.append("Legal: ", style="dim")
            t.append(f"{'  '.join(ucis)}{suffix}\n\n", style="dim")

        t.append("Moves:\n", style="bold")
        tmp = chess.Board()
        pairs: list[str] = []
        for i, m in enumerate(self._moves_played):
            san = tmp.san(m)
            tmp.push(m)
            if i % 2 == 0:
                pairs.append(f"{i // 2 + 1}. {san}")
            else:
                pairs[-1] += f" {san}"
        if pairs:
            for pair in pairs[-8:]:
                t.append(f"  {pair}\n")
        else:
            t.append("  (empty)\n", style="dim")
        return t


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="ChessBot Textual TUI")
    p.add_argument("--mode",             required=True, choices=["replay", "play"])
    p.add_argument("--pgn",              help="PGN file path (replay mode)")
    p.add_argument("--side",             default="white", choices=["white", "black"])
    p.add_argument("--checkpoint-dir",   default="./checkpoints")
    p.add_argument("--checkpoint-file",  default="best.pth.tar")
    p.add_argument("--mcts-sims",        type=int, default=100)
    p.add_argument("--num-channels",     type=int, default=256)
    p.add_argument("--num-res-blocks",   type=int, default=20)
    p.add_argument("--pgn-dir",          default="./games")
    args = p.parse_args()

    if args.mode == "replay":
        if not args.pgn:
            p.error("--pgn is required for replay mode")
        ReplayApp(pgn_path=args.pgn).run()

    elif args.mode == "play":
        nnet_args = dotdict({
            "num_channels":   args.num_channels,
            "num_res_blocks": args.num_res_blocks,
        })
        mcts_args = dotdict({
            "numMCTSSims":     args.mcts_sims,
            "cpuct":           1.0,
            "dirichlet_alpha": 0.0,
        })
        game = ChessGame()
        nnet = ChessNNet(game, nnet_args)
        ckpt = os.path.join(args.checkpoint_dir, args.checkpoint_file)
        if os.path.isfile(ckpt):
            nnet.load_checkpoint(args.checkpoint_dir, args.checkpoint_file)
            print(f"Loaded checkpoint: {ckpt}")
        else:
            print(f"Warning: no checkpoint at {ckpt}, using random weights")
        PlayApp(
            game=game,
            nnet=nnet,
            mcts_args=mcts_args,
            human_is_white=(args.side == "white"),
            pgn_dir=args.pgn_dir,
        ).run()


if __name__ == "__main__":
    main()
