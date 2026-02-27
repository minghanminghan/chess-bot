"""
ChessGame — concrete implementation of the Game interface for chess.

Board representation: ChessBoardState (wraps python-chess + history).
Player convention: 1 = white, -1 = black (matches alpha-zero-general).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import chess

from Game import Game
from chessbot.ChessBoard import (
    ChessBoardState, ACTION_SIZE, move_to_action, action_to_move
)


class ChessGame(Game):

    def getInitBoard(self) -> ChessBoardState:
        return ChessBoardState()

    def getBoardSize(self):
        return (8, 8)

    def getActionSize(self) -> int:
        return ACTION_SIZE  # 4672

    def getNextState(
        self, board: ChessBoardState, player: int, action: int
    ) -> tuple[ChessBoardState, int]:
        """
        Apply action to board. Returns (next_board, next_player).
        action is an index into the canonical (current-player) action space.
        For black we must un-flip the move before applying it.
        """
        move = action_to_move(action)
        if move is None:
            raise ValueError(f"Action {action} has no corresponding move.")

        # If it's black's turn, the board was presented flipped → un-flip the move
        if board.board.turn == chess.BLACK:
            move = _flip_move(move)

        next_board = board.push(move)
        next_player = -player
        return next_board, next_player

    def getValidMoves(self, board: ChessBoardState, player: int) -> np.ndarray:
        """
        Returns binary mask of valid actions in canonical (current-player) space.
        ChessBoardState.valid_moves_mask() already respects the current board turn;
        we need to flip move indices for black.
        """
        if board.board.turn == chess.WHITE:
            return board.valid_moves_mask()
        else:
            # Generate mask in white-perspective space, then re-encode
            mask = np.zeros(ACTION_SIZE, dtype=np.float32)
            for move in board.board.legal_moves:
                flipped = _flip_move(move)
                idx = move_to_action(flipped)
                if idx is not None:
                    mask[idx] = 1.0
            return mask

    def getGameEnded(self, board: ChessBoardState, player: int) -> float:
        """
        0 = game ongoing.
        1 = player won, -1 = player lost, 1e-4 = draw.
        Result is from the perspective of `player` (1=white, -1=black).
        """
        if not board.is_game_over():
            return 0
        res = board.board.result()
        if res == "1-0":
            return 1.0 if player == 1 else -1.0
        elif res == "0-1":
            return 1.0 if player == -1 else -1.0
        else:
            return 1e-4

    def getCanonicalForm(
        self, board: ChessBoardState, player: int
    ) -> ChessBoardState:
        """
        Return board from current player's perspective.
        The canonical form is simply the board as-is; to_tensor(canonical=True)
        handles the spatial flip internally by checking board.board.turn.

        NOTE: not called from the self-play loop (Coach.executeEpisode /
        _run_episode_worker) — those pass board directly. Called by
        Arena.playGame() which follows the canonical AlphaZero interface.
        """
        return board

    def getSymmetries(
        self, board: ChessBoardState, pi: np.ndarray
    ) -> list[tuple[ChessBoardState, np.ndarray]]:
        """
        Chess has no useful symmetries (unlike Othello or Go).
        Returns the identity only — no augmentation is applied.

        NOTE: not called from the self-play loop (Coach.executeEpisode /
        _run_episode_worker) — the call was removed since it is a no-op.
        Kept for compliance with the Game abstract interface.
        """
        return [(board, pi)]

    def stringRepresentation(self, board: ChessBoardState) -> tuple:
        return board.string_representation()

    def actionToMove(self, board: ChessBoardState, action: int) -> "chess.Move":
        """Convert a canonical action index to the chess.Move for the current board turn.
        Used by MCTS make-unmake: avoids getNextState's board copy."""
        move = action_to_move(action)
        if move is None:
            raise ValueError(f"Action {action} has no corresponding move.")
        if board.board.turn == chess.BLACK:
            move = _flip_move(move)
        return move


# ── Move flipping helpers ────────────────────────────────────────────────────

def _flip_sq(sq: int) -> int:
    """Mirror square vertically (rank flip: rank 0 ↔ rank 7)."""
    rank, file = divmod(sq, 8)
    return (7 - rank) * 8 + file


def _flip_move(move: chess.Move) -> chess.Move:
    """Flip a move for the canonical black-to-move representation."""
    return chess.Move(
        _flip_sq(move.from_square),
        _flip_sq(move.to_square),
        promotion=move.promotion,
    )
