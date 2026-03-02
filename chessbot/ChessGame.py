"""
ChessGame — concrete implementation of the Game interface for chess.

Board representation: C++ cboard.Position (via ChessBoardState shim).
Player convention: 1 = white, -1 = black (matches alpha-zero-general).

python-chess is no longer imported here; colour dispatch and move-flipping
are handled inside the C++ extension.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from alphazero_general.Game import Game
from chessbot.ChessBoard import ChessBoardState, ACTION_SIZE


class ChessGame(Game):

    def getInitBoard(self) -> ChessBoardState:
        return ChessBoardState()

    def getBoardSize(self):
        return (8, 8)

    def getActionSize(self) -> int:
        return ACTION_SIZE  # 4672

    def getNextState(
        self, board: ChessBoardState, player: int, action: int
    ) -> tuple:
        """Apply action to board. Returns (next_board, next_player).
        The C++ apply() handles canonical flip for black internally."""
        next_board = board.copy()
        next_board.apply(action)
        return next_board, -player

    def getValidMoves(self, board: ChessBoardState, player: int) -> np.ndarray:
        """Binary mask of valid actions in canonical (current-player) space.
        The C++ valid_moves_mask() handles colour dispatch internally."""
        return board.valid_moves_mask()

    def getGameEnded(self, board: ChessBoardState, player: int) -> float:
        """
        0 = game ongoing.
        1e-4 = draw. ±1 from the perspective of `player`.

        board.result() is from the perspective of the player TO MOVE.
        getGameEnded is always called with player == side_to_move in practice,
        so the guard below is defensive.
        """
        r = board.result()
        if r == 0.0:
            return 0
        return r if board.side_to_move() == player else -r

    def getCanonicalForm(
        self, board: ChessBoardState, player: int
    ) -> ChessBoardState:
        """Return board as-is; to_tensor(canonical=True) handles orientation."""
        return board

    def getSymmetries(
        self, board: ChessBoardState, pi: np.ndarray
    ) -> list:
        """Chess has no useful symmetries — return identity only."""
        return [(board, pi)]

    def stringRepresentation(self, board: ChessBoardState) -> tuple:
        return board.string_representation()
