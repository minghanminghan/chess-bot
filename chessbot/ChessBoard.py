# chessbot/ChessBoard.py — thin shim over the C++ cboard extension.
#
# ChessBoardState is now the C++ BoardWrapper class (exposed as Position).
# ACTION_SIZE is imported from the extension.
# python-chess is no longer required for training; it remains in [ui] extras.

from chessbot.cboard import Position as ChessBoardState, ACTION_SIZE

__all__ = ["ChessBoardState", "ACTION_SIZE"]
