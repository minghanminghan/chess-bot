"""
AlphaZero board representation for chess.

Input tensor: 8 × 8 × 119
  Planes 0..111  — 8 history steps × 14 planes each:
      [0..5]   white pieces: P, N, B, R, Q, K (binary occupancy)
      [6..11]  black pieces: p, n, b, r, q, k (binary occupancy)
      [12]     repetition count == 1
      [13]     repetition count >= 2
  Planes 112..118 — scalar feature planes (broadcast to 8×8):
      [112]    colour: 1 = white to move, 0 = black to move
      [113]    total move count / 500
      [114]    white kingside castling rights
      [115]    white queenside castling rights
      [116]    black kingside castling rights
      [117]    black queenside castling rights
      [118]    no-progress count / 100

Action encoding: 64 × 73 = 4672 actions
  For each source square (python-chess: sq = file + rank*8, rank 0=rank1):
    - 56 queen-style moves: 8 directions × 7 distances (slots 0..55)
    - 8 knight moves (slots 56..63)
    - 9 under-promotions: 3 pieces × 3 directions (slots 64..72)
  Queen promotions share an index with the corresponding queen-direction move.
"""

import numpy as np
import chess
from collections import deque

# ── Coordinate convention ────────────────────────────────────────────────────
# python-chess: sq = file + rank*8  (file 0=a-file, rank 0=rank1, rank 7=rank8)
# We decompose as: rank, file = divmod(sq, 8)
# ─────────────────────────────────────────────────────────────────────────────

PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

# 8 queen/bishop/rook directions: (drank, dfile)
QUEEN_DIRS = [
    (1,  0),   # N
    (1,  1),   # NE
    (0,  1),   # E
    (-1, 1),   # SE
    (-1, 0),   # S
    (-1, -1),  # SW
    (0,  -1),  # W
    (1,  -1),  # NW
]  # 8 dirs × 7 distances = 56 slots (0..55)

KNIGHT_DELTAS = [
    (2,  1), (2, -1), (-2,  1), (-2, -1),
    (1,  2), (1, -2), (-1,  2), (-1, -2),
]  # slots 56..63

UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
UNDERPROMO_DFILES = [-1, 0, 1]  # left-capture, straight, right-capture
# slots 64..72: piece_i*3 + dir_i


ACTION_SIZE = 64 * 73  # 4672


def _sq(rank: int, file: int) -> int:
    return rank * 8 + file


def _build_action_maps():
    """
    Build bidirectional maps over all 4672 action slots.
    Off-board destinations are stored as None in idx_to_move.
    """
    move_to_idx: dict[chess.Move, int] = {}
    idx_to_move: dict[int, "chess.Move | None"] = {}

    for sq in range(64):
        rank, file = divmod(sq, 8)
        base = sq * 73

        # ── Queen-style moves (slots 0..55) ──────────────────────────────
        slot = 0
        for (dr, df) in QUEEN_DIRS:
            for dist in range(1, 8):
                idx = base + slot
                r2, f2 = rank + dr * dist, file + df * dist
                slot += 1
                if 0 <= r2 < 8 and 0 <= f2 < 8:
                    to_sq = _sq(r2, f2)
                    # Queen promotion: white pawn from rank6 → rank7
                    promo = chess.QUEEN if (rank == 6 and r2 == 7) else None
                    m = chess.Move(sq, to_sq, promotion=promo)
                    if m not in move_to_idx:
                        move_to_idx[m] = idx
                    idx_to_move[idx] = m
                    # Also map the bare move (no promotion) to the same slot
                    m_bare = chess.Move(sq, to_sq)
                    if m_bare not in move_to_idx:
                        move_to_idx[m_bare] = idx
                else:
                    idx_to_move[idx] = None

        # ── Knight moves (slots 56..63) ───────────────────────────────────
        for k, (dr, df) in enumerate(KNIGHT_DELTAS):
            idx = base + 56 + k
            r2, f2 = rank + dr, file + df
            if 0 <= r2 < 8 and 0 <= f2 < 8:
                to_sq = _sq(r2, f2)
                m = chess.Move(sq, to_sq)
                if m not in move_to_idx:
                    move_to_idx[m] = idx
                idx_to_move[idx] = m
            else:
                idx_to_move[idx] = None

        # ── Under-promotions (slots 64..72) ──────────────────────────────
        for p_i, piece in enumerate(UNDERPROMO_PIECES):
            for d_i, df in enumerate(UNDERPROMO_DFILES):
                idx = base + 64 + p_i * 3 + d_i
                r2, f2 = rank + 1, file + df
                if rank == 6 and 0 <= f2 < 8:
                    to_sq = _sq(r2, f2)
                    m = chess.Move(sq, to_sq, promotion=piece)
                    if m not in move_to_idx:
                        move_to_idx[m] = idx
                    idx_to_move[idx] = m
                else:
                    idx_to_move[idx] = None

    return move_to_idx, idx_to_move


MOVE_TO_IDX, IDX_TO_MOVE = _build_action_maps()


def move_to_action(move: chess.Move) -> "int | None":
    """Return action index for move, or None if not in table."""
    return MOVE_TO_IDX.get(move)


def action_to_move(idx: int) -> "chess.Move | None":
    return IDX_TO_MOVE.get(idx)


# ── Board state container ────────────────────────────────────────────────────

class ChessBoardState:
    """
    Wraps a python-chess Board with a rolling 8-position history for
    AlphaZero feature-plane construction.
    """

    HISTORY_LEN = 8

    def __init__(self, board: "chess.Board | None" = None):
        if board is None:
            board = chess.Board()
        self.board = board.copy()
        self._history: deque = deque(maxlen=self.HISTORY_LEN)
        self._push_history()

    def _push_history(self):
        b = self.board
        rep1 = int(b.is_repetition(2))
        rep2 = int(b.is_repetition(3))
        self._history.append((b.board_fen(), rep1, rep2))

    def copy(self) -> "ChessBoardState":
        new = ChessBoardState.__new__(ChessBoardState)
        new.board = self.board.copy()
        new._history = deque(list(self._history), maxlen=self.HISTORY_LEN)
        return new

    def push(self, move: chess.Move) -> "ChessBoardState":
        """Return new state after applying move (non-destructive)."""
        new = self.copy()
        new.board.push(move)
        new._push_history()
        return new

    # ── Feature tensor ───────────────────────────────────────────────────────

    def to_tensor(self, canonical: bool = True) -> np.ndarray:
        """
        Return (8, 8, 119) float32 array.
        canonical=True: board is oriented from current player's perspective
        (black's pieces appear at the bottom when it is black's turn).
        """
        planes = np.zeros((8, 8, 119), dtype=np.float32)
        history = list(self._history)   # oldest → newest

        # Pad to HISTORY_LEN with None entries at the front
        while len(history) < self.HISTORY_LEN:
            history.insert(0, None)

        flip = canonical and (self.board.turn == chess.BLACK)

        for t, entry in enumerate(history):
            offset = t * 14
            if entry is None:
                continue
            fen, rep1, rep2 = entry
            tmp = chess.Board(fen)

            for sq in chess.SQUARES:
                piece = tmp.piece_at(sq)
                if piece is None:
                    continue
                plane_idx = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
                rank, file = divmod(sq, 8)
                row = (7 - rank) if flip else rank
                planes[row, file, offset + plane_idx] = 1.0

            if rep1:
                planes[:, :, offset + 12] = 1.0
            if rep2:
                planes[:, :, offset + 13] = 1.0

        # Scalar planes 112–118
        planes[:, :, 112] = float(self.board.turn == chess.WHITE)
        planes[:, :, 113] = self.board.fullmove_number / 500.0
        planes[:, :, 114] = float(self.board.has_kingside_castling_rights(chess.WHITE))
        planes[:, :, 115] = float(self.board.has_queenside_castling_rights(chess.WHITE))
        planes[:, :, 116] = float(self.board.has_kingside_castling_rights(chess.BLACK))
        planes[:, :, 117] = float(self.board.has_queenside_castling_rights(chess.BLACK))
        planes[:, :, 118] = self.board.halfmove_clock / 100.0

        return planes

    def valid_moves_mask(self) -> np.ndarray:
        """Binary float32 vector of length ACTION_SIZE."""
        mask = np.zeros(ACTION_SIZE, dtype=np.float32)
        for move in self.board.legal_moves:
            idx = move_to_action(move)
            if idx is not None:
                mask[idx] = 1.0
        return mask

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def result(self) -> float:
        """
        Outcome from the perspective of the CURRENT player (the one to move).
        After a checkmate the side to move has been mated → returns -1.
        Draw → 1e-4.
        """
        res = self.board.result()
        if res == "1-0":
            return 1.0 if self.board.turn == chess.WHITE else -1.0
        elif res == "0-1":
            return 1.0 if self.board.turn == chess.BLACK else -1.0
        else:
            return 1e-4

    def string_representation(self) -> str:
        return self.board.fen()
