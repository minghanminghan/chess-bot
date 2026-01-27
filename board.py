import numpy as np
import chess
from collections import deque

'''
overview:
input to the NN is an N x N x (MT + L) image stack that represents state
- image stack: 8 x 8 x (12 x 8 + 8) = 8 x 8 x 104
- each set of planes represents the board position at time t-T+1, ..., t
- plane is set to 0 for time-steps less than 1
- board is oriented to the perspective of the current player

feature planes (includes 8-step history)
- p1 pieces (p1 is current player):
    - pawn
    - rook
    - knight
    - bishop
    - queen
    - king
- p2 pieces:
    - analogous

feature values (one-hot encoded):
- color (1: 1 for white, 0 for black)
- total move count (1)
- p1 castling (2: kingside, queenside)
- p2 castling (2: kingside, queenside)
- no-progress count (1: 50-move rule)
- position repetition count (1)
'''

class Board:
    def __init__(self, board: chess.Board | None = None):
        self.board = board if board is not None else chess.Board()
        self.history = deque(maxlen=8) # 8 moves stored as (fen, repetition_count)
        self.fullmove_count = 1
        self.history.append(encode_board(self))

def make_move(b: Board, move: chess.Move):
    '''
    make a move on the board and update history + board internals
    try to use chess library as much as possible
    '''
    b.board.push(move)
    b.history.append(encode_board(b))


def encode_board(b: Board) -> np.ndarray:
    '''
    return encoded position of current board: 8 x 8 x (12)
    uint64 = 8 bytes

    figure out efficient p1, p2 ordering (pieces, repetition, castling)
    '''
    encoded = np.zeros((12, 8, 8), dtype=np.uint64)

    # piece planes
    w_pawn = np.uint64(b.board.pieces(chess.PAWN, chess.WHITE))
    encoded[0, :, :] = ((w_pawn >> np.arange(64, dtype=np.uint64)) & 1).reshape(8, 8)

    w_rook = np.uint64(b.board.pieces(chess.ROOK, chess.WHITE))
    encoded[1, :, :] = ((w_rook >> np.arange(64, dtype=np.uint64)) & 1).reshape(8, 8)

    w_knight = np.uint64(b.board.pieces(chess.KNIGHT, chess.WHITE))
    encoded[2, :, :] = ((w_knight >> np.arange(64, dtype=np.uint64)) & 1).reshape(8, 8)

    w_bishop = np.uint64(b.board.pieces(chess.BISHOP, chess.WHITE))
    encoded[3, :, :] = ((w_bishop >> np.arange(64, dtype=np.uint64)) & 1).reshape(8, 8)

    w_queen = np.uint64(b.board.pieces(chess.QUEEN, chess.WHITE))
    encoded[4, :, :] = ((w_queen >> np.arange(64, dtype=np.uint64)) & 1).reshape(8, 8)

    w_king = np.uint64(b.board.pieces(chess.KING, chess.WHITE))
    encoded[5, :, :] = ((w_king >> np.arange(64, dtype=np.uint64)) & 1).reshape(8, 8)
    
    b_pawn = np.uint64(b.board.pieces(chess.PAWN, chess.BLACK))
    encoded[6, :, :] = ((b_pawn >> np.arange(64, dtype=np.uint64)) & 1).reshape(8, 8)

    b_rook = np.uint64(b.board.pieces(chess.ROOK, chess.BLACK))
    encoded[7, :, :] = ((b_rook >> np.arange(64, dtype=np.uint64)) & 1).reshape(8, 8)

    b_knight = np.uint64(b.board.pieces(chess.KNIGHT, chess.BLACK))
    encoded[8, :, :] = ((b_knight >> np.arange(64, dtype=np.uint64)) & 1).reshape(8, 8)

    b_bishop = np.uint64(b.board.pieces(chess.BISHOP, chess.BLACK))
    encoded[9, :, :] = ((b_bishop >> np.arange(64, dtype=np.uint64)) & 1).reshape(8, 8)

    b_queen = np.uint64(b.board.pieces(chess.QUEEN, chess.BLACK))
    encoded[10, :, :] = ((b_queen >> np.arange(64, dtype=np.uint64)) & 1).reshape(8, 8)

    b_king = np.uint64(b.board.pieces(chess.KING, chess.BLACK))
    encoded[11, :, :] = ((b_king >> np.arange(64, dtype=np.uint64)) & 1).reshape(8, 8)

    return encoded


def get_encoded_state(b: Board) -> np.ndarray:
    '''
    return 8 x 8 x (12 * 8 + 8) board state representation
    '''
    encoded = np.zeros((12 * 8 + 8, 8, 8), dtype=np.uint64)
    
    # piece planes (8 time-steps)
    history = np.array(b.history).reshape(-1, 8, 8)
    pad_amount = 12 * 8 - history.shape[0]
    history = np.pad(history, ((0, max(0, pad_amount)), (0, 0), (0, 0)))[:12 * 8]
    encoded[0:12 * 8, :, :] = history

    # constant feature values (one-hot encoded)
    color_plane = np.uint64(b.board.turn == chess.WHITE) # 1 for white, 0 for black
    encoded[-8, :, :] = color_plane

    move_count_plane = np.uint64(b.board.fullmove_number)
    encoded[-7, :, :] = move_count_plane

    white_kingside_castle = np.uint64(int(b.board.has_kingside_castling_rights(chess.WHITE)))
    white_queenside_castle = np.uint64(int(b.board.has_queenside_castling_rights(chess.WHITE)))
    black_kingside_castle = np.uint64(int(b.board.has_kingside_castling_rights(chess.BLACK)))
    black_queenside_castle = np.uint64(int(b.board.has_queenside_castling_rights(chess.BLACK)))

    encoded[-6, :, :] = white_kingside_castle
    encoded[-5, :, :] = white_queenside_castle
    encoded[-4, :, :] = black_kingside_castle
    encoded[-3, :, :] = black_queenside_castle

    # TODO: turn this into a number instead of a boolean (Zobrist Hashing?)
    repetition = np.uint64(b.board.is_repetition())
    encoded[-2, :, :] = repetition

    no_progress = np.uint64(b.board.halfmove_clock)
    encoded[-1, :, :] = no_progress

    return encoded


def main():
    np.set_printoptions(threshold=np.inf)

    # b = chess.Board()
    # w_pawn = int(b.pieces(chess.PAWN, chess.WHITE))
    # b_rook = int(b.pieces(chess.ROOK, chess.BLACK))
    # print(bin(b_rook))

    b = Board()
    # encoded = encode_board(b)
    # print(encoded)
    # print(encoded.shape)

    # history = np.array(b.history)
    # print(history)
    # print(history.shape)
    encoded = get_encoded_state(b)
    print(encoded)
    print(encoded.shape)

if __name__ == "__main__":
    main()