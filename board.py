import numpy as np
import chess

'''
overview:
input to the NN is an N x N x (MT + L) image stack that represents state
- image stack: 8 x 8 x (14 x 8 + 7) = 8 x 8 x 119
- each set of planes represents the board position at time t-T+1, ..., t
- plane is set to 0 for time-steps less than 1
- board is oriented to the perspective of the current player

feature planes (includes 8-step history)
- p1 pieces (p1 is current player):
    - pawn
    - knight
    - bishop
    - rook
    - queen
    - king
- p2 pieces:
    - analogous
- p1 repetition plane
- p2 repetition plane

feature values (one-hot encoded):
- color (1)
- total move count (1)
- p1 castling (2: kingside, queenside)
- p2 castling (2: kingside, queenside)
- no-progress count (50-move rule)
'''

class Board:
    def __init__(self):
        self.board = chess.Board()
        self.history = [] # 8 moves stored as (fen, repetition_count)
        self.move_count = 0


def encode_board(board: Board) -> np.ndarray:
    '''
    return current board encoding without history
    '''


def make_move(board: Board, move: chess.Move):
    '''
    make a move on the board and update history + board internals
    try to use chess library as much as possible
    '''


def get_encoded_state(board: Board) -> np.ndarray:
    '''
    return 8 x 8 x 119 board state representation
    '''