class Game:
    """
    Abstract base class for two-player zero-sum board games.
    Mirrors the interface from alpha-zero-general.
    """

    def getInitBoard(self):
        """Return initial board state."""
        raise NotImplementedError

    def getBoardSize(self):
        """Return (rows, cols) of the board."""
        raise NotImplementedError

    def getActionSize(self):
        """Return total number of possible actions."""
        raise NotImplementedError

    def getNextState(self, board, player, action):
        """
        Apply action to board. Return (next_board, next_player).
        player ∈ {1, -1}; next_player is the opponent.
        """
        raise NotImplementedError

    def getValidMoves(self, board, player):
        """Return binary vector of length getActionSize(); 1 = valid."""
        raise NotImplementedError

    def getGameEnded(self, board, player):
        """
        Return 0 if game not ended.
        Return 1 if player won, -1 if player lost, 1e-4 for draw.
        """
        raise NotImplementedError

    def getCanonicalForm(self, board, player):
        """Return board from the perspective of player (always 'white')."""
        raise NotImplementedError

    def getSymmetries(self, board, pi):
        """
        Return list of (board, pi) pairs after applying symmetry transforms.
        Used for data augmentation during training.
        """
        raise NotImplementedError

    def stringRepresentation(self, board):
        """Return hashable string for board (used as MCTS dict key)."""
        raise NotImplementedError
