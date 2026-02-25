class NeuralNet:
    """
    Abstract neural network interface.
    Implementations wrap a specific deep learning framework (e.g. PyTorch).
    """

    def __init__(self, game):
        pass

    def train(self, examples):
        """
        Train on list of (board, pi, v) tuples.
        board: numpy array (8,8,119)
        pi:    numpy array (action_size,)
        v:     float in [-1, 1]
        """
        raise NotImplementedError

    def predict(self, board):
        """
        Return (pi, v) for a single board state.
        pi: numpy array (action_size,) — move probabilities
        v:  float — value estimate in [-1, 1]
        """
        raise NotImplementedError

    def save_checkpoint(self, folder, filename):
        raise NotImplementedError

    def load_checkpoint(self, folder, filename):
        raise NotImplementedError
