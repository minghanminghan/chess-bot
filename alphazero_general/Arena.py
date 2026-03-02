"""
Arena — pits two players against each other to evaluate model improvement.

Players are callables: player(canonicalBoard) -> action (int).
Games alternate colours to remove first-mover bias.
Returns (player1_wins, player2_wins, draws).
"""

from tqdm import tqdm


class Arena:

    def __init__(self, player1, player2, game, display=None):
        """
        player1, player2: callables (canonicalBoard) -> action
        game:             Game instance
        display:          optional callable(board) for printing
        """
        self.player1 = player1
        self.player2 = player2
        self.game    = game
        self.display = display

    def playGame(self, verbose: bool = False) -> float:
        """
        Play a single game. player1 moves first.
        Returns 1 if player1 wins, -1 if player2 wins, 0 for draw.
        """
        players = [self.player2, None, self.player1]  # indexed by player (−1, 0, 1)
        cur_player = 1
        board = self.game.getInitBoard()
        step = 0

        while True:
            step += 1
            if self.display:
                self.display(board)

            canonical = self.game.getCanonicalForm(board, cur_player)
            action = players[cur_player + 1](canonical)

            valids = self.game.getValidMoves(canonical, 1)
            if valids[action] == 0:
                # Illegal move — forfeit (shouldn't happen with a correct player)
                return -cur_player

            board, cur_player = self.game.getNextState(board, cur_player, action)
            result = self.game.getGameEnded(board, cur_player)

            if result != 0:
                if self.display:
                    self.display(board)
                # result is from cur_player's perspective; cur_player just received it
                # A result of -1 means cur_player lost, so the previous player won
                return -cur_player * result  # positive = player1 won

    def playGames(self, num: int, verbose: bool = False):
        """
        Play num games, alternating who goes first each game.
        Returns (wins, losses, draws) from player1's perspective.
        """
        num = int(num)
        half = num // 2
        wins = losses = draws = 0

        # player1 goes first for the first half
        for _ in tqdm(range(half), desc="Arena (p1 first)", leave=False):
            result = self.playGame(verbose=verbose)
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1

        # Swap: player2 goes first for the second half
        self.player1, self.player2 = self.player2, self.player1
        for _ in tqdm(range(num - half), desc="Arena (p2 first)", leave=False):
            result = self.playGame(verbose=verbose)
            if result == -1:
                wins += 1   # player1 (now going second) won
            elif result == 1:
                losses += 1
            else:
                draws += 1
        # Restore original order
        self.player1, self.player2 = self.player2, self.player1

        return wins, losses, draws
