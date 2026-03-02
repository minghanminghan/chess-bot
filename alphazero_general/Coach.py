"""
Coach — orchestrates AlphaZero self-play training.

Each iteration:
  1. Self-play: generate numEps games using MCTS → collect (s, π, z) examples
  2. Train: fit the network on the collected examples
  3. Evaluate: pit new network vs previous in the Arena
  4. Accept or reject the new network

Training examples are (board_tensor, pi, v):
  board_tensor : (8, 8, 119) float32
  pi           : (action_size,) float32  — MCTS visit-count policy
  v            : float ∈ [-1, 1]          — game outcome from that player's perspective
"""

import os
import pickle
import numpy as np
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from alphazero_general.Arena import Arena
from alphazero_general.MCTS import MCTS


# ── Module-level worker (must be top-level for multiprocessing pickling) ──────

def _run_episode_worker(checkpoint_dir: str, checkpoint_file: str, args_dict: dict) -> list:
    """Spawned by ProcessPoolExecutor. Loads its own model copy, runs one episode."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chessbot.ChessGame import ChessGame
    from chessbot.ChessNNet import ChessNNet
    from alphazero_general.MCTS import MCTS
    from utils import dotdict

    args  = dotdict(args_dict)
    game  = ChessGame()
    nnet  = ChessNNet(game, args)
    nnet.load_checkpoint(checkpoint_dir, checkpoint_file)

    mcts       = MCTS(game, nnet, args)
    board      = game.getInitBoard()
    cur_player = 1
    step       = 0
    examples   = []

    while True:
        step += 1
        # getCanonicalForm / getSymmetries omitted: both are chess no-ops.
        # to_tensor(canonical=True) handles the player-perspective flip internally.
        temp = 1 if step <= args.tempThreshold else 0
        pi   = mcts.getActionProb(board, temp=temp)
        examples.append([board.to_tensor(canonical=True), pi, cur_player])
        action = np.random.choice(len(pi), p=pi)
        board, cur_player = game.getNextState(board, cur_player, action)
        r = game.getGameEnded(board, cur_player)
        if r != 0:
            return [(t, p, r * (1 if pl == cur_player else -1)) for t, p, pl in examples]


class Coach:

    def __init__(self, game, nnet, args):
        self.game  = game
        self.nnet  = nnet
        self.pnet  = nnet.__class__(game, args)   # previous network (for comparison)
        self.args  = args
        self.trainExamplesHistory = []             # list of episode-example lists
        self.skipFirstSelfPlay = False             # set True to resume from checkpoint

    # ── Self-play ────────────────────────────────────────────────────────────

    def executeEpisode(self):
        """
        Play one full game of self-play using MCTS.
        Returns list of (board_tensor, pi, v) for each step.
        v is filled in retroactively once the game result is known.
        """
        mcts = MCTS(self.game, self.nnet, self.args)
        examples = []   # (board_tensor, pi, player_who_moved)
        board = self.game.getInitBoard()
        cur_player = 1
        step = 0

        while True:
            step += 1
            # getCanonicalForm / getSymmetries omitted: both are chess no-ops.
            # to_tensor(canonical=True) handles the player-perspective flip internally.
            temp = 1 if step <= self.args.tempThreshold else 0

            pi = mcts.getActionProb(board, temp=temp)
            examples.append([board.to_tensor(canonical=True), pi, cur_player])

            action = np.random.choice(len(pi), p=pi)
            board, cur_player = self.game.getNextState(board, cur_player, action)

            r = self.game.getGameEnded(board, cur_player)
            if r != 0:
                # Assign outcome to each example: v = r if player == cur_player else -r
                # cur_player is the player who is now to move *after* the game ended,
                # meaning r is from their perspective.
                return [
                    (tensor, pi_ex, r * (1 if player == cur_player else -1))
                    for tensor, pi_ex, player in examples
                ]

    # ── LR schedule ──────────────────────────────────────────────────────────

    def _get_lr(self, iteration: int) -> float:
        """
        Return the learning rate for the given iteration.
        args.lr_schedule is a dict {start_iter: lr}; the entry with the
        highest key that is <= iteration wins.  Falls back to args.lr.
        """
        schedule = self.args.get('lr_schedule', {})
        if not schedule:
            return self.args.lr
        lr = self.args.lr
        for milestone, rate in sorted(schedule.items()):
            if iteration >= milestone:
                lr = rate
        return lr

    # ── Training loop ────────────────────────────────────────────────────────

    def learn(self):
        for i in range(1, self.args.numIters + 1):
            print(f"\n{'='*50}")
            print(f"Iteration {i}/{self.args.numIters}")
            print('='*50)

            # ── Self-play ────────────────────────────────────────────────────
            if not self.skipFirstSelfPlay or i > 1:
                os.makedirs(self.args.checkpoint, exist_ok=True)
                WORKER_CKPT = 'worker.pth.tar'
                self.nnet.save_checkpoint(self.args.checkpoint, WORKER_CKPT)

                num_workers = self.args.get('num_workers', 1)
                iteration_examples = []

                if num_workers > 1:
                    with ProcessPoolExecutor(max_workers=num_workers) as pool:
                        futures = [
                            pool.submit(_run_episode_worker,
                                        self.args.checkpoint, WORKER_CKPT, dict(self.args))
                            for _ in range(self.args.numEps)
                        ]
                        for f in tqdm(as_completed(futures), total=self.args.numEps, desc="Self-play"):
                            iteration_examples += f.result()
                else:
                    for _ in tqdm(range(self.args.numEps), desc="Self-play"):
                        iteration_examples += self.executeEpisode()

                self.trainExamplesHistory.append(iteration_examples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples(i - 1)

            # Flatten and shuffle all historical examples
            train_examples = []
            for eps in self.trainExamplesHistory:
                train_examples.extend(eps)
            np.random.shuffle(train_examples)

            # ── Train ────────────────────────────────────────────────────────
            self.nnet.save_checkpoint(self.args.checkpoint, 'temp.pth.tar')
            self.pnet.load_checkpoint(self.args.checkpoint, 'temp.pth.tar')

            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(train_examples, lr=self._get_lr(i))
            nmcts = MCTS(self.game, self.nnet, self.args)

            # ── Arena ────────────────────────────────────────────────────────
            print(f"\nArena: {self.args.arenaCompare} games  "
                  f"(accept threshold: {self.args.updateThreshold:.0%})")
            arena = Arena(
                lambda b, m=nmcts: np.argmax(m.getActionProb(b, temp=0)),
                lambda b, m=pmcts: np.argmax(m.getActionProb(b, temp=0)),
                self.game,
            )
            nwins, pwins, draws = arena.playGames(self.args.arenaCompare)
            print(f"New wins: {nwins}  Prev wins: {pwins}  Draws: {draws}")

            total = nwins + pwins
            if total == 0 or nwins / total < self.args.updateThreshold:
                print("Rejecting new model.")
                self.nnet.load_checkpoint(self.args.checkpoint, 'temp.pth.tar')
            else:
                print("Accepting new model.")
                self.nnet.save_checkpoint(self.args.checkpoint, 'best.pth.tar')

    # ── Persistence ──────────────────────────────────────────────────────────

    def saveTrainExamples(self, iteration: int):
        os.makedirs(self.args.checkpoint, exist_ok=True)
        path = os.path.join(
            self.args.checkpoint, f"checkpoint_{iteration}.examples"
        )
        with open(path, "wb") as f:
            pickle.dump(self.trainExamplesHistory, f)

    def loadTrainExamples(self):
        model_file = os.path.join(self.args.checkpoint, 'best.pth.tar')
        examples_file = model_file + ".examples"
        if not os.path.isfile(examples_file):
            return
        with open(examples_file, "rb") as f:
            self.trainExamplesHistory = pickle.load(f)
        self.skipFirstSelfPlay = True
