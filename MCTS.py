"""
Monte Carlo Tree Search — AlphaZero style.

Each node is identified by a string board representation.
The tree stores:
  Qsa[(s, a)]  Q-value of (state, action)
  Nsa[(s, a)]  visit count of (state, action)
  Ns[s]        visit count of state s
  Ps[s]        prior policy vector (from neural network)
  Es[s]        game-ended result (cached)
  Vs[s]        valid moves mask (cached)

UCB formula:
  U(s, a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

Dirichlet noise is added to the root prior on the first visit:
  P(s, ·) = 0.75 * P(s, ·) + 0.25 * Dir(alpha)
"""

import math
import numpy as np


EPS = 1e-8


class MCTS:

    def __init__(self, game, nnet, args):
        self.game  = game
        self.nnet  = nnet
        self.args  = args

        self.Qsa = {}   # (s, a) -> float
        self.Nsa = {}   # (s, a) -> int
        self.Ns  = {}   # s -> int
        self.Ps  = {}   # s -> np.ndarray(action_size)
        self.Es  = {}   # s -> float (game-ended result, 0 = ongoing)
        self.Vs  = {}   # s -> np.ndarray(action_size)  valid-move mask

    def getActionProb(self, canonicalBoard, temp: float = 1.0) -> np.ndarray:
        """
        Run numMCTSSims simulations from canonicalBoard and return a policy
        vector proportional to N(s,a)^(1/temp).

        temp=1  → proportional to visit count  (exploration)
        temp=0  → one-hot on most-visited move  (exploitation)
        """
        for _ in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = np.array([
            self.Nsa.get((s, a), 0)
            for a in range(self.game.getActionSize())
        ], dtype=np.float32)

        if temp == 0:
            best = np.argmax(counts)
            probs = np.zeros_like(counts)
            probs[best] = 1.0
            return probs

        counts = counts ** (1.0 / temp)
        total = counts.sum()
        if total == 0:
            # Fallback: uniform over valid moves
            valids = self.Vs.get(s)
            if valids is None:
                valids = self.game.getValidMoves(canonicalBoard, 1)
            counts = valids.copy()
            total = counts.sum()
        return counts / total

    def search(self, canonicalBoard) -> float:
        """
        Perform one MCTS simulation starting from canonicalBoard.
        Returns the negative value (for the parent to back-propagate).
        """
        s = self.game.stringRepresentation(canonicalBoard)

        # ── Terminal node ───────────────────────────────────────────────────
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        # ── Leaf expansion ──────────────────────────────────────────────────
        if s not in self.Ps:
            tensor = canonicalBoard.to_tensor(canonical=True)
            self.Ps[s], v = self.nnet.predict(tensor)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Vs[s] = valids

            # Mask invalid moves
            self.Ps[s] *= valids
            sum_ps = self.Ps[s].sum()
            if sum_ps > EPS:
                self.Ps[s] /= sum_ps
            else:
                # All moves got masked — fallback to uniform over valid moves
                self.Ps[s] = valids / (valids.sum() + EPS)

            # Dirichlet noise at root (first visit counts as root)
            if self.args.get('dirichlet_alpha', 0) > 0:
                noise = np.random.dirichlet(
                    [self.args.dirichlet_alpha] * self.game.getActionSize()
                )
                eps = self.args.get('dirichlet_eps', 0.25)
                self.Ps[s] = (1 - eps) * self.Ps[s] + eps * noise * valids
                sum_ps = self.Ps[s].sum()
                if sum_ps > EPS:
                    self.Ps[s] /= sum_ps

            self.Ns[s] = 0
            return -v

        # ── Select action via UCB ───────────────────────────────────────────
        valids = self.Vs[s]
        best_u, best_a = -float('inf'), -1
        sqrt_ns = math.sqrt(self.Ns[s] + EPS)

        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            q = self.Qsa.get((s, a), 0.0)
            n = self.Nsa.get((s, a), 0)
            u = q + self.args.cpuct * self.Ps[s][a] * sqrt_ns / (1 + n)
            if u > best_u:
                best_u = u
                best_a = a

        a = best_a
        next_board, _ = self.game.getNextState(canonicalBoard, 1, a)
        next_canonical = self.game.getCanonicalForm(next_board, -1)

        v = self.search(next_canonical)

        # ── Back-propagate ──────────────────────────────────────────────────
        if (s, a) in self.Qsa:
            n = self.Nsa[(s, a)]
            self.Qsa[(s, a)] = (self.Qsa[(s, a)] * n + v) / (n + 1)
            self.Nsa[(s, a)] = n + 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
