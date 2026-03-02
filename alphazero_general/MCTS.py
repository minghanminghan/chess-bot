"""
Monte Carlo Tree Search — AlphaZero style with batch inference.

Simulations are collected in batches of `mcts_batch_size`; all NEW leaves
in a batch are evaluated in a single GPU forward pass via nnet.predict_batch().
Duplicate leaves within the same batch (multiple sims hitting the same unvisited
node) are deduplicated before inference; their paths all receive the same v.

Each node is identified by board.string_representation() — a (zobrist, halfmove)
int tuple; ~2 µs vs ~50 µs for board.fen().

The tree stores:
  Qsa[s]   Q-values for all actions from s  (ndarray, shape action_size, f32)
  Nsa[s]   visit counts for all actions     (ndarray, shape action_size, i32)
  Ns[s]    visit count of state s
  Ps[s]    prior policy vector (from neural network)
  Es[s]    game-ended result (cached)
  Vs[s]    valid moves mask (cached)

UCB formula:
  U(s, a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
"""

import math
import numpy as np


EPS = 1e-8


class MCTS:

    def __init__(self, game, nnet, args):
        self.game  = game
        self.nnet  = nnet
        self.args  = args

        self.Qsa = {}   # s -> ndarray(action_size, f32)
        self.Nsa = {}   # s -> ndarray(action_size, i32)
        self.Ns  = {}   # s -> int
        self.Ps  = {}   # s -> np.ndarray(action_size)
        self.Es  = {}   # s -> float (game-ended result, 0 = ongoing)
        self.Vs  = {}   # s -> np.ndarray(action_size)  valid-move mask

    # ── Public API ───────────────────────────────────────────────────────────

    def getActionProb(self, canonicalBoard, temp: float = 1.0) -> np.ndarray:
        """
        Run numMCTSSims simulations from canonicalBoard and return a policy
        vector proportional to N(s,a)^(1/temp).

        Simulations are grouped into batches of mcts_batch_size; each batch
        issues one GPU call via nnet.predict_batch() instead of one call per
        simulation.

        temp=1  → proportional to visit count  (exploration)
        temp=0  → one-hot on most-visited move  (exploitation)
        """
        batch_size = self.args.get('mcts_batch_size', 8)
        sims_left  = self.args.numMCTSSims
        root_s     = self.game.stringRepresentation(canonicalBoard)

        while sims_left > 0:
            n = min(batch_size, sims_left)
            self._run_batch(canonicalBoard, n, root_s)
            sims_left -= n

        s      = self.game.stringRepresentation(canonicalBoard)
        counts = self.Nsa.get(s, np.zeros(self.game.getActionSize(), dtype=np.int32)).astype(np.float32)

        if temp == 0:
            best  = np.argmax(counts)
            probs = np.zeros_like(counts)
            probs[best] = 1.0
            return probs

        counts = counts ** (1.0 / temp)
        total  = counts.sum()
        if total == 0:
            # Fallback: uniform over valid moves
            valids = self.Vs.get(s)
            if valids is None:
                valids = self.game.getValidMoves(canonicalBoard, 1)
            counts = valids.copy()
            total  = counts.sum()
        return counts / total

    # ── Batch machinery ──────────────────────────────────────────────────────

    def _run_batch(self, root_board, n: int, root_s=None) -> None:
        """
        Walk n simulations to leaves, expand all with one GPU call, backprop.

        Each walk applies moves (make-unmake) on root_board and restores it
        before the next walk starts — no board copies.
        """
        pending = []   # (path, s_leaf, tensor, valids)
        for _ in range(n):
            result = self._walk(root_board)
            if result is not None:
                pending.append(result)

        if not pending:
            return

        # Deduplicate: multiple sims may reach the same unvisited node.
        # All deduplicated leaves get one inference call; every path to that
        # node receives the same v in backprop.
        unique_s: list       = []
        unique_tensors: list = []
        s_to_valids: dict    = {}
        paths_by_s: dict     = {}

        for path, s, tensor, valids in pending:
            if s not in paths_by_s:
                unique_s.append(s)
                unique_tensors.append(tensor)
                s_to_valids[s] = valids
                paths_by_s[s]  = []
            paths_by_s[s].append(path)

        # Single batched GPU call for all unique new leaves.
        batch_pis, batch_vs = self.nnet.predict_batch(np.stack(unique_tensors))

        action_size = self.game.getActionSize()
        for s, pi, v in zip(unique_s, batch_pis, batch_vs):
            valids = s_to_valids[s]

            # Mask illegal moves, renormalize.
            pi    *= valids
            sum_ps = pi.sum()
            pi     = pi / sum_ps if sum_ps > EPS else valids / (valids.sum() + EPS)

            # Dirichlet noise at root only (AlphaZero paper §B).
            if s == root_s and self.args.get('dirichlet_alpha', 0) > 0:
                noise = np.random.dirichlet([self.args.dirichlet_alpha] * action_size)
                eps   = self.args.get('dirichlet_eps', 0.25)
                pi    = (1 - eps) * pi + eps * noise * valids
                pi    = pi / (pi.sum() + EPS)

            self.Ps[s]  = pi
            self.Vs[s]  = valids
            self.Ns[s]  = 0
            self.Qsa[s] = np.zeros(action_size, dtype=np.float32)
            self.Nsa[s] = np.zeros(action_size, dtype=np.int32)

            for path in paths_by_s[s]:
                self._backprop(path, v)

    def _walk(self, board) -> "tuple | None":
        """
        Walk tree from root via UCB until reaching an unvisited leaf or terminal.
        Uses apply/undo: board is always restored to its original (root) state
        before this method returns.

        Returns (path, s_leaf, tensor, valids) for a new leaf needing inference.
        Returns None for terminals (backprop is handled internally).

        path: list of (s, a) pairs from root toward the leaf, used for backprop.
        """
        path = []

        while True:
            s = self.game.stringRepresentation(board)

            # ── Terminal ─────────────────────────────────────────────────────
            if s not in self.Es:
                self.Es[s] = self.game.getGameEnded(board, 1)
            if self.Es[s] != 0:
                for _ in path:
                    board.undo()
                self._backprop(path, self.Es[s])
                return None

            # ── Unvisited leaf: needs neural net ─────────────────────────────
            if s not in self.Ps:
                tensor = board.to_tensor(canonical=True)
                valids = self.game.getValidMoves(board, 1)
                for _ in path:
                    board.undo()
                return path, s, tensor, valids

            # ── Interior node: UCB selection ──────────────────────────────────
            valids  = self.Vs[s]
            sqrt_ns = math.sqrt(self.Ns[s] + EPS)
            u       = self.Qsa[s] + self.args.cpuct * self.Ps[s] * sqrt_ns / (1 + self.Nsa[s])
            u[valids == 0] = -np.inf
            a = int(np.argmax(u))

            path.append((s, a))
            board.apply(a)

    def _backprop(self, path, leaf_v: float) -> None:
        """
        Backpropagate leaf value along a path.

        path:   list of (s, a) from root toward the leaf.
        leaf_v: value of the leaf from the LEAF player's perspective.
                The parent of the leaf sees -leaf_v (it is the opponent).

        Q and N are updated from the leaf's parent up to the root, flipping
        the sign at each level to account for alternating players.
        """
        v = -leaf_v   # parent's perspective is opposite the leaf's
        for s, a in reversed(path):
            n = self.Nsa[s][a]
            self.Qsa[s][a] = (self.Qsa[s][a] * n + v) / (n + 1)
            self.Nsa[s][a] = n + 1
            self.Ns[s]     += 1
            v = -v   # grandparent's perspective is opposite the parent's
