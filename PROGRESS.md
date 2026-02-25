# AlphaZero Chess Bot — Progress

Reference: [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) | Paper: https://arxiv.org/pdf/1712.01815

---

## File Layout

```
chess-bot/
├── chessbot/               # renamed from chess/ (avoids shadowing python-chess)
│   ├── __init__.py
│   ├── ChessGame.py        # Game interface: canonical flip, _flip_move, getNextState
│   ├── ChessBoard.py       # 8×8×119 tensor, ACTION_SIZE=4672, move_to_action/action_to_move
│   └── ChessNNet.py        # ResNet (20 res blocks, 256 ch), policy + value heads
├── Game.py                 # abstract base class
├── NeuralNet.py            # abstract NN interface
├── MCTS.py                 # UCB tree search + Dirichlet noise at root
├── Arena.py                # model evaluation: (player1, player2, game) → (w, l, d)
├── Coach.py                # self-play training loop
├── utils.py                # AverageMeter, dotdict
├── train.py                # 100k-step training entry point (do NOT auto-run)
├── tui.py                  # [planned] Textual TUI: replay PGN + play vs bot
├── uci_engine.py           # [planned] UCI protocol wrapper for cutechess-cli
├── elo.py                  # [planned] cutechess-cli ELO evaluation helper
└── pyproject.toml          # uv: torch, python-chess, numpy, tqdm
```

### Key Design Decisions
- **Input**: 8×8×119 — 8 history steps × 14 planes + 7 scalar planes
- **Network**: 20 residual blocks, 256 filters; policy → 4672 log-probs; value → tanh scalar
- **MCTS**: UCB with c_puct=1.0, Dirichlet α=0.3 noise at root, 800 sims/move
- **Loss**: `(z−v)² − π^T log(p) + c‖θ‖²`, c=1e-4
- **Canonical form**: spatial flip only (rows mirrored for black), piece colour planes unchanged
- **Action space**: 64×73=4672; always white-perspective; `_flip_move` converts black moves

---

## Phase 1 — AlphaZero Core (COMPLETE)

| # | File | Status | Verified by |
|---|------|--------|-------------|
| 1 | `pyproject.toml` (uv setup) | ✅ | `import chess, torch, numpy` → all OK |
| 2 | `Game.py`, `NeuralNet.py`, `utils.py` | ✅ | imports clean, AverageMeter avg=2.0 |
| 3 | `chessbot/ChessBoard.py` | ✅ | shape (8,8,119), ACTION_SIZE=4672, 20 legal moves at start, Fool's mate −1 |
| 4 | `chessbot/ChessGame.py` | ✅ | 10-move random rollout, checkmate detection, string repr |
| 5 | `chessbot/ChessNNet.py` | ✅ | forward shapes, train step, checkpoint round-trip |
| 6 | `MCTS.py` | ✅ | 50 sims, policy sums to 1, zero mass on illegal moves |
| 7 | `Arena.py` | ✅ | 2-game smoke test, w+l+d==2 |
| 8 | `Coach.py` | ✅ | 1-iter loop (self-play→train→arena→accept/reject) |
| 9 | `train.py` | ✅ | 1000×100=100k config verified; NOT run |

---

## Phase 2 — Strength Testing (PLANNED)

| # | File | Status | Description |
|---|------|--------|-------------|
| 10 | `tui.py` | ⬜ | Textual TUI: replay PGN + live play vs bot, auto-save PGN |
| 11 | `uci_engine.py` | ⬜ | UCI stdin/stdout wrapper so bot works with cutechess-cli |
| 12 | `elo.py` | ⬜ | Shell out to cutechess-cli, stream results, print ELO estimate |

---

## Step 10 — `tui.py` (Textual TUI)

### What it does
Two modes selected via `--mode replay|play`:
- **Replay**: load a PGN file, step through moves with keyboard shortcuts (←/→ arrows or buttons)
- **Play**: play live vs the trained bot by typing UCI moves; auto-saves game to `./games/game_<timestamp>.pgn`

### Framework
[**Textual**](https://github.com/Textualize/textual) — modern Python TUI framework with reactive widgets, CSS layout, async workers, and mouse support. Add via `uv add textual`.

---

### Layout — Replay mode

```
┌─ ChessBot Replay ───────────────────────────────────────────┐
│ File: game_20240101_120000.pgn                               │
├────────────────────────────┬────────────────────────────────┤
│  8  ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜      │  White: Engine_v2              │
│  7  ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟      │  Black: Engine_v1              │
│  6  · · · · · · · ·        │  Result: 1-0                   │
│  5  · · · · · · · ·        │                                │
│  4  · · · · ♙ · · ·        │  Move 5 / 42                  │
│  3  · · · · · · · ·        │  1. e4 e5                     │
│  2  ♙ ♙ ♙ ♙ · ♙ ♙ ♙      │  2. Nf3 Nc6                   │
│  1  ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖      │  3. Bb5 a6                    │
│     a  b  c  d  e  f  g  h │  4. Ba4 Nf6                   │
├────────────────────────────┤  5. O-O ▶                      │
│  [◀ Prev]       [Next ▶]  │                                │
│  ← / → arrow keys          │                                │
└────────────────────────────┴────────────────────────────────┘
```

### Layout — Play mode

```
┌─ ChessBot Play ─────────────────────────────────────────────┐
│ You: White   Bot: Black   Sims: 100                          │
├────────────────────────────┬────────────────────────────────┤
│  8  ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜      │  Status: Your turn             │
│  7  ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟      │                                │
│  6  · · · · · · · ·        │  Legal moves:                 │
│  5  · · · · · · · ·        │  a2a3 a2a4 b2b3 b2b4 …       │
│  4  · · · · · · · ·        │                                │
│  3  · · · · · · · ·        │  Move history:                │
│  2  ♙ ♙ ♙ ♙ ♙ ♙ ♙ ♙      │  (empty)                      │
│  1  ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖      │                                │
│     a  b  c  d  e  f  g  h │                                │
├────────────────────────────┴────────────────────────────────┤
│  Move (UCI): [e2e4              ]  [Submit]  [Resign]        │
└─────────────────────────────────────────────────────────────┘
```

---

### Implementation plan

**App structure (Textual):**
```python
class ChessTUI(App):
    CSS_PATH = None   # inline CSS via CSS string
    BINDINGS = [
        ("left",  "prev_move", "Previous"),
        ("right", "next_move", "Next"),
        ("q",     "quit",      "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Horizontal(
            BoardWidget(),      # custom Static subclass
            InfoPanel(),        # Static: player names, move list, status
        )
        yield MoveInput()       # Input + Button row (play mode only)
        yield Footer()
```

**`BoardWidget` (custom `Widget`):**
- Renders the 8×8 board as a `Rich` `Table` or formatted string using `Text`
- Light squares: `on grey82`, dark: `on dark_goldenrod` (Rich color names)
- Pieces: Unicode map by `(piece_type, color)`:
  ```python
  GLYPHS = {
      (chess.KING,   chess.WHITE): "♔", (chess.QUEEN,  chess.WHITE): "♕",
      (chess.ROOK,   chess.WHITE): "♖", (chess.BISHOP, chess.WHITE): "♗",
      (chess.KNIGHT, chess.WHITE): "♘", (chess.PAWN,   chess.WHITE): "♙",
      (chess.KING,   chess.BLACK): "♚", (chess.QUEEN,  chess.BLACK): "♛",
      (chess.ROOK,   chess.BLACK): "♜", (chess.BISHOP, chess.BLACK): "♝",
      (chess.KNIGHT, chess.BLACK): "♞", (chess.PAWN,   chess.BLACK): "♟",
  }
  ```
- Last-move squares highlighted with `bold` + `on dark_green`
- Rank/file labels on edges
- Orientation: white at bottom; flip rank order for `--side black`

**Replay mode state:**
```python
self.moves: list[chess.Move] = list(game.mainline_moves())
self.move_idx: int = 0   # 0 = starting position

def action_next_move(self):
    if self.move_idx < len(self.moves):
        self.board.push(self.moves[self.move_idx])
        self.move_idx += 1
        self.query_one(BoardWidget).refresh()

def action_prev_move(self):
    if self.move_idx > 0:
        self.board.pop()
        self.move_idx -= 1
        self.query_one(BoardWidget).refresh()
```

**Play mode — move submission:**
```python
def on_button_pressed(self, event):
    uci_str = self.query_one(Input).value.strip()
    self._submit_move(uci_str)

def _submit_move(self, uci_str: str):
    # Validate
    try:
        move = chess.Move.from_uci(uci_str)
    except ValueError:
        self.status = f"Invalid UCI: {uci_str}"
        return
    if move not in self.board_state.board.legal_moves:
        # Auto-promote to queen if bare pawn move to back rank
        promo = chess.Move.from_uci(uci_str + "q")
        if promo in self.board_state.board.legal_moves:
            move = promo
        else:
            self.status = f"Illegal move: {uci_str}"
            return
    # Encode → canonical action
    canonical_move = _flip_move(move) if self.board_state.board.turn == chess.BLACK else move
    action = move_to_action(canonical_move)
    self.board_state, self.cur_player = self.game.getNextState(
        self.board_state, self.cur_player, action
    )
    self.moves_played.append(move)
    self._check_game_over()
    if not self.game_over:
        self.run_worker(self._bot_move, exclusive=True)   # async worker

async def _bot_move(self):
    self.status = "Bot thinking…"
    self.refresh()
    # Run MCTS in thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    action = await loop.run_in_executor(None, self._run_mcts)
    # Apply bot move
    move = action_to_move(action)
    if self.board_state.board.turn == chess.BLACK:
        move = _flip_move(move)
    self.board_state, self.cur_player = self.game.getNextState(
        self.board_state, self.cur_player, action
    )
    self.moves_played.append(move)
    self._check_game_over()
    self.status = "Your turn"
    self.refresh()
```

**PGN save (same as before, triggered on game end):**
```python
def _save_pgn(self, result: str):
    os.makedirs(self.pgn_dir, exist_ok=True)
    pgn_game = chess.pgn.Game()
    pgn_game.headers.update({
        "White": "Human" if self.human_is_white else "ChessBot-AZ",
        "Black": "ChessBot-AZ" if self.human_is_white else "Human",
        "Result": result,
    })
    node = pgn_game
    for move in self.moves_played:
        node = node.add_variation(move)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(self.pgn_dir, f"game_{ts}.pgn")
    with open(path, "w") as f:
        print(pgn_game, file=f)
    self.notify(f"Game saved to {path}")   # Textual toast notification
    return path
```

---

### New dependency
```bash
uv add textual
```

### CLI args

| Arg | Default | Note |
|-----|---------|------|
| `--mode` | required | `replay` or `play` |
| `--pgn` | — | path to PGN file (replay mode) |
| `--side` | `white` | `white` or `black` (play mode) |
| `--checkpoint-dir` | `./checkpoints` | |
| `--checkpoint-file` | `best.pth.tar` | |
| `--mcts-sims` | `100` | smaller = more responsive |
| `--num-channels` | `256` | |
| `--num-res-blocks` | `20` | |
| `--pgn-dir` | `./games` | save location for live games |

---

### Verification checks

```bash
# 1. Install textual and verify import
uv add textual
uv run python -c "import textual; import tui; print('tui imports OK')"

# 2. Replay mode headless test (Textual supports --screenshot for CI)
#    Requires a PGN file; create a minimal one first:
uv run python -c "
import chess.pgn
g = chess.pgn.Game()
g.headers.update({'White':'A','Black':'B','Result':'1-0'})
node = g
for uci in ['e2e4','e7e5','d1h5','b8c6','f1c4','g8f6','h5f7']:
    node = node.add_variation(chess.Move.from_uci(uci))
with open('/tmp/test.pgn','w') as f: print(g, file=f)
print('test PGN written')
"
uv run python tui.py --mode replay --pgn /tmp/test.pgn
# Expected: board displays, ← → keys step through Scholar's mate moves

# 3. Play mode startup (--mcts-sims 5 for fast response)
uv run python tui.py --mode play --mcts-sims 5 --side white
# Expected:
#   - Starting board shows
#   - Status: "Your turn"
#   - Type "e2e4" in input, press Enter/Submit → board updates, bot responds
#   - Status shows "Bot thinking…" then returns to "Your turn"
#   - Illegal move (e.g. "e2e5") → status shows error, board unchanged
#   - Auto-promotion: type "e7e8" when legal → promotes to queen

# 4. Game-end and PGN save (trigger Scholar's mate as black)
#    Play as black, let white (bot) win; or play moves manually to reach mate
#    Expected: result banner shown, ./games/game_<ts>.pgn written

# 5. Validate saved PGN
uv run python -c "
import chess.pgn, glob, os
files = sorted(glob.glob('./games/*.pgn'))
assert files, 'No PGN files found in ./games/'
with open(files[-1]) as f:
    g = chess.pgn.read_game(f)
assert g is not None, 'Failed to parse PGN'
result = g.headers.get('Result')
assert result in {'1-0','0-1','1/2-1/2'}, f'Unexpected result: {result}'
moves = list(g.mainline_moves())
print(f'PGN OK: {len(moves)} moves, result={result}')
"

# 6. Keyboard bindings work in replay mode
#    - q quits cleanly
#    - left/right arrows step moves
#    (manual check; Textual does not expose a headless key-test API easily)
```

---

## Step 11 — `uci_engine.py` (UCI wrapper)

### What it does
Implements the [UCI protocol](https://www.chessprogramming.org/UCI) over stdin/stdout so the bot can be launched as a subprocess by cutechess-cli or any other UCI-compatible GUI/tool.

### UCI protocol flow
```
Host                    uci_engine.py
  uci              →
                   ←    id name ChessBot-AZ
                   ←    id author <user>
                   ←    uciok
  isready          →
                   ←    readyok
  position startpos moves e2e4 e7e5
                   →    (update internal board)
  go movetime 100  →
                   ←    bestmove e2e4
  quit             →    sys.exit(0)
```

### Implementation plan

**Startup:** load model once:
```python
# Parse CLI args: --checkpoint-dir, --checkpoint-file, --mcts-sims, --num-channels, --num-res-blocks
game = ChessGame()
nnet = ChessNNet(game, args)
nnet.load_checkpoint(args.checkpoint_dir, args.checkpoint_file)
```

**`position` handler:**
```python
# "position startpos moves e2e4 e7e5 ..."
# "position fen <fen> [moves ...]"
board = chess.Board()
if "fen" in tokens: board.set_fen(fen_string)
for uci_str in move_tokens:
    board.push_uci(uci_str)
board_state = ChessBoardState(board)
cur_player = 1 if board.turn == chess.WHITE else -1
```

**`go` handler:**
```python
# Fresh MCTS per move (no tree reuse — simple and correct)
mcts = MCTS(game, nnet, args)
canonical = game.getCanonicalForm(board_state, cur_player)
probs = mcts.getActionProb(canonical, temp=0)
action = int(np.argmax(probs))
move = action_to_move(action)
if board_state.board.turn == chess.BLACK:
    move = _flip_move(move)      # un-flip from canonical → real coordinates
print(f"bestmove {move.uci()}", flush=True)
```

**Main loop:**
```python
while True:
    line = sys.stdin.readline()
    if not line: break
    line = line.strip()
    if line == "uci":         handle_uci()
    elif line == "isready":   print("readyok", flush=True)
    elif line.startswith("position"): handle_position(line)
    elif line.startswith("go"):       handle_go(line)
    elif line == "quit":      sys.exit(0)
    # setoption / stop / ponderhit: silently ignored
```

**CLI args for uci_engine.py:**

| Arg | Default |
|-----|---------|
| `--checkpoint-dir` | `./checkpoints` |
| `--checkpoint-file` | `best.pth.tar` |
| `--mcts-sims` | `200` |
| `--num-channels` | `256` |
| `--num-res-blocks` | `20` |

### Verification checks

```bash
# 1. uci handshake
echo -e "uci\nquit" | uv run python uci_engine.py --mcts-sims 5
# Expected output contains: "id name ChessBot-AZ" and "uciok"

# 2. isready
echo -e "uci\nisready\nquit" | uv run python uci_engine.py --mcts-sims 5
# Expected: "readyok" on its own line

# 3. Full move from start position
echo -e "uci\nisready\nposition startpos\ngo movetime 100\nquit" \
  | uv run python uci_engine.py --mcts-sims 5
# Expected: "bestmove <uci>" where <uci> is a valid opening move (e.g. e2e4)

# 4. Move after 1.e4 e5 (playing as black)
echo -e "uci\nisready\nposition startpos moves e2e4 e7e5\ngo movetime 100\nquit" \
  | uv run python uci_engine.py --mcts-sims 5
# Expected: "bestmove <uci>" — a valid black move

# 5. Validate bestmove is always a legal UCI move
uv run python -c "
import subprocess, chess
proc = subprocess.run(
    ['uv', 'run', 'python', 'uci_engine.py', '--mcts-sims', '5'],
    input='uci\nisready\nposition startpos\ngo movetime 100\nquit\n',
    capture_output=True, text=True
)
lines = proc.stdout.strip().splitlines()
bm = next(l for l in lines if l.startswith('bestmove'))
uci = bm.split()[1]
board = chess.Board()
move = chess.Move.from_uci(uci)
assert move in board.legal_moves, f'{uci} is not legal'
print(f'bestmove {uci} is legal  OK')
"
```

---

## Step 12 — `elo.py` (cutechess-cli ELO evaluation)

### What it does
Constructs and launches a `cutechess-cli` command to run automated matches between the bot and an opponent (Stockfish or another checkpoint). Streams live output and prints final ELO estimate.

### cutechess-cli invocation

```bash
cutechess-cli \
  -engine cmd="uv run python uci_engine.py" dir=. proto=uci name=ChessBot-AZ \
  -engine cmd=stockfish proto=uci name=Stockfish \
  -each tc=40/60 \
  -games 100 \
  -concurrency 1 \
  -pgnout ./matches/match_<timestamp>.pgn \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05
```

When `--engine2-elo` is set, prepend Stockfish options:
```bash
-engine cmd=stockfish option.UCI_LimitStrength=true option.UCI_Elo=1500 proto=uci name=Stockfish-1500
```

### Implementation plan

```python
def build_command(args) -> list[str]:
    cmd = [args.cutechess_path]
    # Engine 1: our bot
    cmd += ["-engine",
            f"cmd={args.engine1_cmd}", "dir=.", "proto=uci", "name=ChessBot-AZ"]
    # Engine 2: opponent
    e2 = [f"cmd={args.engine2_cmd}", "proto=uci", f"name={args.engine2_name}"]
    if args.engine2_elo:
        e2 += [f"option.UCI_LimitStrength=true",
               f"option.UCI_Elo={args.engine2_elo}"]
    cmd += ["-engine"] + e2
    # Match settings
    cmd += ["-each", f"tc={args.tc}"]
    cmd += ["-games", str(args.games)]
    cmd += ["-concurrency", str(args.concurrency)]
    cmd += ["-pgnout", args.pgn_out]
    if args.sprt:
        cmd += ["-sprt", "elo0=0", "elo1=5", "alpha=0.05", "beta=0.05"]
    return cmd

# Stream output live
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
```

**CLI args for elo.py:**

| Arg | Default | Note |
|-----|---------|------|
| `--engine1-cmd` | `uv run python uci_engine.py` | our bot |
| `--engine2-cmd` | `stockfish` | opponent binary |
| `--engine2-name` | `Stockfish` | display name |
| `--engine2-elo` | `None` | Stockfish UCI_Elo (1320–3190) |
| `--games` | `100` | |
| `--tc` | `40/60` | time control |
| `--concurrency` | `1` | parallel games |
| `--pgn-out` | auto-timestamped | |
| `--sprt` | `False` | enable SPRT stopping rule |
| `--cutechess-path` | `cutechess-cli` | binary on PATH |
| `--dry-run` | `False` | print command, do not execute |

**Suggested Stockfish ELO presets (document in comments):**

| UCI_Elo | Approximate skill |
|---------|-------------------|
| 1320 | Beginner |
| 1500 | Casual club |
| 1800 | Intermediate club |
| 2000 | Strong amateur |
| 2500 | Master level |
| 3190 | Full Stockfish |

### Verification checks

```bash
# 1. Import check
uv run python -c "import elo; print('elo imports OK')"

# 2. Dry-run: print command without executing
uv run python elo.py --games 4 --tc 40/10 --dry-run
# Expected: prints the full cutechess-cli command, exits cleanly

# 3. Graceful missing cutechess-cli
uv run python elo.py --cutechess-path nonexistent_binary --games 4 --dry-run
# Expected: prints "cutechess-cli not found" message (only when --dry-run is off)

# 4. Full run (requires cutechess-cli installed and Stockfish on PATH)
uv run python elo.py \
  --games 10 \
  --tc 40/10 \
  --engine2-elo 1320 \
  --pgn-out ./matches/test_match.pgn
# Expected: live score output, final result printed, PGN file written

# 5. Validate output PGN
uv run python -c "
import chess.pgn
with open('./matches/test_match.pgn') as f:
    games = []
    while True:
        g = chess.pgn.read_game(f)
        if g is None: break
        games.append(g)
print(f'{len(games)} games in PGN  OK')
assert all(g.headers.get('Result') in {'1-0','0-1','1/2-1/2'} for g in games)
print('All results valid  OK')
"
```

---

## Hyperparameters (train.py)

| Param | Value | Note |
|-------|-------|------|
| numIters | 1000 | training iterations |
| numEps | 100 | self-play episodes per iter |
| tempThreshold | 30 | moves before τ→0 |
| updateThreshold | 0.55 | win-rate to accept new model |
| maxlenOfQueue | 200000 | max training examples kept |
| numMCTSSims | 800 | MCTS simulations per move |
| arenaCompare | 40 | arena games per evaluation |
| cpuct | 1.0 | UCB exploration constant |
| epochs | 10 | gradient steps per train call |
| batch_size | 512 | |
| num_channels | 256 | ResNet filter count |
| num_res_blocks | 20 | ResNet depth |
| l2_reg | 1e-4 | weight decay |
