"""
UI utilities: convert between integer action indices and UCI move strings.

No python-chess dependency — pure integer/string operations on the same
action encoding used by board.cpp (64 squares × 73 move types = 4672 actions).

For use by tui.py and uci_engine.py only; not imported on the training path.
"""

_QUEEN_DIRS    = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
_KNIGHT_DELTAS = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
_UPROMO_CHARS  = ['n', 'b', 'r']   # underpromotion pieces (no queen — queen is default)
_UPROMO_DFILES = [-1, 0, 1]


def _sq(rank: int, file: int) -> int:
    return rank * 8 + file

def _sq_to_str(sq: int) -> str:
    rank, file = divmod(sq, 8)
    return chr(ord('a') + file) + str(rank + 1)

def _str_to_sq(file_c: str, rank_c: str) -> int:
    return (int(rank_c) - 1) * 8 + (ord(file_c) - ord('a'))

def _flip(sq: int) -> int:
    """Flip rank: converts between real and canonical coordinates for black."""
    return sq ^ 56


def _build():
    idx_to = {}   # action_idx -> (from_sq, to_sq, promo_char or None)
    to_idx = {}   # (from_sq, to_sq, promo_char or None) -> action_idx

    for s in range(64):
        r, f = divmod(s, 8)
        base = s * 73
        slot = 0

        # Queen-style slides (8 directions × 7 distances = 56 slots)
        for dr, df in _QUEEN_DIRS:
            for dist in range(1, 8):
                idx = base + slot
                slot += 1
                r2, f2 = r + dr * dist, f + df * dist
                if 0 <= r2 < 8 and 0 <= f2 < 8:
                    t = _sq(r2, f2)
                    promo = 'q' if (r == 6 and r2 == 7) else None
                    idx_to[idx] = (s, t, promo)
                    to_idx.setdefault((s, t, promo), idx)
                    if promo is None:
                        to_idx.setdefault((s, t, None), idx)

        # Knight jumps (8 slots)
        for k, (dr, df) in enumerate(_KNIGHT_DELTAS):
            idx = base + 56 + k
            r2, f2 = r + dr, f + df
            if 0 <= r2 < 8 and 0 <= f2 < 8:
                t = _sq(r2, f2)
                idx_to[idx] = (s, t, None)
                to_idx.setdefault((s, t, None), idx)

        # Underpromotions (3 pieces × 3 file-deltas = 9 slots)
        for pi, pc in enumerate(_UPROMO_CHARS):
            for di, df in enumerate(_UPROMO_DFILES):
                idx = base + 64 + pi * 3 + di
                r2, f2 = r + 1, f + df
                if r == 6 and 0 <= f2 < 8:
                    t = _sq(r2, f2)
                    idx_to[idx] = (s, t, pc)
                    to_idx[(s, t, pc)] = idx

    return idx_to, to_idx

_IDX_TO, _TO_IDX = _build()


def action_to_uci(action: int, is_black: bool = False) -> str | None:
    """
    Convert an action index (canonical / current-player space) to a UCI string.

    is_black=True flips the squares back to real board coordinates.
    Returns None if action is out of range.
    """
    entry = _IDX_TO.get(action)
    if entry is None:
        return None
    from_sq, to_sq, promo = entry
    if is_black:
        from_sq = _flip(from_sq)
        to_sq   = _flip(to_sq)
    uci = _sq_to_str(from_sq) + _sq_to_str(to_sq)
    if promo:
        uci += promo
    return uci


def uci_to_action(uci: str, is_black: bool = False) -> int | None:
    """
    Convert a UCI move string to an action index (canonical space).

    is_black=True flips the squares into canonical coordinates before lookup.
    Returns None if the move is not in the action table.
    """
    from_sq = _str_to_sq(uci[0], uci[1])
    to_sq   = _str_to_sq(uci[2], uci[3])
    promo   = uci[4] if len(uci) > 4 else None
    if is_black:
        from_sq = _flip(from_sq)
        to_sq   = _flip(to_sq)
    key = (from_sq, to_sq, promo)
    if key in _TO_IDX:
        return _TO_IDX[key]
    # Queen promotion is the default for pawn-to-backrank with no suffix
    if promo is None:
        key_q = (from_sq, to_sq, 'q')
        if key_q in _TO_IDX:
            return _TO_IDX[key_q]
    return None


def flip_uci(uci: str) -> str:
    """Flip a UCI string between real and canonical coordinates."""
    result = _sq_to_str(_flip(_str_to_sq(uci[0], uci[1])))
    result += _sq_to_str(_flip(_str_to_sq(uci[2], uci[3])))
    if len(uci) > 4:
        result += uci[4]
    return result
