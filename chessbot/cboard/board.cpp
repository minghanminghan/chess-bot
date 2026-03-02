/*
 * board.cpp — pybind11 wrapper around MidnightMoveGen::Position
 *
 * Exposes class BoardWrapper (Python name "Position") with:
 *  - Action table mirroring ChessBoard.py's _build_action_maps()
 *  - apply(action_idx) / undo()  with full make-unmake
 *  - 8-step history ring buffer for AlphaZero feature planes
 *  - to_tensor(canonical)  → numpy float32 (119,8,8)
 *  - valid_moves_mask()    → numpy float32 (4672,)
 *  - is_game_over() / result()
 *  - string_representation(), copy(), set_fen(), push_uci()
 */

#include <bitset>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include "move_gen.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace Midnight;

// ─────────────────────────────────────────────────────────────────────────────
// Action table
//   Mirrors ChessBoard.py _build_action_maps() exactly.
//   Square convention: sq = rank*8 + file  (same as python-chess).
//   Promo codes: 0=none, 1=knight, 2=bishop, 3=rook, 4=queen.
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int ACTION_SIZE = 4672;   // 64 * 73

struct ActionEntry { int from_sq, to_sq, promo; };

static ActionEntry IDX_TO_ENTRY[ACTION_SIZE];
// Reverse map key = from_sq*512 + to_sq*8 + promo → action index (-1 = invalid)
static int ENTRY_TO_IDX[64 * 64 * 8];

// 8 queen directions: (drank, dfile)
static constexpr int QUEEN_DR[8] = { 1,  1,  0, -1, -1, -1,  0,  1};
static constexpr int QUEEN_DF[8] = { 0,  1,  1,  1,  0, -1, -1, -1};
// 8 knight deltas: (drank, dfile)
static constexpr int KNIGHT_DR[8] = { 2,  2, -2, -2,  1,  1, -1, -1};
static constexpr int KNIGHT_DF[8] = { 1, -1,  1, -1,  2, -2,  2, -2};
// Under-promotions: promo codes 1,2,3; dfile offsets -1,0,+1
static constexpr int UNDERPROMO_PROMO[3] = {1, 2, 3};
static constexpr int UNDERPROMO_DF[3]    = {-1, 0, 1};

static void init_action_tables() {
    for (int i = 0; i < ACTION_SIZE;   i++) IDX_TO_ENTRY[i] = {-1, -1, -1};
    for (int i = 0; i < 64 * 64 * 8;  i++) ENTRY_TO_IDX[i] = -1;

    for (int sq = 0; sq < 64; sq++) {
        int rank = sq / 8, file = sq % 8;
        int base = sq * 73;

        // ── Queen-style moves (slots 0..55) ──────────────────────────────
        int slot = 0;
        for (int d = 0; d < 8; d++) {
            for (int dist = 1; dist <= 7; dist++, slot++) {
                int idx = base + slot;
                int r2 = rank + QUEEN_DR[d] * dist;
                int f2 = file + QUEEN_DF[d] * dist;
                if (r2 < 0 || r2 >= 8 || f2 < 0 || f2 >= 8) continue;
                int to_sq = r2 * 8 + f2;
                // Queen promotion: white pawn from rank 6 → rank 7
                int promo = (rank == 6 && r2 == 7) ? 4 : 0;
                IDX_TO_ENTRY[idx] = {sq, to_sq, promo};
                // Forward map: first occurrence wins (matches Python)
                {
                    int key = sq * 512 + to_sq * 8 + promo;
                    if (ENTRY_TO_IDX[key] < 0) ENTRY_TO_IDX[key] = idx;
                }
                // Bare (no-promo) key also maps here (Python m_bare logic)
                if (promo != 0) {
                    int key2 = sq * 512 + to_sq * 8;
                    if (ENTRY_TO_IDX[key2] < 0) ENTRY_TO_IDX[key2] = idx;
                }
            }
        }

        // ── Knight moves (slots 56..63) ───────────────────────────────────
        for (int k = 0; k < 8; k++) {
            int idx = base + 56 + k;
            int r2 = rank + KNIGHT_DR[k];
            int f2 = file + KNIGHT_DF[k];
            if (r2 < 0 || r2 >= 8 || f2 < 0 || f2 >= 8) continue;
            int to_sq = r2 * 8 + f2;
            IDX_TO_ENTRY[idx] = {sq, to_sq, 0};
            int key = sq * 512 + to_sq * 8;
            if (ENTRY_TO_IDX[key] < 0) ENTRY_TO_IDX[key] = idx;
        }

        // ── Under-promotions (slots 64..72) ──────────────────────────────
        for (int p = 0; p < 3; p++) {
            for (int d = 0; d < 3; d++) {
                int idx = base + 64 + p * 3 + d;
                if (rank != 6) continue;
                int f2 = file + UNDERPROMO_DF[d];
                if (f2 < 0 || f2 >= 8) continue;
                int to_sq = 7 * 8 + f2;
                int promo = UNDERPROMO_PROMO[p];
                IDX_TO_ENTRY[idx] = {sq, to_sq, promo};
                int key = sq * 512 + to_sq * 8 + promo;
                if (ENTRY_TO_IDX[key] < 0) ENTRY_TO_IDX[key] = idx;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Promotion code from a Midnight MoveType
// ─────────────────────────────────────────────────────────────────────────────
static inline int promo_code(Move m) {
    MoveType t = m.type() & static_cast<MoveType>(~CAPTURE_TYPE);
    switch (t) {
        case PR_KNIGHT: return 1;
        case PR_BISHOP: return 2;
        case PR_ROOK:   return 3;
        case PR_QUEEN:  return 4;
        default:        return 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// History ring buffer
// ─────────────────────────────────────────────────────────────────────────────

struct HistoryFrame {
    float piece_planes[12][8][8];  // stored in REAL (un-flipped) coordinates
    bool  rep1, rep2;
};

struct UndoHistEntry {
    HistoryFrame evicted;
    bool had_eviction;
};

// Piece type order matching Python's PIECE_TO_PLANE dict (P,N,B,R,Q,K)
static constexpr PieceType PIECE_ORDER[6] = {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING};

static void fill_piece_planes_real(const Position& pos, HistoryFrame& fr) {
    for (int c = 0; c < 2; c++) {
        for (int pt = 0; pt < 6; pt++) {
            float (*plane)[8] = fr.piece_planes[c * 6 + pt];
            memset(plane, 0, 8 * 8 * sizeof(float));
            Bitboard bb = pos.piece_bb(Color(c), PIECE_ORDER[pt]);
            while (bb) {
                int sq = __builtin_ctzll(bb);
                bb &= bb - 1ULL;
                plane[sq / 8][sq % 8] = 1.0f;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small UCI / FEN helpers
// ─────────────────────────────────────────────────────────────────────────────
static inline int parse_uci_sq(char file_c, char rank_c) {
    return (rank_c - '1') * 8 + (file_c - 'a');
}

static inline int char_to_promo(char c) {
    switch (c) {
        case 'n': return 1; case 'b': return 2;
        case 'r': return 3; case 'q': return 4;
        default:  return 0;
    }
}

// Return the Nth space-delimited token (0-indexed) from a FEN string.
static std::string fen_token(const std::string& fen, int n) {
    int spaces = 0;
    size_t start = 0;
    for (size_t i = 0; i <= fen.size(); i++) {
        if (i == fen.size() || fen[i] == ' ') {
            if (spaces == n) return fen.substr(start, i - start);
            spaces++;
            start = i + 1;
        }
    }
    return "";
}

static int parse_fullmove(const std::string& fen) {
    std::string t = fen_token(fen, 5);
    return t.empty() ? 1 : std::stoi(t);
}

static int parse_halfmove(const std::string& fen) {
    std::string t = fen_token(fen, 4);
    return t.empty() ? 0 : std::stoi(t);
}

// ─────────────────────────────────────────────────────────────────────────────
// BoardWrapper — exposes "Position" to Python
// ─────────────────────────────────────────────────────────────────────────────

class BoardWrapper {
public:
    // Default ctor: startpos
    BoardWrapper() { reset_to_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1); }

    // Ctor from FEN
    explicit BoardWrapper(const std::string& fen) { set_fen(fen); }

    // ── set_fen ───────────────────────────────────────────────────────────
    void set_fen(const std::string& fen) {
        try { pos_.set_fen(fen); }
        catch (...) { throw std::invalid_argument("Invalid FEN: " + fen); }
        // Midnight's set_fen() ignores the halfmove clock; apply it manually.
        pos_.set_fifty_move_rule(static_cast<uint16_t>(parse_halfmove(fen)));
        move_stack_.clear();
        undo_hist_stack_.clear();
        history_size_  = 0;
        history_start_ = 0;
        fullmove_number_ = parse_fullmove(fen);
        push_frame();
    }

    // ── apply ─────────────────────────────────────────────────────────────
    void apply(int action_idx) {
        if (action_idx < 0 || action_idx >= ACTION_SIZE)
            throw std::out_of_range("action_idx out of range");
        const ActionEntry& e = IDX_TO_ENTRY[action_idx];
        if (e.from_sq < 0)
            throw std::invalid_argument("action_idx maps to off-board slot");

        // Convert canonical → real squares for black
        int flip   = (pos_.turn() == BLACK) ? 56 : 0;
        int from_r = e.from_sq ^ flip;
        int to_r   = e.to_sq   ^ flip;

        Move m = find_legal(from_r, to_r, e.promo);

        // Save undo info BEFORE modifying history
        UndoHistEntry ue;
        ue.had_eviction = (history_size_ == 8);
        if (ue.had_eviction) ue.evicted = history_frames_[history_start_];

        move_stack_.push_back(m);
        undo_hist_stack_.push_back(ue);

        pos_.play(m);
        if (pos_.turn() == WHITE) fullmove_number_++;  // black just moved
        push_frame();
    }

    // ── undo ──────────────────────────────────────────────────────────────
    void undo() {
        if (move_stack_.empty()) throw std::runtime_error("undo() on empty stack");
        Move m = move_stack_.back(); move_stack_.pop_back();
        bool was_black = (pos_.turn() == WHITE);  // white to move → black made it
        pop_frame();
        pos_.undo(m);
        if (was_black) fullmove_number_--;
    }

    // ── copy ──────────────────────────────────────────────────────────────
    BoardWrapper copy() const {
        BoardWrapper b;
        b.pos_            = pos_;
        b.move_stack_     = move_stack_;
        b.history_frames_ = history_frames_;
        b.history_start_  = history_start_;
        b.history_size_   = history_size_;
        b.undo_hist_stack_= undo_hist_stack_;
        b.fullmove_number_= fullmove_number_;
        return b;
    }

    // ── to_tensor ─────────────────────────────────────────────────────────
    py::array_t<float> to_tensor(bool canonical = true) const {
        const bool flip = canonical && (pos_.turn() == BLACK);
        auto arr = py::array_t<float>({119, 8, 8});
        float* data = arr.mutable_data();
        memset(data, 0, 119 * 64 * sizeof(float));

        // Planes 0..111: 8 history steps × 14 planes.
        // Python pads zeros at the FRONT (far past), newest is always at t=7.
        // history_t = t - (8 - history_size_) maps tensor-t to ring-buffer index.
        for (int t = 0; t < 8; t++) {
            int history_t = t - (8 - history_size_);
            if (history_t < 0) continue;               // zero-pad far past
            int slot = (history_start_ + history_t) % 8;
            const HistoryFrame& fr = history_frames_[slot];
            int off = t * 14;

            for (int p = 0; p < 12; p++) {
                float* dst = data + (off + p) * 64;
                if (flip) {
                    for (int r = 0; r < 8; r++)
                        for (int f = 0; f < 8; f++)
                            dst[(7 - r) * 8 + f] = fr.piece_planes[p][r][f];
                } else {
                    memcpy(dst, fr.piece_planes[p], 64 * sizeof(float));
                }
            }
            if (fr.rep1) { float* d = data + (off+12)*64; for(int i=0;i<64;i++) d[i]=1.f; }
            if (fr.rep2) { float* d = data + (off+13)*64; for(int i=0;i<64;i++) d[i]=1.f; }
        }

        // Scalar planes 112..118
        auto fill = [&](int plane, float v) {
            float* d = data + plane * 64;
            for (int i = 0; i < 64; i++) d[i] = v;
        };
        fill(112, (pos_.turn() == WHITE) ? 1.f : 0.f);
        fill(113, fullmove_number_ / 500.f);
        fill(114, pos_.castling_rights_ks(WHITE) ? 1.f : 0.f);
        fill(115, pos_.castling_rights_qs(WHITE) ? 1.f : 0.f);
        fill(116, pos_.castling_rights_ks(BLACK) ? 1.f : 0.f);
        fill(117, pos_.castling_rights_qs(BLACK) ? 1.f : 0.f);
        fill(118, pos_.fifty_move_rule() / 100.f);

        return arr;
    }

    // ── valid_moves_mask ──────────────────────────────────────────────────
    py::array_t<float> valid_moves_mask() const {
        auto arr = py::array_t<float>(ACTION_SIZE);
        float* data = arr.mutable_data();
        std::fill(data, data + ACTION_SIZE, 0.f);
        if (pos_.turn() == WHITE) fill_mask_<WHITE>(data);
        else                      fill_mask_<BLACK>(data);
        return arr;
    }

    // ── game status ───────────────────────────────────────────────────────
    bool is_game_over() const {
        if (pos_.fifty_move_rule() >= 100) return true;
        if (const_cast<Position&>(pos_).has_repetition(Position::THREE_FOLD)) return true;
        return pos_.legal_move_count() == 0;
    }

    float result() const {
        if (pos_.fifty_move_rule() >= 100) return 1e-4f;
        if (const_cast<Position&>(pos_).has_repetition(Position::THREE_FOLD)) return 1e-4f;
        if (pos_.legal_move_count() > 0) return 0.f;   // ongoing
        if (pos_.in_check())             return -1.f;  // mated
        return 1e-4f;                                  // stalemate
    }

    // ── accessors ─────────────────────────────────────────────────────────
    py::tuple string_representation() const {
        return py::make_tuple((long long)pos_.hash(), (int)pos_.fifty_move_rule());
    }
    int       side_to_move()    const { return (pos_.turn() == WHITE) ? 1 : -1; }
    long long hash()            const { return (long long)pos_.hash(); }
    int       halfmove_clock()  const { return (int)pos_.fifty_move_rule(); }
    int       fullmove_number() const { return fullmove_number_; }

    // ── push_uci ──────────────────────────────────────────────────────────
    void push_uci(const std::string& uci) {
        if (uci.size() < 4) throw std::invalid_argument("Bad UCI: " + uci);
        int from_r = parse_uci_sq(uci[0], uci[1]);
        int to_r   = parse_uci_sq(uci[2], uci[3]);
        int promo  = (uci.size() >= 5) ? char_to_promo(uci[4]) : 0;
        // Canonical flip for black
        int flip  = (pos_.turn() == BLACK) ? 56 : 0;
        int from_c = from_r ^ flip, to_c = to_r ^ flip;
        int key = from_c * 512 + to_c * 8 + promo;
        int idx = ENTRY_TO_IDX[key];
        if (idx < 0) throw std::invalid_argument("UCI not in action table: " + uci);
        apply(idx);
    }

private:
    Position                  pos_;
    std::vector<Move>         move_stack_;
    std::array<HistoryFrame,8>history_frames_{};
    int                       history_start_ = 0;
    int                       history_size_  = 0;
    std::vector<UndoHistEntry>undo_hist_stack_;
    int                       fullmove_number_ = 1;

    void reset_to_fen(const char* fen, int fm) {
        pos_.set_fen(fen);
        move_stack_.clear(); undo_hist_stack_.clear();
        history_size_ = 0; history_start_ = 0;
        fullmove_number_ = fm;
        push_frame();
    }

    // Push a history frame for the current position (unflipped, real coords)
    void push_frame() {
        int slot;
        if (history_size_ == 8) {
            slot = history_start_;                            // overwrite oldest
            history_start_ = (history_start_ + 1) % 8;
        } else {
            slot = (history_start_ + history_size_) % 8;
            history_size_++;
        }
        HistoryFrame& fr = history_frames_[slot];
        fill_piece_planes_real(pos_, fr);
        fr.rep1 = const_cast<Position&>(pos_).has_repetition(Position::TWO_FOLD);
        fr.rep2 = const_cast<Position&>(pos_).has_repetition(Position::THREE_FOLD);
    }

    void pop_frame() {
        history_size_--;
        const UndoHistEntry& ue = undo_hist_stack_.back();
        if (ue.had_eviction) {
            history_start_ = (history_start_ - 1 + 8) % 8;
            history_frames_[history_start_] = ue.evicted;
            history_size_++;
        }
        undo_hist_stack_.pop_back();
    }

    // Find a legal move matching from/to/promo (O(~30) iterations).
    // When promo_c==4 (queen-promo slot in action table), non-pawn pieces like
    // bishops making a rank-6→rank-7 move have promo_code=0; fall back to 0
    // if no promo=4 match is found.
    template<Color C>
    static bool search_movelist(Position& pos, int from_r, int to_r, int pc, Move& out) {
        for (Move m : MoveList<C, ALL>(pos)) {
            if ((int)m.from() == from_r && (int)m.to() == to_r && promo_code(m) == pc) {
                out = m; return true;
            }
        }
        return false;
    }

    Move find_legal(int from_r, int to_r, int promo_c) const {
        Position& p = const_cast<Position&>(pos_);
        Move found{};
        bool ok;
        if (pos_.turn() == WHITE) ok = search_movelist<WHITE>(p, from_r, to_r, promo_c, found);
        else                      ok = search_movelist<BLACK>(p, from_r, to_r, promo_c, found);

        if (!ok && promo_c == 4) {
            // Action table encodes rank-6→7 as queen-promo; fall back to bare move
            // for non-pawn pieces (bishop/rook/queen/knight) that also move there.
            if (pos_.turn() == WHITE) ok = search_movelist<WHITE>(p, from_r, to_r, 0, found);
            else                      ok = search_movelist<BLACK>(p, from_r, to_r, 0, found);
        }
        if (!ok) throw std::invalid_argument("No legal move for given action");
        return found;
    }

    // Private template to fill the valid-moves mask (one instantiation per colour)
    template<Color C>
    void fill_mask_(float* data) const {
        constexpr int FLIP = (C == BLACK) ? 56 : 0;
        for (Move m : MoveList<C, ALL>(const_cast<Position&>(pos_))) {
            int from_c = (int)m.from() ^ FLIP;
            int to_c   = (int)m.to()   ^ FLIP;
            int promo  = promo_code(m);
            int key    = from_c * 512 + to_c * 8 + promo;
            int idx    = ENTRY_TO_IDX[key];
            if (idx >= 0) data[idx] = 1.f;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// pybind11 module
// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(cboard, m) {
    init_action_tables();

    m.attr("ACTION_SIZE") = ACTION_SIZE;

    py::class_<BoardWrapper>(m, "Position")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def("apply",                 &BoardWrapper::apply)
        .def("undo",                  &BoardWrapper::undo)
        .def("copy",                  &BoardWrapper::copy)
        .def("valid_moves_mask",      &BoardWrapper::valid_moves_mask)
        .def("to_tensor",             &BoardWrapper::to_tensor,
             py::arg("canonical") = true)
        .def("string_representation", &BoardWrapper::string_representation)
        .def("is_game_over",          &BoardWrapper::is_game_over)
        .def("result",                &BoardWrapper::result)
        .def("side_to_move",          &BoardWrapper::side_to_move)
        .def("hash",                  &BoardWrapper::hash)
        .def("halfmove_clock",        &BoardWrapper::halfmove_clock)
        .def("fullmove_number",       &BoardWrapper::fullmove_number)
        .def("set_fen",               &BoardWrapper::set_fen)
        .def("push_uci",              &BoardWrapper::push_uci);
}
