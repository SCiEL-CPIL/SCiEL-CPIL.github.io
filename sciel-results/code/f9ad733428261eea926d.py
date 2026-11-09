import numpy as np
import time

_P = 7
_Q = 49
_SLOTS = 2401
_D = 50
_INF = 10

# ---------- point <-> slot ----------
def _xy(s):
    yb = s % 7; s //= 7
    ya = s % 7; s //= 7
    xb = s % 7; s //= 7
    xa = s % 7
    return xa, xb, ya, yb

def _enc(xa, xb, ya, yb):
    return ((xa*7+xb)*49)+(ya*7+yb)

# ---------- directions ----------
# dir d in 0..48: v=(1, d)  where d = da + db*alpha, da=d//7, db=d%7
# dir 49: v=(0,1)
def _dir_vec(d):
    if d == _INF:
        return (0, 0, 0, 1)  # dx_a, dx_b, dy_a, dy_b
    da = d // 7
    db = d % 7
    return (1, da, 0, db)

# Precompute per-direction step arrays:
# steps[d][s][t] = slot reached from s by + (t+1)*v_d, t=0..47
_steps = []
# line membership: line_id[d][s] and line_slots[d] (list of 49 lines x 49 slots)
_line_id = []
_line_slots = []

for d in range(_D):
    da, db, _ya, _yb = _dir_vec(d)
    # step arrays
    st = np.empty((_SLOTS, 48), dtype=np.int32)
    for s in range(_SLOTS):
        xa, xb, ya, yb = _xy(s)
        for t in range(1, 49):
            if d == _INF:
                nxa, nxb = xa, xb
                nya = (ya + t) % 7
                nyb = (yb + t) % 7
            else:
                nxa = (xa + t) % 7
                if db == 0:
                    nxb = xb
                else:
                    inv = pow(db, 6, 7)
                    nxb = (xb + t*da*inv) % 7
                nya = (ya + t*da) % 7
                nyb = (yb + t*db) % 7
            st[s, t-1] = _enc(nxa, nxb, nya, nyb)
    _steps.append(st)

    # canonical line id per slot: min slot on the line
    line_of = np.zeros(_SLOTS, dtype=np.int32)
    # build lines
    seen = {}
    lines = []
    for s in range(_SLOTS):
        if s in seen:
            line_of[s] = seen[s]
            continue
        # line through s: s + t*v for t in 0..48
        slots = [s] + [int(st[s, t]) for t in range(48)]
        lid = len(lines)
        lines.append(slots)
        for x in slots:
            seen[x] = lid
        line_of[s] = lid
    _line_id.append(line_of)
    _line_slots.append(np.array(lines, dtype=np.int32))  # shape (49,49)

# ---------- vectorized validation ----------
def _is_valid(S):
    """S: bytes/bytearray/np array of 0/1 length 2401.
    Every point s must lie on some line with >=2 OTHER S points."""
    Sv = np.frombuffer(S, dtype=np.uint8) if isinstance(S, (bytes, bytearray)) else S
    for d in range(_D):
        ls = _line_slots[d]          # (49,49)
        cnt = ls.sum(axis=1)         # S per line (uint8 sum)
        # for each line, other-count for member = cnt - S[member]
        # a point is covered by dir d if cnt[line_of[s]] - S[s] >= 2
        covered = np.zeros(_SLOTS, dtype=bool)
        # vectorize: for each line i, members ls[i], need cnt[i]-S[member]>=2
        # build per-member via fancy:
        all_members = ls.reshape(-1)          # 2401
        line_idx = np.repeat(np.arange(49), 49)
        c = cnt[line_idx]                     # 2401
        s_on = Sv[all_members]
        covered = (c - s_on) >= 2
        # if all covered by this dir, done
        if covered.all():
            return True
        # else accumulate
        if d == 0:
            acc = covered
        else:
            acc |= covered
    return bool(acc.all())

def _validate_fast(S):
    """returns (valid, n_bad). n_bad = #points with no clean line."""
    Sv = np.frombuffer(S, dtype=np.uint8) if isinstance(S, (bytes, bytearray)) else S
    acc = np.zeros(_SLOTS, dtype=bool)
    for d in range(_D):
        ls = _line_slots[d]
        cnt = ls.sum(axis=1)
        all_members = ls.reshape(-1)
        line_idx = np.repeat(np.arange(49), 49)
        c = cnt[line_idx]
        s_on = Sv[all_members]
        covered = (c - s_on) >= 2
        acc |= covered
    if acc.all():
        return True, 0
    return False, int((~acc).sum())

# ---------- deterministic large-S baseline: union of 49 lines (dir 0) ----------
def _baseline_S():
    S = bytearray(_SLOTS)
    ls = _line_slots[0]   # 49 lines of dir 0
    for i in range(49):
        for s in ls[i]:
            S[int(s)] = 1
    return S

# ---------- to output ----------
def _to_array(S):
    S = np.frombuffer(S, dtype=np.uint8) if isinstance(S, (bytes, bytearray)) else S
    mask = S == 0
    pts = []
    for s in np.where(mask)[0]:
        xa, xb, ya, yb = _xy(int(s))
        pts.append((xa, xb, ya, yb))
    if not pts:
        # must be nonempty; take slot 0
        xa, xb, ya, yb = _xy(0)
        pts = [(xa, xb, ya, yb)]
    return np.array(pts, dtype=np.int64).reshape(-1, 2, 2)

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 560
    # Start from deterministic baseline (49-line union) -> valid, large S.
    best_S = bytearray(_baseline_S())
    valid, nbad = _validate_fast(best_S)
    if not valid:
        # extremely unlikely; fall back to a single-line S (guaranteed valid)
        S = bytearray(_SLOTS)
        for s in _line_slots[0][0]:
            S[int(s)] = 1
        best_S = S
    best_score = int(sum(best_S))

    rng = np.random.default_rng(12345)

    # Local search: try swaps (move an S point to a non-S point) and inserts.
    # A move is accepted only if it keeps validity and increases |S|.
    # Represent S as np.uint8 array for speed.
    cur = np.frombuffer(bytes(best_S), dtype=np.uint8).copy()
    cur_score = int(cur.sum())

    S_slots = np.arange(_SLOTS)
    # Precompute, for each slot, the 50 line-ids it belongs to (for fast delta eval).
    # line_id[d][s] gives line index; we need per-slot list of (d, line_idx).
    slot_lines = []
    for s in range(_SLOTS):
        lst = []
        for d in range(_D):
            lst.append((d, int(_line_id[d][s])))
        slot_lines.append(lst)

    def _try_move(S, out_s, in_s, cur_score):
        """Return (new_S, new_score) if moving out_s->in_s keeps valid & improves, else None."""
        if S[out_s] == 0 or S[in_s] == 1:
            return None
        # tentatively apply
        S[out_s] = 0
        S[in_s] = 1
        nv, _ = _validate_fast(S)
        if nv:
            return S, cur_score  # size unchanged; only accept if score improved? size same.
        # revert
        S[out_s] = 1
        S[in_s] = 0
        return None

    # Main loop: random restarts of local search + occasional insert attempts.
    it = 0
    while time.time() < deadline:
        it += 1
        # --- local search on cur ---
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try a batch of random swaps
            for _ in range(400):
                if time.time() > deadline:
                    break
                a = int(rng.integers(_SLOTS))
                b = int(rng.integers(_SLOTS))
                if a == b:
                    continue
                # only consider swapping an S point with a non-S point
                if cur[a] == 1 and cur[b] == 0:
                    S = cur.copy()
                    S[a] = 0
                    S[b] = 1
                    nv, _ = _validate_fast(S)
                    if nv:
                        # size unchanged -> not an improvement in score; skip (we want larger S)
                        pass
                # insert: add a non-S point if still valid (increases |S|)
                if cur[b] == 0:
                    S = cur.copy()
                    S[b] = 1
                    nv, _ = _validate_fast(S)
                    if nv:
                        cur = S
                        cur_score += 1
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_S = bytearray(cur)
            # delete: remove an S point if still valid (shrinks S; only useful as a move step)
            # We skip deletes for score, but allow them to escape local optima via swaps.
            # Try swaps that keep validity (size-neutral) to relocate, enabling later inserts.
            for _ in range(200):
                if time.time() > deadline:
                    break
                a = int(rng.integers(_SLOTS))
                b = int(rng.integers(_SLOTS))
                if a == b or cur[a] != 1 or cur[b] != 0:
                    continue
                S = cur.copy()
                S[a] = 0
                S[b] = 1
                nv, _ = _validate_fast(S)
                if nv:
                    cur = S
                    improved = True
        # --- random restart: new baseline + few inserts ---
        if it % 3 == 0:
            S = np.frombuffer(bytes(_baseline_S()), dtype=np.uint8).copy()
            sc = int(S.sum())
            for _ in range(150):
                if time.time() > deadline:
                    break
                b = int(rng.integers(_SLOTS))
                if S[b] == 0:
                    S[b] = 1
                    nv, _ = _validate_fast(S)
                    if nv:
                        sc += 1
                    else:
                        S[b] = 0
            if sc > best_score:
                best_score = sc
                best_S = bytearray(S)
                cur = S
                cur_score = sc

    # Final safety: ensure nonempty N and validity of returned best.
    if best_score >= _SLOTS:
        # remove one point to make N nonempty; verify
        S = np.frombuffer(bytes(best_S), dtype=np.uint8).copy()
        # find a removable point
        for s in range(_SLOTS):
            S[s] = 0
            nv, _ = _validate_fast(S)
            if nv:
                best_S = bytearray(S)
                best_score -= 1
                break
    valid, _ = _validate_fast(best_S)
    if not valid:
        # ultimate fallback: single line (guaranteed valid, nonempty N)
        S = bytearray(_SLOTS)
        for s in _line_slots[0][0]:
            S[int(s)] = 1
        best_S = S
    return _to_array(bytearray(best_S))
