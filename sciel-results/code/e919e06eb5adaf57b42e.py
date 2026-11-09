import numpy as np
import time
import random

_p = 7
_kq = 3
_Q = 49
_SLOTS = 2401
_NUM_DIRS = 50
_INF = 10  # vertical direction index

def _encode(xa, xb, ya, yb):
    return ((xa * 7 + xb) * 49) + (ya * 7 + yb)

def _xy_from_slot(s):
    yb = s % 7; s //= 7
    ya = s % 7; s //= 7
    xb = s % 7; s //= 7
    xa = s % 7
    return xa, xb, ya, yb

# Precompute lines for each direction
# _lines[d] is a list of 49 arrays, each of length 49
_lines = []
for d in range(_NUM_DIRS):
    if d == _INF:
        dl = []
        for xa in range(7):
            for xb in range(7):
                line = []
                for ya in range(7):
                    for yb in range(7):
                        line.append(_encode(xa, xb, ya, yb))
                dl.append(np.array(line, dtype=np.int32))
    else:
        da, db = (d // 7, d % 7)
        dl = []
        for ya0 in range(7):
            for yb0 in range(7):
                line = []
                for ta in range(7):
                    for tb in range(7):
                        txa = (ta * da + tb * db * _kq) % _p
                        txb = (ta * db + tb * da) % _p
                        nxa = ta % _p
                        nxb = tb % _p
                        nya = (ya0 + txa) % _p
                        nyb = (yb0 + txb) % _p
                        line.append(_encode(nxa, nxb, nya, nyb))
                dl.append(np.array(line, dtype=np.int32))
    _lines.append(dl)

# Precompute line membership: _line_of[d][s] = index of line in _lines[d] containing slot s
_line_of = np.zeros((_NUM_DIRS, _SLOTS), dtype=np.int16)
for d in range(_NUM_DIRS):
    for li, line in enumerate(_lines[d]):
        _line_of[d, line] = li

def _compute_Scount(S_arr):
    """Compute Scount[d][s] = number of S points on line through s in direction d."""
    Scount = np.zeros((_NUM_DIRS, _SLOTS), dtype=np.int16)
    for d in range(_NUM_DIRS):
        for line in _lines[d]:
            c = int(np.sum(S_arr[line]))
            Scount[d, line] = c
    return Scount

def _is_valid(S_arr):
    """Check if S gives a valid Nikodym set (N = complement of S)."""
    Scount = _compute_Scount(S_arr)
    for s in range(_SLOTS):
        ss = S_arr[s]
        ok = False
        for d in range(_NUM_DIRS):
            if Scount[d, s] - ss == 0:
                ok = True
                break
        if not ok:
            return False
    return True

def _to_array(S_arr):
    """Convert S (bool array) to the output array of N = complement of S."""
    pts = []
    for s in range(_SLOTS):
        if not S_arr[s]:
            xa, xb, ya, yb = _xy_from_slot(s)
            pts.append([xa, xb, ya, yb])
    return np.array(pts, dtype=np.int64).reshape(-1, 2, 2)

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 570  # safety margin
    
    # Start with all points in S (N is empty, trivially valid if we can verify)
    # Actually, start with a simple valid construction.
    # Simplest valid S: empty set (N = all points). But that's not minimal.
    # Let's start with S = all points and try to remove points from S (add to N).
    # Wait, we want to MINIMIZE |N|, so MAXIMIZE |S|.
    # Start with S = all points, then try to remove points from S while maintaining validity.
    
    S_arr = np.ones(_SLOTS, dtype=np.uint8)
    
    # Verify initial state is valid
    if not _is_valid(S_arr):
        # Fallback: empty S
        S_arr = np.zeros(_SLOTS, dtype=np.uint8)
        if not _is_valid(S_arr):
            # Should never happen, but return something
            return _to_array(S_arr)
    
    Scount = _compute_Scount(S_arr)
    
    # For each point, track clean directions
    # clean[s, d] = (Scount[d, s] - S_arr[s] == 0)
    clean = np.zeros((_SLOTS, _NUM_DIRS), dtype=np.bool_)
    for s in range(_SLOTS):
        ss = S_arr[s]
        for d in range(_NUM_DIRS):
            clean[s, d] = (Scount[d, s] - ss == 0)
    
    # Count clean directions per point
    clean_count = clean.sum(axis=1)
    
    # Greedy: repeatedly find a point p in S such that removing it from S (adding to N) 
    # doesn't invalidate any point.
    # Removing p from S: S_arr[p] becomes 0.
    # For each direction d, the line through p in dir d loses one S point.
    # For each other point q on that line, Scount[d][q] decreases by 1.
    # q gains clean status for dir d if Scount[d][q] - S[q] becomes 0 (was 1 before).
    # q loses clean status? No, decreasing Scount can only help or keep same.
    # Wait: clean means Scount[d][q] - S[q] == 0.
    # If we remove p from S (p≠q), Scount[d][q] decreases by 1.
    # New value: (Scount[d][q] - 1) - S[q].
    # If old was 0, new is -1, which is NOT 0. So q LOSES clean status!
    # If old was 1, new is 0. So q GAINS clean status.
    # So removing p from S can both help and hurt.
    
    # Actually, let me think again. We want to maximize |S|, so we start with S = all and remove.
    # But removing from S means adding to N, which makes N bigger. We want N small!
    # So we want S big. Start with S big, can't make it bigger. 
    # Alternative: start with S small (N big) and add to S (shrink N).
    
    # Let me restart: start with S = empty (N = all points), and add points to S.
    S_arr = np.zeros(_SLOTS, dtype=np.uint8)
    Scount = _compute_Scount(S_arr)
    
    clean = np.zeros((_SLOTS, _NUM_DIRS), dtype=np.bool_)
    for s in range(_SLOTS):
        for d in range(_NUM_DIRS):
            clean[s, d] = (Scount[d, s] - S_arr[s] == 0)
    
    clean_count = clean.sum(axis=1)
    
    # When we add point p to S:
    # S_arr[p] becomes 1.
    # For each direction d, line through p in dir d gains one S point.
    # For each other point q on that line, Scount[d][q] increases by 1.
    # New clean for q: (Scount[d][q]+1) - S[q] == 0?
    # Old clean: Scount[d][q] - S[q] == 0.
    # If old was 0, new is 1, not 0. q LOSES clean.
    # If old was -1 (impossible since Scount >= S[q]).
    # So adding p to S: for each d, for each q on line_d(p)\{p} that was clean for d, q loses clean for d.
    # q becomes invalid if it had no other clean directions.
    
    # For p itself: Scount[d][p] increases by 1, S_arr[p] increases by 1.
    # New clean for p: (Scount[d][p]+1) - 1 = Scount[d][p]. 
    # Old clean: Scount[d][p] - 0 = Scount[d][p].
    # Same! So p's clean status doesn't change when added to S.
    
    # Strategy: find points p not in S where adding p doesn't invalidate any point.
    # A point q is invalidated if it had exactly 1 clean direction d, and p is on line_d(q)\{q}, and q was clean for d.
    
    # Precompute for efficiency
    # For each point p not in S, compute the set of (q, d) pairs that would be affected.
    # q is affected if q is on line_d(p)\{p} and clean[q, d] is True and clean_count[q] == 1.
    
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        candidates = np.where(S_arr == 0)[0]
        random.shuffle(candidates.tolist())
        
        for p in candidates:
            if time.time() > deadline - 20:
                break
            # Check if adding p to S is safe
            safe = True
            for d in range(_NUM_DIRS):
                li = _line_of[d, p]
                line = _lines[d][li]
                for q in line:
                    q = int(q)
                    if q == p:
                        continue
                    if clean[q, d] and clean_count[q] == 1:
                        safe = False
                        break
                if not safe:
                    break
            if safe:
                # Add p to S
                S_arr[p] = 1
                # Update Scount and clean
                for d in range(_NUM_DIRS):
                    li = _line_of[d, p]
                    line = _lines[d][li]
                    for q in line:
                        q = int(q)
                        Scount[d, q] += 1
                        if q != p:
                            if clean[q, d]:
                                clean[q, d] = False
                                clean_count[q] -= 1
                improved = True
    
    # Verify final state
    if not _is_valid(S_arr):
        # Fallback
        S_arr = np.zeros(_SLOTS, dtype=np.uint8)
    
    return _to_array(S_arr)
