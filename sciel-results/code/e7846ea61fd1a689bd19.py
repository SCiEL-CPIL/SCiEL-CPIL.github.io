import numpy as np
import time
import random
import heapq

_p = 7
_kq = 3
_Q = 49
_SLOTS = 2401
_INF = 10

_PTS_XA = np.empty(_SLOTS, dtype=np.uint8)
_PTS_XB = np.empty(_SLOTS, dtype=np.uint8)
_PTS_YA = np.empty(_SLOTS, dtype=np.uint8)
_PTS_YB = np.empty(_SLOTS, dtype=np.uint8)

def _encode(xa, xb, ya, yb):
    return ((xa*7+xb)*49)+(ya*7+yb)

def _slot_xy(xa, xb, ya, yb):
    xa %= 7; xb %= 7; ya %= 7; yb %= 7
    return _encode(xa, xb, ya, yb)

def _xy_from_slot(s):
    yb = (s % 7); s //= 7
    ya = (s % 7); s //= 7
    xb = (s % 7); s //= 7
    xa = (s % 7)
    return xa, xb, ya, yb

for xa in range(7):
    for xb in range(7):
        base_x = (xa*7+xb)
        for ya in range(7):
            for yb in range(7):
                s = base_x*49+(ya*7+yb)
                _PTS_XA[s]=xa; _PTS_XB[s]=xb; _PTS_YA[s]=ya; _PTS_YB[s]=yb

# Precompute lines: lines[d][s] = array of 48 slots on line through s in dir d (excluding s)
_lines = []
for d in range(50):
    line_arr = np.empty((2401, 48), dtype=np.int32)
    for s in range(2401):
        xa, xb, ya, yb = _xy_from_slot(s)
        for t in range(1, 49):
            if d == _INF:
                nxa = xa; nxb = xb
                nya = (ya+t) % 7; nyb = (yb+t) % 7
            else:
                da, db = (d//7, d%7)
                nxa = (xa+t) % 7
                if db == 0:
                    nxb = xb
                else:
                    inv = pow(db, 6, 7)
                    nxb = (xb + t*da*inv) % 7
                nya = (ya+t*da) % 7
                nyb = (yb+t*db) % 7
            line_arr[s, t-1] = _encode(nxa, nxb, nya, nyb)
    _lines.append(line_arr)

def _compute_clean_counts(S):
    # clean_count[x] = number of directions d such that line through x in dir d has no S points (other than x)
    # If S[x]=1, we check if other 48 are 0.
    # If S[x]=0, we check if other 48 are 0.
    # Condition: L \ {x} subset of N <=> L \ {x} has no S points.
    # So for each x, count d such that sum(S[lines[d][x]]) == 0.
    clean_counts = np.zeros(2401, dtype=np.int16)
    for x in range(2401):
        for d in range(50):
            # Check if any point on line is in S
            # lines[d][x] contains 48 points
            if not np.any(S[_lines[d][x]]):
                clean_counts[x] += 1
    return clean_counts

def _greedy_add(S, clean_counts, deadline):
    # S: bytearray or np.array of 0/1
    # clean_counts: np.array
    # Returns new S, new clean_counts, added_count
    added = 0
    candidates = list(range(2401))
    random.shuffle(candidates)
    
    # We need to check validity quickly.
    # When we add p, we update clean_counts for x in union of lines through p.
    # Specifically, for each d, for each x in lines[d][p], clean_counts[x] -= 1.
    # If clean_counts[x] becomes 0, x is invalid.
    
    # To speed up, we can precompute the set of affected x for each p?
    # Affected x for p: all x such that p is on a line through x.
    # This is all x != p? No.
    # p is on line L_{x,d} iff x is on line L_{p,d}.
    # So affected x are all points on lines through p (in all 50 directions).
    # This is 50 * 48 = 2400 points.
    # Basically all points except p.
    
    # So adding p affects almost all points.
    # We must check if any affected x has clean_counts[x] == 1 (will become 0).
    
    # Optimization:
    # Maintain a count of how many points have clean_counts[x] == 1?
    # Or just check the affected points.
    
    # Let's just check the 2400 affected points.
    
    while candidates and time.time() < deadline:
        p = candidates.pop()
        if S[p]:
            continue
            
        # Tentative add
        # Identify affected points
        affected = set()
        for d in range(50):
            affected.update(_lines[d][p].tolist())
        
        # Check if any affected point has clean_counts == 1
        # If yes, adding p will make it 0 -> invalid.
        valid_move = True
        for x in affected:
            if clean_counts[x] == 1:
                valid_move = False
                break
        
        if valid_move:
            # Commit
            S[p] = 1
            added += 1
            # Update clean_counts
            for d in range(50):
                for x in _lines[d][p]:
                    clean_counts[x] -= 1
        else:
            # Move is invalid. 
            # Since S only grows, this move will never be valid in the future.
            # So we don't need to keep p in candidates (already popped).
            pass
            
    return S, clean_counts, added

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 580
    best_S = np.zeros(2401, dtype=np.int8)
    best_score = 0
    
    # Strategy 1: Start from empty S
    # Strategy 2: Start from Baseline S
    
    # Baseline S construction
    base_S = np.zeros(2401, dtype=np.int8)
    y_special = set()
    for u in range(7):
        for v in range(7):
            y_special.add(((u*u)%7, v))
    for s in range(2401):
        xa, xb, ya, yb = _xy_from_slot(s)
        in_unital = (2*xa + ya*ya - yb*yb*3) % 7 == 0
        in_y_special = (ya, yb) in y_special
        if in_unital and not in_y_special:
            base_S[s] = 1
            
    # Try starting from empty
    S_empty = np.zeros(2401, dtype=np.int8)
    cc_empty = _compute_clean_counts(S_empty)
    S_empty, cc_empty, added_empty = _greedy_add(S_empty, cc_empty, deadline)
    score_empty = np.sum(S_empty)
    if score_empty > best_score:
        best_score = score_empty
        best_S = S_empty.copy()
        
    # Try starting from baseline
    if time.time() < deadline:
        S_base = base_S.copy()
        cc_base = _compute_clean_counts(S_base)
        S_base, cc_base, added_base = _greedy_add(S_base, cc_base, deadline)
        score_base = np.sum(S_base)
        if score_base > best_score:
            best_score = score_base
            best_S = S_base.copy()

    # Random restarts from empty
    seed = 123
    while time.time() < deadline:
        S_r = np.zeros(2401, dtype=np.int8)
        cc_r = _compute_clean_counts(S_r)
        # Random shuffle in _greedy_add handles randomness
        # But _greedy_add uses global random. Let's seed it.
        random.seed(seed)
        S_r, cc_r, added_r = _greedy_add(S_r, cc_r, deadline)
        score_r = np.sum(S_r)
        if score_r > best_score:
            best_score = score_r
            best_S = S_r.copy()
        seed += 1
        
    # Convert best_S to N (complement)
    N = np.zeros(2401 - best_score, dtype=np.int64).reshape(-1, 2, 2)
    idx = 0
    for s in range(2401):
        if not best_S[s]:
            xa, xb, ya, yb = _xy_from_slot(s)
            N[idx, 0, 0] = xa
            N[idx, 0, 1] = xb
            N[idx, 1, 0] = ya
            N[idx, 1, 1] = yb
            idx += 1
            
    return N
