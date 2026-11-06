"""
Minimum maximal no-isosceles set on [0, N)^2.
Strategy:
1. Precompute isosceles conflict masks for all pairs.
2. Greedy construction of a maximal independent set (no-isosceles).
3. Reduction phase: remove points if the set remains maximal.
4. Multiple random restarts.
"""
import time
import random
import numpy as np

N = 16
DEADLINE = 315.0
ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
N_TOTAL = len(ALL_POINTS)

# Precompute distances
_POINTS_ARR = np.array(ALL_POINTS, dtype=np.int32)
_px = _POINTS_ARR[:, 0]
_py = _POINTS_ARR[:, 1]
DIST2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i in range(N_TOTAL):
    for j in range(N_TOTAL):
        dx = _px[i] - _px[j]
        dy = _py[i] - _py[j]
        DIST2[i, j] = dx*dx + dy*dy

# Precompute bad[k] for each pair (i, j)
# bad[i, j] is a boolean array of length N_TOTAL
# bad[i, j][k] is True if (i, j, k) forms an isosceles triangle
# Note: i, j, k must be distinct.
BAD_FOR_PAIR = np.zeros((N_TOTAL, N_TOTAL, N_TOTAL), dtype=bool)
for i in range(N_TOTAL):
    for j in range(i + 1, N_TOTAL):
        di_j = DIST2[i, j]
        for k in range(N_TOTAL):
            if k == i or k == j:
                continue
            di_k = DIST2[i, k]
            dj_k = DIST2[j, k]
            if di_k == dj_k or di_k == di_j or dj_k == di_j:
                BAD_FOR_PAIR[i, j, k] = True
                BAD_FOR_PAIR[j, i, k] = True

def _can_add(p, S_set):
    """
    Check if p can be added to S_set (set of indices).
    S_set must be a no-isosceles set.
    """
    S_list = list(S_set)
    n = len(S_list)
    if n < 2:
        return True
    
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if BAD_FOR_PAIR[a, b, p]:
                return False
    return True

def _is_maximal(S_set):
    """
    Check if S_set is a maximal no-isosceles set.
    """
    for p in range(N_TOTAL):
        if p in S_set:
            continue
        if _can_add(p, S_set):
            return False
    return True

def _reduce_to_minimal(S_set):
    """
    Try to remove points from S_set while maintaining maximality.
    """
    S_list = list(S_set)
    changed = True
    while changed:
        changed = False
        for i in range(len(S_list)):
            p = S_list[i]
            S_new = S_list[:i] + S_list[i+1:]
            S_new_set = set(S_new)
            
            # Check if S_new is maximal
            if _is_maximal(S_new_set):
                S_list = S_new
                S_set = S_new_set
                changed = True
                break
    return S_set

def _greedy_maximal(rng):
    """
    Construct a maximal no-isosceles set using greedy random selection.
    """
    S_set = set()
    available = list(range(N_TOTAL))
    rng.shuffle(available)
    
    for p in available:
        if _can_add(p, S_set):
            S_set.add(p)
    
    return S_set

def construct():
    start_time = time.time()
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Multiple restarts
    for attempt in range(1000):
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate a maximal set
        S = _greedy_maximal(rng)
        
        # Reduce to minimal maximal set
        S_reduced = _reduce_to_minimal(S)
        
        if len(S_reduced) < best_size:
            best_size = len(S_reduced)
            best_set = S_reduced
            
        if best_size <= 6:
            break
            
    if best_set is None:
        # Fallback: just return a small valid set, even if not minimal?
        # But it must be maximal.
        S = _greedy_maximal(rng)
        best_set = S
        
    return [ALL_POINTS[i] for i in best_set]
