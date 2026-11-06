"""
Optimized: precomputed isosceles bitmasks + vectorized maximality + fast reduction.
"""
import time
import random
import numpy as np

N = 16
DEADLINE = 318.0  # safety margin before 330s kill

# Precompute all points
POINTS = [(x, y) for y in range(N) for x in range(N)]
NPTS = N * N
IDX = {p: i for i, p in enumerate(POINTS)}

# Precompute squared distances
# dist[i, j] = squared distance between POINTS[i] and POINTS[j]
_coords = np.array(POINTS, dtype=np.int32)
_diff = _coords[:, np.newaxis, :] - _coords[np.newaxis, :, :]
_dist2 = np.sum(_diff * _diff, axis=2).astype(np.int64)

# Precompute isosceles mask:
# iso[i, j] is a bitmask of points k such that {i, j, k} forms an isosceles triangle.
# We need: for a set S, a point p is blocked if exists a, b in S with p in iso[a, b].
# Equivalently, p is NOT blocked if for all a, b in S, p is NOT in iso[a, b].

# iso_pair[i, j]: set of k (k != i, k != j) such that triangle (i, j, k) is isosceles
# Use a list of bitmasks (Python ints) for fast OR operations
_iso_pair = [[0] * NPTS for _ in range(NPTS)]

for i in range(NPTS):
    for j in range(i + 1, NPTS):
        d_ij = _dist2[i, j]
        mask = 0
        for k in range(NPTS):
            if k == i or k == j:
                continue
            d_ik = _dist2[i, k]
            d_jk = _dist2[j, k]
            if d_ik == d_ij or d_jk == d_ij or d_ik == d_jk:
                mask |= (1 << k)
        _iso_pair[i][j] = mask
        _iso_pair[j][i] = mask

# For a given set S (represented as a bitmask), the set of blocked points is:
# blocked(S) = union over all pairs (a,b) in S of iso_pair[a,b]
# S is maximal iff blocked(S) covers all points not in S.
# i.e., for every p not in S: p is in blocked(S)

def _blocked_mask(S_mask, S_list):
    """Compute the union of iso_pair[a,b] for all pairs in S_list."""
    if len(S_list) < 2:
        return 0
    result = 0
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        row = _iso_pair[a]
        for j in range(i + 1, n):
            b = S_list[j]
            result |= row[b]
    return result

def _is_maximal_fast(S_mask, S_list):
    """Check maximality using precomputed masks."""
    if len(S_list) < 2:
        return False
    blocked = _blocked_mask(S_mask, S_list)
    # All points not in S must be in blocked
    all_mask = (1 << NPTS) - 1
    complement = all_mask & ~S_mask
    return (complement & ~blocked) == 0

def _can_add_fast(p_idx, S_mask, S_list):
    """Check if p can be added to S without creating isosceles."""
    for a in S_list:
        da = _dist2[p_idx, a]
        for b in S_list[S_list.index(a) + 1:]:  # inefficient, use indices
            pass
    # Better: iterate by index
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        da = _dist2[p_idx, a]
        for j in range(i + 1, n):
            b = S_list[j]
            dp_b = _dist2[p_idx, b]
            d_ab = _dist2[a, b]
            if da == dp_b or da == d_ab or dp_b == d_ab:
                return False
    return True

def _reduce_to_minimal(S_list):
    """Iteratively remove points if set remains maximal."""
    S_list = list(S_list)
    changed = True
    while changed:
        changed = False
        for i in range(len(S_list)):
            p = S_list[i]
            S_test = S_list[:i] + S_list[i + 1:]
            if len(S_test) < 2:
                continue
            S_test_mask = 0
            for x in S_test:
                S_test_mask |= (1 << x)
            if _is_maximal_fast(S_test_mask, S_test):
                S_list = S_test
                changed = True
                break
    return S_list

def _greedy_construct(order):
    """Greedy: add points in given order if possible."""
    chosen = []
    for p in order:
        if _can_add_fast(p, 0, chosen):
            chosen.append(p)
    return chosen

def _extend_to_maximal(S_list):
    """Add points to S to make it maximal (if not already)."""
    S_list = list(S_list)
    all_indices = list(range(NPTS))
    S_set = set(S_list)
    for p in all_indices:
        if p in S_set:
            continue
        if _can_add_fast(p, 0, S_list):
            S_list.append(p)
            S_set.add(p)
    return S_list

def _local_search_swap(S_list):
    """Try swapping one point for another to reduce size."""
    S_list = list(S_list)
    n_S = len(S_list)
    all_indices = list(range(NPTS))
    
    for i in range(n_S):
        p_remove = S_list[i]
        S_without = S_list[:i] + S_list[i + 1:]
        S_without_set = set(S_without)
        
        # Try adding each point not in S_without
        for p_add in all_indices:
            if p_add in S_without_set:
                continue
            if not _can_add_fast(p_add, 0, S_without):
                continue
            S_new = S_without + [p_add]
            S_new_mask = 0
            for x in S_new:
                S_new_mask |= (1 << x)
            if _is_maximal_fast(S_new_mask, S_new):
                reduced = _reduce_to_minimal(S_new)
                if len(reduced) < n_S:
                    return reduced
    return None

def construct():
    start_time = time.time()
    best_set = None
    best_size = NPTS + 1
    rng = random.Random(42)
    
    # Strategy 1: Random order greedy + reduction
    for attempt in range(500):
        if time.time() - start_time > DEADLINE:
            break
        order = list(range(NPTS))
        rng.shuffle(order)
        chosen = _greedy_construct(order)
        chosen = _extend_to_maximal(chosen)
        reduced = _reduce_to_minimal(chosen)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
        if best_size <= 8:
            break
    
    # Strategy 2: Local search on best
    if best_set is not None:
        for _ in range(50):
            if time.time() - start_time > DEADLINE:
                break
            improved = _local_search_swap(best_set)
            if improved is not None and len(improved) < best_size:
                best_size = len(improved)
                best_set = improved[:]
            else:
                break
    
    # Strategy 3: More random restarts with different seeds
    for attempt in range(300):
        if time.time() - start_time > DEADLINE:
            break
        seed = rng.randint(0, 10**9)
        rng2 = random.Random(seed)
        order = list(range(NPTS))
        rng2.shuffle(order)
        chosen = _greedy_construct(order)
        chosen = _extend_to_maximal(chosen)
        reduced = _reduce_to_minimal(chosen)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
        if best_size <= 8:
            break
    
    # Strategy 4: Local search again
    if best_set is not None:
        for _ in range(30):
            if time.time() - start_time > DEADLINE:
                break
            improved = _local_search_swap(best_set)
            if improved is not None and len(improved) < best_size:
                best_size = len(improved)
                best_set = improved[:]
            else:
                break
    
    if best_set is None:
        # Fallback
        chosen = []
        for p in range(NPTS):
            if _can_add_fast(p, 0, chosen):
                chosen.append(p)
        best_set = _extend_to_maximal(chosen)
    
    # Convert indices back to (x, y)
    return [POINTS[i] for i in best_set]
