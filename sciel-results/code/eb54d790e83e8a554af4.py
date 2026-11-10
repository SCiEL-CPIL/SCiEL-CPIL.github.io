"""Minimum maximal no-isosceles set on [0, N)^2 grid."""
N = 16
import time
import numpy as np
import random
from itertools import combinations

# Precompute all grid points and conflict structure
# Point i and j "conflict" if adding a third point k would form an isosceles triangle
# But actually: a set is "no-isosceles" if NO three points in it form an isosceles triangle.
# So we need: for every triple (a,b,c) in the set, it is NOT isosceles.
# Two points a,b don't directly conflict; the conflict is on triples.
# A set S is no-isosceles iff for all distinct a,b,c in S, the triangle abc is not isosceles.
#
# Key insight: if we have a set S and want to add point p, we need that for all pairs (a,b) in S,
# the triangle (p,a,b) is not isosceles. So p conflicts with pair (a,b) if p,a,b form isosceles.
#
# For maximal: no point can be added without creating an isosceles triangle.

def _dist2(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

# Precompute: for each point p and each pair (a,b), whether p,a,b is isosceles
# Actually, let's think differently.
# For a candidate point p and existing set S: p can be added iff for all pairs (a,b) in S,
# (p,a,b) is NOT isosceles.
#
# Let's precompute for each point p, the set of "forbidden pairs" - pairs (a,b) such that
# (p,a,b) is isosceles. Then p can be added to S iff no pair in S is in p's forbidden set.

# Total points: N*N = 256
# Number of pairs: 256*255/2 = 32640
# For each point p (256), forbidden pairs could be many.
# This is a 3-uniform hypergraph independent set problem.

# Let's use a different approach: represent each point by its index 0..255
# Precompute isosceles triples as a set of frozensets or use bitsets.

# For efficiency, precompute for each pair (i,j), the set of points k such that (i,j,k) is isosceles.
# Then: a set S is no-isosceles iff for all pairs (i,j) in S, no other point k in S is in iso_set[(i,j)].
# Equivalently: for all i,j,k in S with i!=j!=k, (i,j,k) is not isosceles.

# Let's precompute iso_sets: dict from (i,j) -> set of k
# But 256*256 = 65536 pairs, each with potentially many k's.

# Better: for each point k, precompute the set of pairs (i,j) such that (i,j,k) is isosceles.
# Then k can be added to S iff no pair in S is in forbidden_pairs[k].

# Let's use bitsets for pairs. There are 32640 pairs. Use a 32640-bit integer or numpy array.
# Actually, let's use Python ints as bitsets for the 32640 pairs.

def _build_conflict_structure():
    """Precompute for each point, the bitset of pairs that conflict with it."""
    n_pts = N * N
    # Map (x,y) to index
    pts = [(x, y) for y in range(N) for x in range(N)]
    # Pair index: for i < j, pair_id = i*(n_pts-1) - i*(i+1)//2 ... let's just use a dict or formula
    # Simpler: pair_id(i,j) for i<j: i*(n_pts-1) - i*(i+1)//2 ... 
    # Actually let's just enumerate pairs and assign IDs
    pair_ids = {}
    pair_list = []
    pid = 0
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            pair_ids[(i, j)] = pid
            pair_list.append((i, j))
            pid += 1
    n_pairs = pid
    
    # For each point k, compute which pairs (i,j) form an isosceles triangle with k
    # i.e., (i,j,k) is isosceles
    forbidden = [0] * n_pts  # bitset as Python int
    for k in range(n_pts):
        kx, ky = pts[k]
        for (i, j), pid in pair_ids.items():
            if i == k or j == k:
                continue
            ix, iy = pts[i]
            jx, jy = pts[j]
            d_ik = (ix-kx)**2 + (iy-ky)**2
            d_jk = (jx-kx)**2 + (jy-ky)**2
            d_ij = (ix-jx)**2 + (iy-jy)**2
            if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                forbidden[k] |= (1 << pid)
    
    return pts, pair_ids, pair_list, forbidden, n_pairs

# Precompute once at module level
_pts, _pair_ids, _pair_list, _forbidden, _n_pairs = _build_conflict_structure()
_n_pts = N * N

def _can_add(point_idx, chosen_pairs_bitset):
    """Check if point can be added given current set (represented as pair bitset)."""
    # A pair (i,j) is in the chosen set if both i and j are chosen.
    # We need: no pair in chosen_pairs_bitset is in forbidden[point_idx]
    return (chosen_pairs_bitset & _forbidden[point_idx]) == 0

def _pairs_of_set(chosen):
    """Compute the pair bitset for a set of point indices."""
    bitset = 0
    for i in range(len(chosen)):
        for j in range(i+1, len(chosen)):
            bitset |= (1 << _pair_ids[(chosen[i], chosen[j])])
    return bitset

def _greedy_maximal(order):
    """Greedy maximal construction following a given point order."""
    chosen = []
    chosen_set = set()
    pair_bitset = 0
    for p in order:
        if _can_add(p, pair_bitset):
            # Add p: update pair_bitset with all pairs (p, q) for q in chosen
            for q in chosen:
                pair_bitset |= (1 << _pair_ids[(min(p,q), max(p,q))])
            chosen.append(p)
            chosen_set.add(p)
    return chosen

def _is_maximal(chosen):
    """Check if the set is maximal (no point can be added)."""
    if not chosen:
        return _n_pts == 0
    pair_bitset = _pairs_of_set(chosen)
    for p in range(_n_pts):
        if p not in set(chosen) and _can_add(p, pair_bitset):
            return False
    return True

def _local_search(chosen, max_iters=5000, seed=42):
    """Try to reduce the set size via swaps while maintaining no-isosceles and maximal."""
    chosen = list(chosen)
    best = len(chosen)
    rng = random.Random(seed)
    
    for it in range(max_iters):
        if len(chosen) <= best:
            best = len(chosen)
        
        # Try to remove a point and see if we can still maintain maximality
        # Actually, removing a point may break maximality. We need to be careful.
        # Better: try swap - remove one, add another, check no-isosceles.
        
        if len(chosen) < 2:
            break
            
        i = rng.randrange(len(chosen))
        removed = chosen[i]
        new_chosen = chosen[:i] + chosen[i+1:]
        new_pair_bitset = _pairs_of_set(new_chosen)
        
        # Try to find a point to add
        candidates = [p for p in range(_n_pts) if p not in set(new_chosen) and _can_add(p, new_pair_bitset)]
        if candidates:
            # Prefer candidates that make the set maximal
            rng.shuffle(candidates)
            for cand in candidates:
                test_chosen = new_chosen + [cand]
                test_bitset = new_pair_bitset | sum(1 << _pair_ids[(min(cand,q), max(cand,q))] for q in new_chosen)
                # Check no-isosceles is maintained (should be by construction)
                # Check maximality
                if _is_maximal(test_chosen):
                    chosen = test_chosen
                    break
        # If we didn't find a good swap, try just removing and checking if still maximal
        # (unlikely but possible if the set was not truly minimal)
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Start with a greedy solution using different orderings
    best = None
    best_size = _n_pts + 1
    
    # Try multiple random orderings
    rng = random.Random(12345)
    n_attempts = 200
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        # Create a random order with some structure
        order = list(range(_n_pts))
        rng.shuffle(order)
        
        chosen = _greedy_maximal(order)
        
        # Verify no-isosceles
        chosen_set = set(chosen)
        valid = True
        for a, b, c in combinations(chosen, 3):
            if _is_isosceles(_pts[a], _pts[b], _pts[c]):
                valid = False
                break
        if not valid:
            continue
        
        # Check maximality
        if not _is_maximal(chosen):
            # Try to make it maximal by adding points
            pair_bitset = _pairs_of_set(chosen)
            for p in range(_n_pts):
                if p not in chosen_set and _can_add(p, pair_bitset):
                    for q in chosen:
                        pair_bitset |= (1 << _pair_ids[(min(p,q), max(p,q))])
                    chosen.append(p)
                    chosen_set.add(p)
            if not _is_maximal(chosen):
                continue
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = list(chosen)
        
        # Try local search on this solution
        if time.time() > deadline:
            break
        improved = _local_search(chosen, max_iters=2000, seed=attempt)
        if len(improved) < best_size:
            best_size = len(improved)
            best = improved
    
    if best is None:
        # Fallback: simple greedy
        best = _greedy_maximal(list(range(_n_pts)))
    
    return [_pts[i] for i in best]

# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Size: {len(result)}, Time: {t1-t0:.1f}s")
    # Verify
    pts = result
    valid = True
    for a, b, c in combinations(pts, 3):
        if _is_isosceles(a, b, c):
            valid = False
            break
    print(f"Valid no-isosceles: {valid}")
    # Check maximal
    chosen_set = set(pts)
    pair_bitset = _pairs_of_set([_pts.index(p) for p in pts]) if pts else 0
    # Re-index
    idx_map = {p: i for i, p in enumerate(_pts)}
    chosen_idx = [idx_map[p] for p in pts]
    pb = _pairs_of_set(chosen_idx)
    max_ok = True
    for p in range(_n_pts):
        if _pts[p] not in chosen_set and _can_add(p, pb):
            max_ok = False
            break
    print(f"Maximal: {max_ok}")
