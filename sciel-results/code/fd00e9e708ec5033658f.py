"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations
from functools import lru_cache

# EVOLVE-BLOCK-START
N = 12
M = 40
TOTAL_SUBSETS = 1 << N  # 4096

# Precompute comparability: for each mask, list of masks it is a strict subset of
# and list of masks that are strict subsets of it.
# We will build these lazily or precompute key structures.

# Precompute popcount for all masks
def _popcount(x):
    return bin(x).count('1')

# Precompute for each mask, the list of all masks that are strict subsets
# This is expensive to precompute fully for 4096 masks, but we can do it.
# Actually, let's precompute a boolean matrix or adjacency lists.
# For 4096 masks, a full adjacency list is feasible.

# Precompute: subset_of[mask] = list of masks that are strict subsets of mask
# superset_of[mask] = list of masks that are strict supersets of mask
# But storing all pairs is 4096^2 ~ 16M, which is fine.

# Let's precompute comparability pairs.
# pair_comparable[i][j] = True if i < j (subset) or j < i (subset)

# Actually, for the greedy algorithm, we need to efficiently compute
# the number of comparable pairs if we add a new set to a collection.

# Let's precompute for each mask m:
#   below[m] = list of masks that are strict subsets of m
#   above[m] = list of masks that are strict supersets of m

# This takes O(4096 * 4096) = ~16M operations, which is fine.

def _precompute_adjacency():
    below = [[] for _ in range(TOTAL_SUBSETS)]
    above = [[] for _ in range(TOTAL_SUBSETS)]
    for m in range(TOTAL_SUBSETS):
        for s in range(TOTAL_SUBSETS):
            if s == m:
                continue
            if (s & m) == s:  # s is subset of m
                if s != m:
                    below[m].append(s)
                    above[s].append(m)
    return below, above

_below, _above = _precompute_adjacency()

def _count_comparable_pairs(selected):
    """Count comparable pairs in a set of masks (as a frozenset or list)."""
    sel = list(selected)
    count = 0
    n = len(sel)
    for i in range(n):
        for j in range(i+1, n):
            a, b = sel[i], sel[j]
            if (a & b) == a or (a & b) == b:
                count += 1
    return count

def _greedy_build(selected, available, budget):
    """
    Greedily add sets to maximize comparable pairs.
    selected: set of masks already selected
    available: set of masks still available
    Returns updated selected set.
    """
    sel = set(selected)
    avail = set(available)
    while len(sel) < M and avail:
        best_mask = None
        best_gain = -1
        for m in avail:
            # Gain = number of new comparable pairs if we add m
            # = |{s in sel : s < m or m < s}|
            gain = 0
            for s in sel:
                if (s & m) == s or (s & m) == m:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = m
        if best_mask is None:
            break
        sel.add(best_mask)
        avail.discard(best_mask)
    return sel

def _local_search(selected, iterations=1000, deadline=None):
    """
    Local search: try swapping one selected set with one unselected set.
    """
    sel = set(selected)
    all_masks = set(range(TOTAL_SUBSETS))
    unselected = all_masks - sel
    
    best_score = _count_comparable_pairs(sel)
    current_score = best_score
    
    for it in range(iterations):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try removing each selected element
        for s in list(sel):
            if deadline is not None and time.time() > deadline:
                break
            sel_new = sel - {s}
            # Find best addition from unselected
            best_add = None
            best_gain = -1
            for u in unselected:
                # Gain of adding u to sel_new
                gain = 0
                for t in sel_new:
                    if (t & u) == t or (t & u) == u:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_add = u
            if best_add is not None:
                score_new = current_score - _gain_of(s, sel) + best_gain
                if score_new > current_score:
                    sel = sel_new | {best_add}
                    unselected = (unselected - {best_add}) | {s}
                    current_score = score_new
                    best_score = max(best_score, current_score)
                    improved = True
                    break
        if not improved:
            break
    return sel, best_score

def _gain_of(m, sel):
    """Number of comparable pairs involving m in sel."""
    gain = 0
    for s in sel:
        if s == m:
            continue
        if (s & m) == s or (s & m) == m:
            gain += 1
    return gain

def _random_restart_greedy(deadline, rng):
    """
    Run a greedy construction from a random starting point, then local search.
    """
    # Start with a random subset of size ~M/2 to diversify
    start_size = rng.randint(M//2, M)
    start = set(rng.sample(range(TOTAL_SUBSETS), start_size))
    
    # Greedy add remaining
    sel = _greedy_build(start, set(range(TOTAL_SUBSETS)) - start, M)
    
    # Local search
    sel, score = _local_search(sel, iterations=500, deadline=deadline)
    return sel, score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s margin
    
    best_sel = None
    best_score = 0
    
    rng = random.Random(42)
    
    # Try multiple random restarts
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        sel, score = _random_restart_greedy(deadline, rng)
        if score > best_score:
            best_score = score
            best_sel = sel
        # If we've used most of the time, stop
        if time.time() > deadline - 5:
            break
    
    if best_sel is None:
        # Fallback: just take first M masks
        best_sel = set(range(M))
    
    # Convert to list of lists
    result = []
    for m in best_sel:
        result.append([b for b in range(N) if (m >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
