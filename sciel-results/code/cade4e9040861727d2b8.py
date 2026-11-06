"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

# Precompute all 4-subsets of [1,16]
_ALL_SUBSETS = [tuple(c) for c in combinations(range(1, 17), 4)]
_N_SUBSETS = len(_ALL_SUBSETS)  # 1820

# For each pair of subsets, compute their intersection as a frozenset (or None if same)
# We index: for each 4-subset i, for each other 4-subset j, store the intersection frozenset
# But 1820^2 = 3.3M pairs, each a frozenset of up to 4 elements. That's a lot of memory.
# Instead, we can store intersection size and the intersection set compactly.

# Better approach: For each subset, represent it as a bitmask (16 bits).
# Then intersection = bitmask_i & bitmask_j, and we can compare intersections quickly.

def _to_mask(subset):
    m = 0
    for x in subset:
        m |= (1 << (x - 1))
    return m

# Precompute masks for all subsets
_MASKS = [_to_mask(s) for s in _ALL_SUBSETS]

# Precompute pairwise intersection masks: inter[i][j] = _MASKS[i] & _MASKS[j]
# Store as a list of lists of ints (16-bit). 1820*1820 = 3,312,400 ints. ~26 MB. Acceptable.
_INTER = [[0] * _N_SUBSETS for _ in range(_N_SUBSETS)]
for i in range(_N_SUBSETS):
    mi = _MASKS[i]
    row = _INTER[i]
    for j in range(i + 1, _N_SUBSETS):
        v = mi & _MASKS[j]
        row[j] = v
        _INTER[j][i] = v

def _greedy_with_order(order, deadline):
    """Greedy: add subsets in the given order if no conflict with existing pairs."""
    chosen_idx = []
    # conflict_count[c] = number of pairs (a,b) in chosen such that inter[a][b] == inter[a][c] == inter[b][c]
    # Actually, for a new candidate c, we need: for all pairs (a,b) in chosen, 
    #   NOT (inter[a][b] == inter[a][c] == inter[b][c])
    # We can check this by iterating over all pairs in chosen. That's O(m^2) per candidate.
    # To speed up, we can maintain for each intersection value v, the set of pairs that have that intersection.
    # Then for candidate c, we need to check: does there exist a pair (a,b) in chosen such that
    #   inter[a][b] == inter[a][c] == inter[b][c]?
    # Let v_c = inter[a][c] for some a... no, it depends on a.
    # Alternative: for candidate c, the condition is violated iff there exist a,b in chosen
    #   such that inter[a][b] == inter[a][c] == inter[b][c].
    # Let's define for each element x in [1,16], and each pair type... this is complex.
    # For now, use the O(m^2) check but with fast bitmask comparisons.
    
    t_start = time.time()
    for c in order:
        if time.time() - t_start > deadline - 2.0:
            break
        # Check conflict with all pairs in chosen
        ok = True
        m = len(chosen_idx)
        for i in range(m):
            a = chosen_idx[i]
            inter_ac = _INTER[a][c]
            for j in range(i + 1, m):
                b = chosen_idx[j]
                if _INTER[a][b] == inter_ac and _INTER[b][c] == inter_ac:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen_idx.append(c)
    return [list(_ALL_SUBSETS[i]) for i in chosen_idx]

def construct():
    deadline = 80.0  # safety margin before 90s kill
    t0 = time.time()
    
    best = []
    
    # First try natural order
    natural_order = list(range(_N_SUBSETS))
    result = _greedy_with_order(natural_order, deadline)
    if len(result) > len(best):
        best = result
    
    elapsed = time.time() - t0
    if elapsed > deadline - 5.0:
        return best
    
    # Try randomized restarts
    rng = random.Random(42)
    for restart in range(200):
        if time.time() - t0 > deadline - 5.0:
            break
        order = list(range(_N_SUBSETS))
        rng.shuffle(order)
        result = _greedy_with_order(order, deadline)
        if len(result) > len(best):
            best = result
    
    return best
# EVOLVE-BLOCK-END
