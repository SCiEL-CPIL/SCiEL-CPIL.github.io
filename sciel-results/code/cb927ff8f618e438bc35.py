"""Optimized greedy set-packing for sunflower-free 4-uniform families on [16]."""
import time
import random
from itertools import combinations

# Precompute all 4-subsets of [1,16] as bitmasks
# There are C(16,4) = 1820 such subsets
_ALL_SUBSETS = []
_ALL_SUBSETS_SETS = []
for _combo in combinations(range(1, 17), 4):
    _mask = 0
    for _x in _combo:
        _mask |= (1 << _x)
    _ALL_SUBSETS.append(_mask)
    _ALL_SUBSETS_SETS.append(set(_combo))

N_SUBSETS = len(_ALL_SUBSETS)  # 1820

# Precompute pairwise intersections
# intersection[i][j] = _ALL_SUBSETS[i] & _ALL_SUBSETS[j]
_INTERSECTIONS = [[0] * N_SUBSETS for _ in range(N_SUBSETS)]
for _i in range(N_SUBSETS):
    _INTERSECTIONS[_i][_i] = _ALL_SUBSETS[_i]
    for _j in range(_i + 1, N_SUBSETS):
        _INTERSECTIONS[_i][_j] = _ALL_SUBSETS[_i] & _ALL_SUBSETS[_j]
        _INTERSECTIONS[_j][_i] = _INTERSECTIONS[_i][_j]

# Precompute which pairs of subsets form a "partial sunflower core"
# Two subsets A, B form a sunflower with a third C if:
#   A ∩ B = A ∩ C = B ∩ C
# So for a given pair (i, j), the "core" is c = A_i ∩ A_j.
# A third subset k conflicts if A_i ∩ A_k = c AND A_j ∩ A_k = c.
# We precompute for each pair (i,j) the set of k that would complete a sunflower.

# Actually, let's precompute conflict pairs more efficiently.
# For a family F, we need: no three distinct sets i, j, k in F have
#   A_i ∩ A_j = A_i ∩ A_k = A_j ∩ A_k

# Precompute for each pair (i, j), the list of k such that (i, j, k) is a sunflower triple.
# This is O(N^3) = 1820^3 ≈ 6 billion — too slow.

# Better approach: for each pair (i,j), compute core = A_i ∩ A_j.
# Then k conflicts iff (A_i ∩ A_k) == core and (A_j ∩ A_k) == core.
# We can precompute for each pair (i,j) the bitmask of conflicting k's.

# But that's N^2 = 3.3 million pairs, each requiring a scan of N=1820 → 6 billion ops. Too slow.

# Alternative: during greedy, when we add a set, we check conflicts with all pairs in the current family.
# Family size is at most ~250, so checking pairs is O(|F|^2) per addition, and we add at most 1820 sets.
# Total: 1820 * 250^2 ≈ 114 million — feasible in Python if optimized.

# Let's use a different strategy: greedy with random restarts.
# For each candidate set, compute its "conflict count" with the current family:
#   A new set k conflicts with the family if there exist i, j in F such that
#   A_i ∩ A_j = A_i ∩ A_k = A_j ∩ A_k.
# We can compute this by, for each pair (i,j) in F, checking if k would complete a sunflower.

# Precompute: for each pair (i, j), the core c = A_i ∩ A_j.
# For a candidate k, it conflicts with pair (i,j) iff:
#   (A_i ∩ A_k) == c and (A_j ∩ A_k) == c

# Let's precompute for each i and each possible core value, which k's have A_i ∩ A_k == core.
# Core values are 4-bit subsets of [16], so there are 2^16 = 65536 possible cores, but only
# those that are subsets of A_i matter.

# Actually, let's just do the greedy with efficient checking.

def _conflicts_with_family(k_mask, family_masks):
    """Check if adding k_mask to family_masks creates a sunflower triple."""
    n = len(family_masks)
    for i in range(n):
        ai = family_masks[i]
        for j in range(i + 1, n):
            aj = family_masks[j]
            core = ai & aj
            # Check if k completes the sunflower
            if (ai & k_mask) == core and (aj & k_mask) == core:
                return True
    return False


def _greedy_single_pass(order):
    """Greedy: add sets in the given order if they don't conflict."""
    family = []
    for idx in order:
        mask = _ALL_SUBSETS[idx]
        if not _conflicts_with_family(mask, family):
            family.append(mask)
    return family


def _greedy_min_degree():
    """Greedy: at each step, pick the set with minimum conflict count."""
    family = []
    available = set(range(N_SUBSETS))
    while available:
        best_idx = None
        best_count = float('inf')
        for idx in available:
            mask = _ALL_SUBSETS[idx]
            count = 0
            n = len(family)
            for i in range(n):
                ai = family[i]
                for j in range(i + 1, n):
                    aj = family[j]
                    core = ai & aj
                    if (ai & mask) == core and (aj & mask) == core:
                        count += 1
                        break
                if count >= best_count:
                    break
            if count < best_count:
                best_count = count
                best_idx = idx
                if best_count == 0:
                    break
        if best_idx is None:
            break
        family.append(_ALL_SUBSETS[best_idx])
        available.discard(best_idx)
    return family


def _random_order(seed):
    """Generate a random permutation of all subset indices."""
    rng = random.Random(seed)
    order = list(range(N_SUBSETS))
    rng.shuffle(order)
    return order


def construct():
    deadline = time.time() + 80.0  # Safety margin before 90s kill
    best_family = []
    
    # First, try a deterministic greedy min-degree pass
    family = _greedy_min_degree()
    if len(family) > len(best_family):
        best_family = family
    
    # Then try multiple randomized greedy passes
    for seed in range(200):
        if time.time() > deadline - 5.0:
            break
        order = _random_order(seed)
        family = _greedy_single_pass(order)
        if len(family) > len(best_family):
            best_family = family
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        s = []
        for x in range(1, 17):
            if mask & (1 << x):
                s.append(x)
        result.append(s)
    
    return result
