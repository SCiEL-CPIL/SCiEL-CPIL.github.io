# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precompute all 4-subsets of [1,16] as bitmasks
# and build the conflict graph (edge if three sets form a sunflower)

_ALL_SETS = []
_ALL_SETS_IDX = {}
for comb in combinations(range(1, 17), 4):
    mask = 0
    for x in comb:
        mask |= (1 << (x - 1))
    _ALL_SETS.append(mask)
    _ALL_SETS_IDX[mask] = len(_ALL_SETS) - 1

N_SETS = len(_ALL_SETS)  # 1820

# Build adjacency: conflict[i] is a bitmask (as a Python int or set) of j such that
# some pair (k, l) with i, k, l already chosen would form a sunflower.
# Actually, the sunflower condition: three distinct sets A, B, C form a sunflower if
# A∩B = A∩C = B∩C (all three pairwise intersections equal).
# We need: for any three distinct sets, their three pairwise intersections are not all equal.

# Precompute for each pair (i, j), the intersection mask.
# Then three sets i, j, k form a sunflower iff inter(i,j) == inter(i,k) == inter(j,k).

# For efficiency, we'll precompute a "conflict" structure:
# For a given set i and a candidate set j, we need to know which existing sets k in our family
# would conflict with the pair (i, j). Specifically, k conflicts with (i,j) if
# inter(i,k) == inter(j,k) == inter(i,j).

# Let's precompute inter[i][j] for all pairs.
# inter[i] is a list of length N_SETS with inter[i][j] = _ALL_SETS[i] & _ALL_SETS[j]

inter = [[0] * N_SETS for _ in range(N_SETS)]
for i in range(N_SETS):
    ai = _ALL_SETS[i]
    for j in range(i + 1, N_SETS):
        ij = ai & _ALL_SETS[j]
        inter[i][j] = ij
        inter[j][i] = ij

# For fast conflict checking during greedy:
# When we have a family F (as a set of indices), and we want to add set j,
# we check: for every pair (i1, i2) in F, does (i1, i2, j) form a sunflower?
# That is: inter[i1][i2] == inter[i1][j] == inter[i2][j]?
#
# A faster approach: for each existing set i in F, compute c_i = inter[i][j].
# Then for each pair (i1, i2) in F, check if c_i1 == c_i2 == inter[i1][i2].
#
# Even faster: group existing sets by their intersection with j.
# For each group G of sets with the same intersection value v = inter[i][j],
# we need to check if any pair (i1, i2) in G has inter[i1][i2] == v.
# Because if inter[i1][j] = inter[i2][j] = v and inter[i1][i2] = v, then all three
# pairwise intersections are v, forming a sunflower.

# So the check is: for candidate j, group F by inter[i][j]. For each group, check if
# any pair in the group has mutual intersection equal to the group's value.

# To make this fast, for each group value v, we need to check pairs.
# We can precompute for each possible intersection value v (which is a 4-bit or fewer mask),
# a set of pairs that have that intersection. But that's complex.

# Alternative: since |F| is at most ~250, and N_SETS=1820, a direct check per candidate
# is O(|F|^2) per candidate, times 1820 candidates = ~1820 * 250^2 = 114M ops per full pass.
# That's too slow for multiple restarts within 90s.

# Better: use the grouping approach. For candidate j:
#   1. For each i in F, compute v_i = inter[i][j]
#   2. Group indices by v_i
#   3. For each group with value v, check if any pair (a,b) in group has inter[a][b] == v
#      We can do this by: for each group, take the list of indices, and for each pair check.
#      But we can optimize: if the group has size > 1, we check pairs.

# Actually, let's think differently. Precompute for each pair (i, j) the intersection.
# For a fixed j, and a set F, we want to know if there exist i1, i2 in F such that
# inter[i1][i2] == inter[i1][j] == inter[i2][j].

# Let's define for each j, and for each possible "core" value c (a bitmask of up to 4 bits),
# the set of i such that inter[i][j] == c. Then for a candidate j, we look at F ∩ S_c for each c,
# and check if any pair in F ∩ S_c has mutual intersection c.

# Precompute: for each j (0..N_SETS-1) and each possible core c, the list of i with inter[i][j] == c.
# But c ranges over all possible 4-bit intersections, which are subsets of the 4 elements of set j.
# Actually, inter[i][j] is a subset of j (since it's the intersection). So c ⊆ j.
# There are at most 2^4 = 16 possible values for c for a given j.

# Let's precompute:
# core_groups[j][c] = list of i such that inter[i][j] == c
# And also: for each c and each j, which pairs (i1, i2) in core_groups[j][c] have inter[i1][i2] == c?
# That's too much.

# Let's just do the greedy with a reasonably efficient check and use multiple restarts.

def _check_conflict(family_indices, candidate_idx):
    """Check if adding candidate_idx to family_indices creates a sunflower.
    family_indices: list of indices already in family.
    Returns True if conflict exists."""
    j = candidate_idx
    # Group by intersection with j
    groups = {}
    for i in family_indices:
        v = inter[i][j]
        if v not in groups:
            groups[v] = []
        groups[v].append(i)
    
    for v, members in groups.items():
        if len(members) < 2:
            continue
        # Check all pairs in members
        mlen = len(members)
        for a_idx in range(mlen):
            ia = members[a_idx]
            for b_idx in range(a_idx + 1, mlen):
                ib = members[b_idx]
                if inter[ia][ib] == v:
                    return True
    return False


def _greedy_randomized(seed):
    """Greedy with randomized ordering."""
    rng = random.Random(seed)
    order = list(range(N_SETS))
    rng.shuffle(order)
    
    family = []
    family_set = set()
    
    for idx in order:
        if _check_conflict(family, idx):
            continue
        family.append(idx)
        family_set.add(idx)
    
    return family


def _local_improve(family, seed, time_limit):
    """Try to add more sets by re-ordering or swapping. Simple: try adding any set not in family."""
    rng = random.Random(seed)
    family_set = set(family)
    not_in_family = [i for i in range(N_SETS) if i not in family_set]
    rng.shuffle(not_in_family)
    
    improved = True
    while improved:
        improved = False
        for idx in not_in_family:
            if _check_conflict(family, idx):
                continue
            family.append(idx)
            family_set.add(idx)
            not_in_family.remove(idx)
            improved = True
            if time.time() > time_limit:
                return family
    return family


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    best_family = []
    
    # Try multiple random restarts
    seed = 42
    while time.time() < deadline:
        family = _greedy_randomized(seed)
        # Local improvement: try to add more
        family = _local_improve(family, seed, deadline)
        
        if len(family) > len(best_family):
            best_family = family[:]
        
        seed += 1
    
    # Convert to sorted lists
    result = [sorted([(_ALL_SETS[i] >> b) & 1 and (b + 1) for b in range(16) if (_ALL_SETS[i] >> b) & 1]) for i in best_family]
    
    return result

# EVOLVE-BLOCK-END
