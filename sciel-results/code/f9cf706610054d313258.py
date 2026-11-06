"""Optimized construction for large 4-uniform families without three sunflowers."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 16
K = 4
MAX_SIZE = 250
DEADLINE_MARGIN = 5.0  # seconds before 90s wall-clock kill

# Precompute all 4-element subsets as bitmasks
ALL_SETS = []
ALL_SETS_AS_TUPLES = []
for combo in combinations(range(N), K):
    mask = 0
    for x in combo:
        mask |= (1 << x)
    ALL_SETS.append(mask)
    ALL_SETS_AS_TUPLES.append(combo)

NUM_SETS = len(ALL_SETS)  # 1820

# Precompute intersection sizes between all pairs
# inter[i][j] = popcount(ALL_SETS[i] & ALL_SETS[j])
# We need this to check sunflower conditions efficiently
# Sunflower: three sets A, B, C where |A∩B| = |A∩C| = |B∩C|
# For 4-sets, intersection sizes are 0, 1, 2, 3 (4 means identical, excluded)

# Build adjacency: for each pair (i,j), store the intersection size
# Memory: 1820*1820 ~ 3.3M entries, fine as a 2D array of bytes
inter = [[0] * NUM_SETS for _ in range(NUM_SETS)]
for i in range(NUM_SETS):
    mi = ALL_SETS[i]
    for j in range(i + 1, NUM_SETS):
        c = bin(mi & ALL_SETS[j]).count('1')
        inter[i][j] = c
        inter[j][i] = c

# For sunflower detection: three sets i, j, k form a sunflower iff
# inter[i][j] == inter[i][k] == inter[j][k]
# We need to avoid adding a set that would create such a triple.

# Strategy: maintain a conflict graph. When we add set k, we check all pairs
# (i, j) already in the family to see if adding k creates a sunflower.
# A sunflower with k means: inter[i][j] == inter[i][k] == inter[j][k]
# So for each pair (i,j) in family, if inter[i][j] == inter[i][k] == inter[j][k], conflict.

# To speed up: for each new candidate k, group existing family members by their
# intersection size with k. Then for each group of size >= 2, check if any pair
# in that group has the same intersection size between them.

def _greedy_with_order(order, deadline):
    """Greedy: process candidates in given order, add if no sunflower conflict."""
    family = []
    family_idx = set()
    
    for idx in order:
        if time.time() > deadline:
            break
        if idx in family_idx:
            continue
        
        # Check if adding idx creates a sunflower with any pair in family
        # Group family members by inter[member][idx]
        groups = {}
        for j in family:
            c = inter[j][idx]
            if c not in groups:
                groups[c] = []
            groups[c].append(j)
        
        conflict = False
        for c, members in groups.items():
            if len(members) < 2:
                continue
            # Check all pairs in this group
            for a_pos in range(len(members)):
                a = members[a_pos]
                for b_pos in range(a_pos + 1, len(members)):
                    b = members[b_pos]
                    if inter[a][b] == c:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                break
        
        if not conflict:
            family.append(idx)
            family_idx.add(idx)
    
    return family

def _local_search(family, deadline, max_swaps=500):
    """Try to improve family by removing one set and adding another."""
    family_set = set(family)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each set and see if we can add a better one
        for remove_idx in list(family):
            if time.time() > deadline:
                break
            candidate_family = [x for x in family if x != remove_idx]
            candidate_set = set(candidate_family)
            
            # Find best addition
            best_add = -1
            best_score = 0
            for add_idx in range(NUM_SETS):
                if add_idx in candidate_set:
                    continue
                # Check if adding add_idx is valid
                groups = {}
                for j in candidate_family:
                    c = inter[j][add_idx]
                    if c not in groups:
                        groups[c] = []
                    groups[c].append(j)
                
                conflict = False
                for c, members in groups.items():
                    if len(members) < 2:
                        continue
                    for a_pos in range(len(members)):
                        a = members[a_pos]
                        for b_pos in range(a_pos + 1, len(members)):
                            b = members[b_pos]
                            if inter[a][b] == c:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        break
                
                if not conflict:
                    # Count how many sets we could potentially add after this
                    # Simple heuristic: just accept any valid addition
                    best_add = add_idx
                    break  # Take first valid to save time
            
            if best_add >= 0:
                family = [x for x in family if x != remove_idx]
                family.append(best_add)
                improved = True
                break
    
    return family

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN
    
    best_family = []
    
    # Multiple randomized greedy restarts
    rng = random.Random(42)
    
    # First, try a deterministic order (lexicographic)
    order = list(range(NUM_SETS))
    family = _greedy_with_order(order, deadline)
    if len(family) > len(best_family):
        best_family = family[:]
    
    # Then randomized orders
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        # Random shuffle
        order = list(range(NUM_SETS))
        rng.shuffle(order)
        
        family = _greedy_with_order(order, deadline)
        if len(family) > len(best_family):
            best_family = family[:]
        
        # Local search on the best family found so far
        if time.time() < deadline - 2.0:
            improved = _local_search(best_family[:], deadline - 1.0, max_swaps=200)
            if len(improved) > len(best_family):
                best_family = improved
    
    # Convert to list of sorted tuples
    result = [ALL_SETS_AS_TUPLES[i] for i in best_family]
    return result

# EVOLVE-BLOCK-END
