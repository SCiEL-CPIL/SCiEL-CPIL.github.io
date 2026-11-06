import random
import time
from itertools import combinations

# Precompute all 4-subsets of [1, 16] as bitmasks
# 16 elements -> 16 bits
# Element i (0-indexed) corresponds to bit 1 << i
# Set {1, 2, 3, 4} -> bits 0, 1, 2, 3

def generate_subsets():
    subsets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        subsets.append(mask)
    return subsets

SUBSETS = generate_subsets()
N_SUBSETS = len(SUBSETS)

# Precompute pairwise intersections
# inter[i][j] = SUBSETS[i] & SUBSETS[j]
# We can store this as a list of lists or a flat array.
# List of lists is easier to access.
INTERSECTIONS = [[0] * N_SUBSETS for _ in range(N_SUBSETS)]
for i in range(N_SUBSETS):
    si = SUBSETS[i]
    for j in range(i + 1, N_SUBSETS):
        sj = SUBSETS[j]
        inter = si & sj
        INTERSECTIONS[i][j] = inter
        INTERSECTIONS[j][i] = inter

def is_valid_addition(new_idx, chosen_indices):
    """
    Check if adding new_idx to chosen_indices creates a sunflower.
    A sunflower exists if there are u, v in chosen_indices such that:
    inter[u][v] == inter[u][new_idx] == inter[v][new_idx]
    """
    new_mask = SUBSETS[new_idx]
    # We only need to check pairs in chosen_indices
    # Optimization: If we maintain the chosen set, we can just iterate.
    n = len(chosen_indices)
    for i in range(n):
        u = chosen_indices[i]
        inter_un = INTERSECTIONS[u][new_idx]
        for j in range(i + 1, n):
            v = chosen_indices[j]
            # If inter_un == inter_uv and inter_un == inter_vn
            if inter_un == INTERSECTIONS[u][v] and inter_un == INTERSECTIONS[v][new_idx]:
                return False
    return True

def greedy_randomized(restart_id, deadline):
    """
    Perform one randomized greedy pass.
    """
    chosen_indices = []
    candidates = list(range(N_SUBSETS))
    random.shuffle(candidates)
    
    for idx in candidates:
        if time.time() > deadline:
            break
        if is_valid_addition(idx, chosen_indices):
            chosen_indices.append(idx)
            
    return chosen_indices

def construct():
    deadline = time.time() + 80.0  # 80s limit to be safe
    best_chosen = []
    best_len = 0
    
    # Run multiple restarts
    # Estimate cost: ~0.1-0.5s per restart depending on size.
    # Let's run as many as possible.
    restart_count = 0
    while time.time() < deadline:
        chosen = greedy_randomized(restart_count, deadline)
        if len(chosen) > best_len:
            best_len = len(chosen)
            best_chosen = chosen
        restart_count += 1
        
    # Convert bitmasks back to sorted lists of 1..16
    result = []
    for idx in best_chosen:
        mask = SUBSETS[idx]
        subset = []
        for i in range(16):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result
