"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random
import numpy as np

N = 16
TOTAL_POINTS = N * N
ALL_POINTS = list(itertools.product(range(N), repeat=2))
PT_TO_IDX = {p: i for i, p in enumerate(ALL_POINTS)}
IDX_TO_PT = {i: p for i, p in enumerate(ALL_POINTS)}

# Precompute isosceles conflict masks
# iso_mask[i][j] is a bitmask of all k such that {i, j, k} forms an isosceles triangle
# Note: i and j are not included in the mask, but we don't add them to blocked anyway.
iso_mask = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]

# Optimization: Precompute squared distances
coords = np.array(ALL_POINTS)
# dist2[i][j]
# We can compute this on the fly for precomputation, it's 256^2 * 256 ops which is ~16M, fine.

def _d2_idx(i, j):
    dx = coords[i][0] - coords[j][0]
    dy = coords[i][1] - coords[j][1]
    return dx*dx + dy*dy

# Precompute iso_mask
for i in range(TOTAL_POINTS):
    for j in range(i + 1, TOTAL_POINTS):
        d_ij = _d2_idx(i, j)
        mask = 0
        for k in range(TOTAL_POINTS):
            if k == i or k == j:
                continue
            d_ik = _d2_idx(i, k)
            d_jk = _d2_idx(j, k)
            if d_ik == d_ij or d_jk == d_ij or d_ik == d_jk:
                mask |= (1 << k)
        iso_mask[i][j] = mask
        iso_mask[j][i] = mask

def _compute_blocked_mask(chosen_idx_set):
    """Compute the blocked mask for a set of chosen indices."""
    blocked = 0
    chosen_list = list(chosen_idx_set)
    n = len(chosen_list)
    for i in range(n):
        idx_i = chosen_list[i]
        for j in range(i + 1, n):
            idx_j = chosen_list[j]
            blocked |= iso_mask[idx_i][idx_j]
    return blocked

def _can_add_idx(p_idx, chosen_idx_set, blocked_mask):
    """Check if point p_idx can be added to chosen_idx_set."""
    if p_idx in chosen_idx_set:
        return False
    if (blocked_mask >> p_idx) & 1:
        return False
    return True

def _greedy(order_idx, deadline=None):
    """Greedy construction using a given ordering of point indices."""
    chosen_idx_set = set()
    blocked_mask = 0
    for p_idx in order_idx:
        if _can_add_idx(p_idx, chosen_idx_set, blocked_mask):
            chosen_idx_set.add(p_idx)
            # Update blocked mask
            # New point p_idx conflicts with all existing points i
            for i in chosen_idx_set:
                if i != p_idx:
                    blocked_mask |= iso_mask[p_idx][i]
    return chosen_idx_set

def _is_maximal(chosen_idx_set, blocked_mask):
    """Check if no further grid point can be added to chosen."""
    # A point p can be added if p not in chosen and p not in blocked
    # We need to check if there exists ANY such p.
    # all_points_mask = (1 << TOTAL_POINTS) - 1
    # available = all_points_mask & ~chosen_mask & ~blocked_mask
    # But chosen_idx_set is a set of ints, not a mask.
    # Let's convert chosen to mask or just iterate.
    # Iterating 256 points is fast.
    for p_idx in range(TOTAL_POINTS):
        if p_idx not in chosen_idx_set:
            if not ((blocked_mask >> p_idx) & 1):
                return False
    return True

def _local_search_remove(chosen_idx_set, deadline):
    """Try to remove points one by one until no more can be removed."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        chosen_list = list(chosen_idx_set)
        random.shuffle(chosen_list) # Randomize removal order to escape local minima
        for idx in chosen_list:
            if time.time() >= deadline:
                break
            new_chosen = set(chosen_idx_set)
            new_chosen.remove(idx)
            
            # Check if new_chosen is maximal
            new_blocked = _compute_blocked_mask(new_chosen)
            if _is_maximal(new_chosen, new_blocked):
                chosen_idx_set = new_chosen
                improved = True
                break  # Restart scan with new set
    return chosen_idx_set

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    all_indices = list(range(TOTAL_POINTS))
    
    best_set = None
    best_size = float('inf')
    
    # 1. Deterministic Greedy Initializations
    # Row-major
    order = list(range(TOTAL_POINTS))
    s = _greedy(order)
    if len(s) < best_size:
        best_size = len(s)
        best_set = s
        
    # Column-major (sort by y then x)
    order = sorted(all_indices, key=lambda i: (IDX_TO_PT[i][1], IDX_TO_PT[i][0]))
    s = _greedy(order)
    if len(s) < best_size:
        best_size = len(s)
        best_set = s
        
    # Diagonal
    order = sorted(all_indices, key=lambda i: (IDX_TO_PT[i][0] + IDX_TO_PT[i][1], IDX_TO_PT[i][0]))
    s = _greedy(order)
    if len(s) < best_size:
        best_size = len(s)
        best_set = s
        
    # Checkerboard
    order = sorted(all_indices, key=lambda i: ((IDX_TO_PT[i][0] + IDX_TO_PT[i][1]) % 2, IDX_TO_PT[i][0], IDX_TO_PT[i][1]))
    s = _greedy(order)
    if len(s) < best_size:
        best_size = len(s)
        best_set = s
        
    # 2. Randomized Search with Local Search
    rng = random.Random(42)
    
    # We will run multiple restarts
    # Each restart: Greedy with random order -> Local Search Remove
    while time.time() < deadline:
        order = all_indices[:]
        rng.shuffle(order)
        s = _greedy(order)
        
        # If this greedy result is already better, update best immediately
        # But we should always try to improve it with local search
        # However, local search is expensive. 
        # Strategy: Only do local search if len(s) is promising or randomly?
        # Given the score improvement from 1.0 to 1.33 to 1.53, local search is key.
        
        # Heuristic: If len(s) < best_size + 5, it's worth searching.
        # Or just always search if time permits.
        # Let's estimate time. Greedy is fast. Local search is O(K^3) roughly?
        # K is ~20-30. K^3 is ~27000. _compute_blocked is O(K^2). _is_maximal is O(N^2).
        # It should be fast enough to do many iterations.
        
        s = _local_search_remove(s, deadline)
        
        if len(s) < best_size:
            best_size = len(s)
            best_set = s
            
    # Final verification
    if best_set is None:
        # Fallback
        best_set = _greedy(all_indices)
        
    # Convert back to list of tuples
    result = [IDX_TO_PT[i] for i in best_set]
    
    # Optional: Final verification of validity and maximality
    # If invalid, fallback to a known good one (though we should be safe)
    # The _local_search_remove ensures maximality.
    # The _greedy ensures no-isosceles.
    
    return result
