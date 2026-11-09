import itertools
import time
import random
import numpy as np

N = 16
TOTAL_POINTS = N * N
ALL_POINTS = list(itertools.product(range(N), repeat=2))
PT_TO_IDX = {p: i for i, p in enumerate(ALL_POINTS)}
IDX_TO_PT = {i: p for i, p in enumerate(ALL_POINTS)}
ALL_MASK = (1 << TOTAL_POINTS) - 1

# Precompute isosceles conflict masks
# iso_mask[i][j] is a bitmask of all k such that {i, j, k} forms an isosceles triangle
# Note: i and j are not included in the mask.
iso_mask = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]

coords = np.array(ALL_POINTS)

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

def _compute_blocked_mask(chosen_idx_list):
    """Compute the blocked mask for a list of chosen indices."""
    blocked = 0
    n = len(chosen_idx_list)
    for i in range(n):
        idx_i = chosen_idx_list[i]
        for j in range(i + 1, n):
            idx_j = chosen_idx_list[j]
            blocked |= iso_mask[idx_i][idx_j]
    return blocked

def _is_maximal(chosen_idx_set, blocked_mask):
    """Check if no further grid point can be added to chosen."""
    chosen_mask = 0
    for i in chosen_idx_set:
        chosen_mask |= (1 << i)
    # All points must be either in chosen or blocked
    return (chosen_mask | blocked_mask) == ALL_MASK

def _greedy(order_idx):
    """Greedy construction using a given ordering of point indices."""
    chosen_idx_set = set()
    blocked_mask = 0
    for p_idx in order_idx:
        if p_idx in chosen_idx_set:
            continue
        if (blocked_mask >> p_idx) & 1:
            continue
        chosen_idx_set.add(p_idx)
        # Update blocked mask
        # New point p_idx conflicts with all existing points i
        for i in chosen_idx_set:
            if i != p_idx:
                blocked_mask |= iso_mask[p_idx][i]
    return chosen_idx_set

def _local_search_remove_swap(chosen_idx_set, deadline, rng):
    """Try to remove points one by one. Also try swaps to escape local minima."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        chosen_list = list(chosen_idx_set)
        rng.shuffle(chosen_list)
        
        # Phase 1: Pure Removal
        for idx in chosen_list:
            if time.time() >= deadline:
                return chosen_idx_set
            if idx not in chosen_idx_set:
                continue
            
            new_chosen = set(chosen_idx_set)
            new_chosen.remove(idx)
            
            # Check if new_chosen is maximal
            new_blocked = _compute_blocked_mask(list(new_chosen))
            if _is_maximal(new_chosen, new_blocked):
                chosen_idx_set = new_chosen
                improved = True
                break  # Restart scan with new set
        
        # Phase 2: Swaps (if no removals were possible, try swaps to change the structure)
        # This helps escape local minima where a point is "essential" for blocking,
        # but replacing it with another point allows further removals later.
        if not improved and time.time() < deadline:
            swaps_made = False
            # Try a limited number of random swaps
            for _ in range(20):
                if time.time() >= deadline:
                    break
                if not chosen_idx_set:
                    break
                # Pick a point to remove
                remove_idx = rng.choice(list(chosen_idx_set))
                # Pick a point to add (must not be in chosen and not blocked by current set? 
                # Actually, if we remove first, the blocked set changes.)
                
                # Let's try all points not in chosen as candidates to add? 
                # That's 256 checks. Might be slow if done many times.
                # Sample a few.
                candidates = [i for i in range(TOTAL_POINTS) if i not in chosen_idx_set]
                if not candidates:
                    break
                add_idx = rng.choice(candidates)
                
                new_chosen = set(chosen_idx_set)
                new_chosen.remove(remove_idx)
                new_chosen.add(add_idx)
                
                # Check validity: No isosceles?
                # Since we removed one and added one, we need to check if the new point 
                # forms an isosceles with any pair in the rest.
                # And the rest was already valid.
                # So we only need to check if add_idx forms an iso with any pair in (new_chosen - {add_idx})
                
                is_iso = False
                rest = new_chosen - {add_idx}
                rest_list = list(rest)
                for i in range(len(rest_list)):
                    i_idx = rest_list[i]
                    # Check if add_idx and i_idx form an iso with any j in rest_list
                    # We can just check if the triangle (add_idx, i_idx, j_idx) is iso for any j.
                    # This is O(K^2).
                    for j in range(i+1, len(rest_list)):
                        j_idx = rest_list[j]
                        d_ai = _d2_idx(add_idx, i_idx)
                        d_aj = _d2_idx(add_idx, j_idx)
                        d_ij = _d2_idx(i_idx, j_idx)
                        if d_ai == d_aj or d_ai == d_ij or d_aj == d_ij:
                            is_iso = True
                            break
                    if is_iso:
                        break
                if is_iso:
                    continue
                
                # Check maximality
                new_blocked = _compute_blocked_mask(list(new_chosen))
                if _is_maximal(new_chosen, new_blocked):
                    chosen_idx_set = new_chosen
                    swaps_made = True
                    # Don't break immediately, might find a better swap? 
                    # But size is same. Break to re-evaluate removals.
                    break
            
            if swaps_made:
                improved = True # Allow another round of removals

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
        
    # Column-major
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
    while time.time() < deadline:
        order = all_indices[:]
        rng.shuffle(order)
        s = _greedy(order)
        
        # Local Search
        s = _local_search_remove_swap(s, deadline, rng)
        
        if len(s) < best_size:
            best_size = len(s)
            best_set = s
            
        # If we found a very small set, we can be more aggressive with time
        # Or just continue.
        
    # Final verification
    if best_set is None:
        best_set = _greedy(all_indices)
        
    result = [IDX_TO_PT[i] for i in best_set]
    return result
