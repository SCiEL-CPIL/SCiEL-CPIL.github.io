import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit

    N = 256
    
    # 1. Identify valid vertices (2*a is not a perfect square)
    valid_indices = []
    for i in range(1, N + 1):
        s = 2 * i
        root = math.isqrt(s)
        if root * root != s:
            valid_indices.append(i)
            
    M = len(valid_indices)
    
    # 2. Build conflict graph using bitmasks
    # neighbors[i] contains bit j if valid_indices[i] + valid_indices[j] is a square
    neighbors = [0] * M
    
    # Precompute squares up to 2*256 = 512
    max_sum = 2 * N
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
        
    for i in range(M):
        val_i = valid_indices[i]
        mask_i = 0
        for j in range(i + 1, M):
            val_j = valid_indices[j]
            if val_i + val_j in squares:
                mask_i |= (1 << j)
                neighbors[j] |= (1 << i)
        neighbors[i] = mask_i

    # 3. Solve Maximum Independent Set using Branch and Bound
    best_size = 0
    best_set_list = []
    
    # Global variables for solver state
    # We use a list to store the current path
    
    def popcount(x):
        return x.bit_count()

    def solve(candidates_mask, current_size, current_set_list):
        nonlocal best_size, best_set_list
        
        # Time check
        if time.time() > deadline:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_list = current_set_list[:]
            return
            
        # Upper bound pruning
        if current_size + popcount(candidates_mask) <= best_size:
            return
            
        # Select vertex with highest degree in the induced subgraph
        # This helps in pruning branches that include high-degree vertices early
        best_v = -1
        best_deg = -1
        temp_mask = candidates_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            # Count neighbors of idx that are also in candidates_mask
            deg = (neighbors[idx] & candidates_mask).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = idx
            temp_mask ^= lsb
            
        if best_v == -1:
            return

        # Branch 1: Include best_v
        # Remove best_v and its neighbors from candidates
        new_candidates = candidates_mask & ~(neighbors[best_v] | (1 << best_v))
        current_set_list.append(valid_indices[best_v])
        solve(new_candidates, current_size + 1, current_set_list)
        current_set_list.pop()
        
        # Branch 2: Exclude best_v
        # Remove best_v from candidates
        new_candidates = candidates_mask & ~(1 << best_v)
        solve(new_candidates, current_size, current_set_list)

    # Initial call
    initial_mask = (1 << M) - 1
    solve(initial_mask, 0, [])
    
    # If no solution found (e.g. timeout very early), return a valid single element
    if not best_set_list:
        if valid_indices:
            return [valid_indices[0]]
        return []

    return best_set_list
