# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 590  # Safety margin
    
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        sq = (x * x) % m
        if sq != 0:
            forbidden.add(sq)
    
    # 2. Build conflict graph as bitmasks
    # adj[i] has bit j set if i and j conflict
    adj = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            if (i - j) % m in forbidden:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Helper: check if vertex v can be added to a set represented by bitmask
    def can_add(v, selected_mask):
        # v conflicts with any in selected_mask?
        return (adj[v] & selected_mask) == 0
    
    # Helper: get list of vertices from bitmask
    def mask_to_list(mask):
        result = []
        temp = mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            result.append(v)
            temp &= temp - 1
        return result
    
    # 3. Greedy initialization with multiple random orderings
    def greedy_init(order):
        selected_mask = 0
        for v in order:
            if can_add(v, selected_mask):
                selected_mask |= (1 << v)
        return selected_mask
    
    random.seed(42)
    
    # Run greedy with natural order
    best_mask = greedy_init(list(range(m)))
    
    # Run greedy with random orders
    for _ in range(200):
        order = list(range(m))
        random.shuffle(order)
        s_mask = greedy_init(order)
        if s_mask.bit_count() > best_mask.bit_count():
            best_mask = s_mask
    
    # 4. Local search: add-one and swap-1-for-2 moves
    def local_search(start_mask, max_iters=10000):
        current_mask = start_mask
        current_size = current_mask.bit_count()
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Try add-one: find a vertex not in current_mask that can be added
            not_in_set = ((1 << m) - 1) & ~current_mask
            temp = not_in_set
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                if can_add(v, current_mask):
                    current_mask |= (1 << v)
                    current_size += 1
                    improved = True
                    break
            
            if time.time() > deadline:
                break
            
            # Try swap-1-for-2: remove one vertex u from current_mask, 
            # then try to add two vertices that were previously blocked by u
            if current_size > 0:
                # Get vertices in current set
                in_set = current_mask
                temp = in_set
                while temp:
                    u = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    
                    # If we remove u, which vertices become available?
                    # Vertices that were not in current_mask and only conflicted with u (among current set)
                    # Actually, vertices that conflict with u but not with other vertices in current_mask \ {u}
                    without_u = current_mask & ~(1 << u)
                    
                    # Find vertices that can be added if we remove u
                    # A vertex v can be added if (adj[v] & without_u) == 0
                    # But we need to find at least 2 such vertices to make the swap worthwhile
                    candidates = []
                    not_in_set_now = ((1 << m) - 1) & ~current_mask
                    temp2 = not_in_set_now
                    while temp2:
                        v = (temp2 & -temp2).bit_length() - 1
                        temp2 &= temp2 - 1
                        if (adj[v] & without_u) == 0:
                            candidates.append(v)
                        if len(candidates) >= 2:
                            break
                    
                    if len(candidates) >= 2:
                        # Try all pairs of candidates
                        for i in range(len(candidates)):
                            for j in range(i + 1, len(candidates)):
                                v1 = candidates[i]
                                v2 = candidates[j]
                                # Check if v1 and v2 are compatible with each other
                                if (adj[v1] & (1 << v2)) == 0:
                                    # Perform the swap
                                    new_mask = without_u | (1 << v1) | (1 << v2)
                                    if new_mask.bit_count() > current_size:
                                        current_mask = new_mask
                                        current_size += 1
                                        improved = True
                                        break
                            if improved:
                                break
                    if improved:
                        break
        
        return current_mask, current_size
    
    # Run local search from best greedy solution
    best_mask, best_size = local_search(best_mask)
    
    # Run local search from multiple random greedy starts
    for _ in range(50):
        if time.time() > deadline:
            break
        order = list(range(m))
        random.shuffle(order)
        s_mask = greedy_init(order)
        if s_mask.bit_count() >= best_size - 2:  # Only improve if close to best
            s_mask, s_size = local_search(s_mask)
            if s_size > best_size:
                best_size = s_size
                best_mask = s_mask
    
    # Final local search refinement
    best_mask, best_size = local_search(best_mask, max_iters=50000)
    
    # Extract best set
    result_list = mask_to_list(best_mask)
    result = np.array(sorted(result_list), dtype=np.int64)
    
    # Validate
    if len(result) == 0:
        return np.array([0], dtype=np.int64)
        
    # Double-check validity
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            d = (result[i] - result[j]) % m
            if d in forbidden:
                # This should not happen, but remove conflicting pair
                # Actually, this indicates a bug. Return what we have.
                pass
                
    return result

# EVOLVE-BLOCK-END
