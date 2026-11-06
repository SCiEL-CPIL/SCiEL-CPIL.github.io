import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Construct Paley tournament on p=67
    p = 67
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    # Adjacency matrix: adj[i][j] = 1 if i -> j
    # Using a list of integers for in-masks for speed
    # in_mask[u] has bit v set if v -> u
    in_mask = [0] * p
    for i in range(p):
        for j in range(p):
            if i != j:
                if (j - i) % p in squares:
                    # i -> j
                    in_mask[j] |= (1 << i)
                else:
                    # j -> i
                    in_mask[i] |= (1 << j)
                    
    # Helper to check if a set of active vertices is valid
    # active_list: list of vertex indices from 0..p-1
    # Returns True if every 4-set in active_list has a dominator in active_list
    def is_valid(active_list):
        n = len(active_list)
        if n < 4:
            return True
            
        # Create a mapping from active vertex to a compact index?
        # Or just use the original indices and bit masks.
        # We need to check intersections of in_masks.
        # But in_mask[u] contains bits for ALL vertices 0..p-1.
        # We only care about bits corresponding to active vertices.
        # Let's create a mask of active vertices.
        active_mask = 0
        for v in active_list:
            active_mask |= (1 << v)
            
        # Iterate over all 4-combinations of active_list
        # To speed up, we can precompute the 4-combinations?
        # The active_list changes size, so we generate them on the fly.
        for i in range(n):
            v1 = active_list[i]
            m1 = in_mask[v1] & active_mask # Only consider dominators in active set?
            # Wait, the dominator must be in the active set.
            # So we intersect with active_mask.
            
            for j in range(i+1, n):
                v2 = active_list[j]
                m12 = m1 & in_mask[v2]
                if m12 == 0:
                    # No dominator in active set for {v1, v2, ...} ?
                    # Not yet, we need 4 vertices.
                    # If m12 is 0, then any superset will also have 0 intersection?
                    # Yes, if A & B = 0, then (A & B) & C = 0.
                    # So we can break early?
                    # No, we need to check if there is a dominator for the specific 4-set.
                    # If the intersection of the first two is empty, the intersection of 4 is empty.
                    # So this 4-set is bad.
                    # But we can prune the inner loops?
                    # If m12 == 0, then for any k, l, the intersection will be 0.
                    # So we can return False immediately?
                    # Yes, if we found a pair with no common in-neighbor in the active set,
                    # then any 4-set containing that pair is bad.
                    # Wait, is that true?
                    # Dominator $v$ must point to $v1$ and $v2$.
                    # If no $v$ in active set points to both $v1$ and $v2$, then no $v$ can point to $v1, v2, v3, v4$.
                    # So yes, if $in\_mask[v1] \cap in\_mask[v2] \cap active\_mask == 0$, then any 4-set containing $v1, v2$ is bad.
                    # So we can return False.
                    return False
                    
                for k in range(j+1, n):
                    v3 = active_list[k]
                    m123 = m12 & in_mask[v3]
                    if m123 == 0:
                        return False
                        
                    for l in range(k+1, n):
                        v4 = active_list[l]
                        m1234 = m123 & in_mask[v4]
                        if m1234 == 0:
                            return False
                            
        return True

    current_active = list(range(p))
    
    # Check initial validity
    if not is_valid(current_active):
        # If Paley 67 is not valid, we have a problem.
        # But it is known to be valid.
        pass

    # Greedy removal
    # We try to remove vertices one by one.
    # We use multiple random restarts/orders to avoid local minima.
    
    best_n = len(current_active)
    best_active = list(current_active)
    
    # Time budget for search
    search_deadline = start_time + 100.0
    
    # Try several random permutations
    for attempt in range(50): # Max 50 attempts
        if time.time() > search_deadline:
            break
            
        random.shuffle(current_active)
        
        for v in current_active:
            if time.time() > search_deadline:
                break
                
            # Try removing v
            test_active = [x for x in current_active if x != v]
            
            if is_valid(test_active):
                current_active = test_active
                if len(current_active) < best_n:
                    best_n = len(current_active)
                    best_active = list(current_active)
                    
    # Use the best active set found
    final_active = best_active
    n_out = len(final_active)
    
    # Construct orientation
    # Map old index to new index
    old_to_new = {v: i for i, v in enumerate(final_active)}
    
    orientation = []
    for i in range(n_out):
        for j in range(i+1, n_out):
            old_i = final_active[i]
            old_j = final_active[j]
            # Check direction in original Paley tournament
            # in_mask[old_j] has bit old_i if old_i -> old_j
            if in_mask[old_j] & (1 << old_i):
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
                
    return {'n': n_out, 'orientation': orientation}
