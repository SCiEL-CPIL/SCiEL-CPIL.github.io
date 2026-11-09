import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def popcount(x):
        return x.bit_count()

    def verify_full(adj_masks, n):
        """
        Verify no subset of size >= 6 is regular.
        adj_masks: list of ints
        n: number of vertices
        """
        # Precompute masks for all subsets? 
        # For n <= 20, 2^n is manageable but iterating combinations is fine.
        
        # Optimization: Check size 6 first.
        for size in range(6, n + 1):
            for subset in combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check degrees
                degs = []
                for v in subset:
                    degs.append(popcount(adj_masks[v] & mask))
                
                if len(set(degs)) == 1:
                    return False
        return True

    def check_addition(adj_masks, new_mask):
        """
        Check if adding a new vertex with adjacency 'new_mask' to the graph 
        represented by adj_masks (size n) results in a valid graph.
        Only checks subsets containing the new vertex.
        """
        n = len(adj_masks)
        new_idx = n
        
        # We need to check subsets S of {0..n-1} such that |S| + 1 >= 6
        # i.e., |S| >= 5.
        # Subset in new graph: S U {new_idx}
        
        # Precompute masks for subsets of existing vertices?
        # It's faster to just iterate combinations.
        
        # Check sizes from 5 up to n-1
        for size_s in range(5, n):
            # Subsets of size size_s from 0..n-1
            for subset in combinations(range(n), size_s):
                mask_s = 0
                for v in subset:
                    mask_s |= (1 << v)
                
                # Degree of new vertex in (S U {new_idx})
                deg_new = popcount(new_mask & mask_s)
                
                # Degrees of vertices in S
                # For u in S, deg_u = popcount(adj_masks[u] & mask_s) + (1 if u connected to new_idx else 0)
                # u connected to new_idx iff bit u is set in new_mask
                
                all_equal = True
                for u in subset:
                    connected = (new_mask >> u) & 1
                    deg_u = popcount(adj_masks[u] & mask_s) + connected
                    if deg_u != deg_new:
                        all_equal = False
                        break
                
                if all_equal:
                    return False
        return True

    # --- Step 1: Find a base valid graph ---
    # Try to find a valid graph with n=10, 11, ...
    # Use random graphs
    base_n = 0
    base_adj_masks = None
    
    # Try small n first to get a valid base
    for target_n in range(10, 15):
        if time_left() < 5.0:
            break
        found = False
        for seed in range(100):
            if time_left() < 2.0:
                break
            rng = np.random.default_rng(seed)
            adj_masks = [0] * target_n
            # Generate random edges
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if rng.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            if verify_full(adj_masks, target_n):
                base_n = target_n
                base_adj_masks = adj_masks.copy()
                found = True
                break
        if found:
            break
    
    if base_adj_masks is None:
        # Fallback: empty graph on 5 vertices is valid (no subset of size 6)
        # But we want to maximize n.
        # Return a small valid graph
        return {'n': 5, 'edges': []}

    # --- Step 2: Greedy extension ---
    current_n = base_n
    current_adj_masks = base_adj_masks
    
    # Try to extend to n=22
    for target_n in range(base_n + 1, 23):
        if time_left() < 10.0:
            break
        
        added = False
        # Try many random edge sets for the new vertex
        for attempt in range(200):
            if time_left() < 5.0:
                break
            
            rng = np.random.default_rng(attempt * 1000 + target_n)
            new_mask = 0
            for i in range(current_n):
                if rng.random() < 0.5:
                    new_mask |= (1 << i)
            
            if check_addition(current_adj_masks, new_mask):
                # Valid addition
                # Update adj_masks
                new_adj_masks = current_adj_masks.copy()
                # Add connections to existing vertices
                for i in range(current_n):
                    if (new_mask >> i) & 1:
                        new_adj_masks[i] |= (1 << current_n)
                
                # Add new vertex's mask
                new_adj_masks.append(new_mask)
                
                current_adj_masks = new_adj_masks
                current_n += 1
                added = True
                break
        
        if not added:
            # Could not add vertex. 
            # We might try to backtrack or just stop.
            # For simplicity, stop.
            break

    # --- Step 3: Convert to output format ---
    n = current_n
    edges = []
    for i in range(n):
        mask = current_adj_masks[i]
        for j in range(i + 1, n):
            if (mask >> j) & 1:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}
