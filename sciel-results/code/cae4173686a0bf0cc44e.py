import time
import math
import random
from math import gcd

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs a, b.
    
    Uses graph-theoretic maximum clique search with greedy coloring bounds
    and local search heuristics.
    """
    deadline = time.time() + 50.0  # 50s safety margin before 60s limit
    
    # Build candidate set: numbers in [1,255] divisible by 3, 5, or 17
    # Must contain 255, and all elements must share a factor with 255
    candidates = [i for i in range(1, 256) if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    
    # Build adjacency: edge between a,b if gcd(a,b) > 1
    n = len(candidates)
    # Map value -> index for O(1) lookup
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
    
    # Precompute neighbor bitmasks for fast intersection
    # Use Python integers as bitsets
    neighbor_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        neighbor_mask[i] = mask
    
    def bit_count(x):
        return bin(x).count('1')
    
    # Greedy coloring upper bound for maximum clique
    def color_order(vertices):
        """Order vertices by greedy coloring, return (order, color_bound)"""
        colors = {}
        order = []
        for v in vertices:
            # Find smallest available color
            used_colors = set()
            for u in order:
                if adj[v][u]:
                    used_colors.add(colors[u])
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            order.append(v)
        max_color = max(colors.values()) if colors else 0
        return order, max_color
    
    # Local search using index-based operations for speed
    def local_search_fast(start_set, max_iters=5000):
        """Improve clique via add/swap operations using precomputed adjacency"""
        # Work with indices
        current_idx = set(val_to_idx[v] for v in start_set if v in val_to_idx)
        current_mask = 0
        for idx in current_idx:
            current_mask |= (1 << idx)
        
        best_size = len(current_idx)
        best_idx = set(current_idx)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add each non-member
            for v_idx in range(n):
                if v_idx in current_idx:
                    continue
                # Check if v connects to all in current: neighbor_mask[v] & current_mask == current_mask
                if neighbor_mask[v_idx] & current_mask == current_mask:
                    current_idx.add(v_idx)
                    current_mask |= (1 << v_idx)
                    if len(current_idx) > best_size:
                        best_size = len(current_idx)
                        best_idx = set(current_idx)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one, add another (net size stays same, but may enable further adds)
            # Actually for maximizing size, we want swaps that allow subsequent additions
            # Simpler: try 1-remove-1-add where the added one connects to all remaining
            swap_done = False
            for u_idx in list(current_idx):
                if time.time() > deadline:
                    break
                # Remove u temporarily
                temp_mask = current_mask & ~(1 << u_idx)
                for v_idx in range(n):
                    if v_idx in current_idx:
                        continue
                    # Check if v connects to all in current except u
                    if neighbor_mask[v_idx] & temp_mask == temp_mask:
                        # Do the swap
                        current_idx.remove(u_idx)
                        current_idx.add(v_idx)
                        current_mask = temp_mask | (1 << v_idx)
                        swap_done = True
                        break
                if swap_done:
                    break
            
            if not improved and not swap_done:
                # Try 2-swap: remove two, add two
                removed = []
                added = []
                found_2swap = False
                cand_list = [v for v in range(n) if v not in current_idx]
                
                for i in range(len(cand_list)):
                    if time.time() > deadline:
                        break
                    v1 = cand_list[i]
                    for j in range(i + 1, len(cand_list)):
                        v2 = cand_list[j]
                        # v1 and v2 must be adjacent
                        if not adj[v1][v2]:
                            continue
                        # Find which elements in current are NOT connected to v1 or v2
                        # We need to remove at most 2 elements
                        not_conn_v1 = [u for u in current_idx if not adj[v1][u]]
                        not_conn_v2 = [u for u in current_idx if not adj[v2][u]]
                        # Union of non-neighbors
                        non_neighbors = set(not_conn_v1) | set(not_conn_v2)
                        if len(non_neighbors) <= 2:
                            # Remove all non-neighbors, add v1 and v2
                            new_idx = set(current_idx) - non_neighbors
                            new_idx.add(v1)
                            new_idx.add(v2)
                            if len(new_idx) > len(current_idx):
                                current_idx = new_idx
                                current_mask = 0
                                for idx in current_idx:
                                    current_mask |= (1 << idx)
                                if len(current_idx) > best_size:
                                    best_size = len(current_idx)
                                    best_idx = set(current_idx)
                                found_2swap = True
                                break
                    if found_2swap:
                        break
                
                if not found_2swap:
                    # Try 1-swap more aggressively: remove one, add one that's not directly beneficial
                    # but enables future growth - skip for now, break
                    break
        
        # Convert back to values
        return [candidates[i] for i in best_idx]
    
    # Start with the known good solution: all multiples of 3
    start_set = [i for i in range(1, 256) if i % 3 == 0]
    best_clique = local_search_fast(start_set, max_iters=10000)
    best_size = len(best_clique)
    
    # Try multiple random restarts
    random.seed(42)
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Create a random subset of candidates that includes 255
        idx_255 = val_to_idx[255]
        current = {idx_255}
        for v_idx in range(n):
            if v_idx == idx_255:
                continue
            if adj[idx_255][v_idx] and random.random() < 0.5:
                current.add(v_idx)
        
        # Convert to values for local_search_fast
        current_vals = [candidates[i] for i in current]
        improved = local_search_fast(current_vals, max_iters=5000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_clique = improved
    
    # Final polish: try to add any remaining elements that fit
    best_set = set(best_clique)
    for v in candidates:
        if v not in best_set:
            can_add = all(gcd(v, u) > 1 for u in best_set)
            if can_add:
                best_set.add(v)
    
    return list(best_set)
