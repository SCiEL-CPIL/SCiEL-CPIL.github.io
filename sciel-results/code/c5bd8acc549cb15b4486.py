import time
from itertools import combinations
import random

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    minv = P['minimum_vertices']
    
    # Build adjacency as bitmasks
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def popcount(x):
        return x.bit_count()
    
    def count_distinct_degrees(subset):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        degrees = [popcount(adj_mask[v] & mask) for v in subset]
        return len(set(degrees))
    
    deadline = time.time() + 85.0
    best_distinct = 0
    best_subset = list(range(minv))
    
    # Phase 1: Exhaustive search for larger subsets (k from n down to 10)
    # These are the most likely to have high distinct degree counts
    for k in range(n, minv - 1, -1):
        if best_distinct >= k:
            break  # Can't beat this with smaller k
        
        for combo in combinations(range(n), k):
            if time.time() > deadline:
                break
            mask = 0
            for v in combo:
                mask |= (1 << v)
            degrees = [popcount(adj_mask[v] & mask) for v in combo]
            distinct_count = len(set(degrees))
            
            if distinct_count > best_distinct:
                best_distinct = distinct_count
                best_subset = list(combo)
            
            if distinct_count == k:
                # Perfect: k distinct degrees in k vertices
                return [int(v) for v in best_subset]
    
    if time.time() > deadline:
        return [int(v) for v in best_subset]
    
    # Phase 2: For subsets of size minv to 9, use greedy + local search
    # Try multiple random initializations
    random.seed(42)
    
    for target_k in range(9, minv - 1, -1):
        if best_distinct >= target_k:
            break
        
        # Try multiple random starts
        num_starts = 200
        for start_idx in range(num_starts):
            if time.time() > deadline:
                break
            
            # Random initial subset
            subset = set(random.sample(range(n), target_k))
            
            # Greedy improvement: try swapping vertices
            improved = True
            while improved and time.time() < deadline:
                improved = False
                current_count = count_distinct_degrees(subset)
                
                if current_count >= target_k:
                    break
                
                # Try all possible swaps
                for v_in in list(subset):
                    for v_out in range(n):
                        if v_out in subset:
                            continue
                        if time.time() > deadline:
                            break
                        
                        new_subset = (subset - {v_in}) | {v_out}
                        new_count = count_distinct_degrees(new_subset)
                        
                        if new_count > current_count:
                            subset = new_subset
                            current_count = new_count
                            improved = True
                            break
                    if time.time() > deadline or improved:
                        break
                
                if time.time() > deadline:
                    break
            
            final_count = count_distinct_degrees(subset)
            if final_count > best_distinct:
                best_distinct = final_count
                best_subset = list(subset)
            
            if best_distinct >= target_k:
                break
    
    return [int(v) for v in best_subset]
