import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def verify_c4_free(n, edges):
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
        for i in range(n):
            for j in range(i + 1, n):
                common = np.sum(adj[i] & adj[j])
                if common >= 2:
                    return False
        return True

    def greedy_c4_free(n, seed):
        rng = random.Random(seed)
        # Bitset representation for neighbors
        # n <= 47, so we can use Python integers as bitsets
        neighbors = [0] * n
        
        # Generate all edges
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(edges)
        
        added_edges = []
        
        for u, v in edges:
            # Check if adding (u, v) creates a C4
            # This happens if |N(u) \cap N(v)| >= 2
            common = neighbors[u] & neighbors[v]
            # Count bits in common
            if common.bit_count() >= 2:
                continue
            
            # Safe to add
            neighbors[u] |= (1 << v)
            neighbors[v] |= (1 << u)
            added_edges.append([u, v])
            
        # Calculate min degree
        min_deg = min(neighbors[i].bit_count() for i in range(n))
        
        return min_deg, added_edges

    best_result = None
    
    # Try to find a solution for n=47
    # Min degree required: 47 - 39 = 8
    
    # We will run multiple random restarts
    # Each restart is O(n^2 * cost_check). n=47, n^2/2 ~ 1100 edges.
    # Check is O(1) with bitsets.
    # So one restart is very fast. We can do many.
    
    max_n = 47
    target_min_deg = max_n - 39
    
    found = False
    for i in range(10000):
        if time.time() > deadline - 10:
            break
            
        min_deg, edges = greedy_c4_free(max_n, seed=i)
        
        if min_deg >= target_min_deg:
            # Verify just in case
            if verify_c4_free(max_n, edges):
                best_result = {'n': max_n, 'edges': edges}
                found = True
                break
            else:
                # This should not happen if logic is correct
                continue

    if not found:
        # Try n=46
        max_n = 46
        target_min_deg = max_n - 39
        
        for i in range(10000):
            if time.time() > deadline - 5:
                break
                
            min_deg, edges = greedy_c4_free(max_n, seed=10000 + i)
            
            if min_deg >= target_min_deg:
                if verify_c4_free(max_n, edges):
                    best_result = {'n': max_n, 'edges': edges}
                    found = True
                    break

    if not found:
        # Fallback to n=45 or lower if 46 fails (unlikely)
        # But n=46 is very likely to succeed.
        # If all else fails, return a trivial graph
        best_result = {'n': 39, 'edges': []}
        
    return best_result
