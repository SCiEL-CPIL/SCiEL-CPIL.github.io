import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s limit, 5s margin
    
    # Helper to check if a graph is valid
    def is_valid(n, edges):
        # Build adjacency list as bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check all subsets of size >= 6
        # We iterate masks from 0 to 2^n - 1? No, that's 2^21 ~ 2M.
        # Better to iterate by size k from 6 to n.
        # Generating combinations is slow in pure Python.
        # Iterating all masks and filtering by bit_count >= 6 is 2^21 ~ 2M masks.
        # 2M * 21 ops ~ 42M ops. Might be ~2-4 seconds in Python.
        # Let's try this simple approach first. If too slow, optimize.
        
        for mask in range(2**n):
            k = mask.bit_count()
            if k < 6:
                continue
            
            # Check regularity
            # Get degrees of all vertices in mask
            # Optimization: Early exit if degrees differ
            first_v = -1
            first_deg = -1
            valid = True
            for v in range(n):
                if (mask >> v) & 1:
                    d = (adj[v] & mask).bit_count()
                    if first_v == -1:
                        first_v = v
                        first_deg = d
                    else:
                        if d != first_deg:
                            valid = False
                            break
            if valid:
                return False
        return True

    # Search strategy
    # Try n=21, then 20, then 19...
    for n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]:
        if time.time() > deadline - 10.0:
            # Not enough time for a complex search, return a small valid graph
            # n=5 is always valid
            return {'n': 5, 'edges': []}
        
        # Try to find a valid graph on n vertices
        # Use random search
        # Number of trials depends on remaining time
        max_trials = 1000
        found = False
        for trial in range(max_trials):
            if time.time() > deadline - 5.0:
                break
            
            # Generate random graph
            # Probability p=0.5 is a good start
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        edges.append([u, v])
            
            # Check validity
            if is_valid(n, edges):
                return {'n': n, 'edges': edges}
        
        # If n=21 failed, try n=20, etc.
        # We can also try different p values for next trials
        # But simple random p=0.5 is robust.
        
    # Fallback
    return {'n': 5, 'edges': []}
