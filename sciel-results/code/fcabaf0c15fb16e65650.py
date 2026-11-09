import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Target n
    # We know n=20 works. Let's try to find one.
    # If we can't find one quickly, we return n=19 or lower.
    
    def is_valid(n, edges):
        # Build adjacency list/matrix
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Check all subsets of size k >= 6
        # Optimization: If a subset is regular, its degrees are equal.
        # For k=6, degrees can be 0..5.
        
        # Helper to check if a specific mask (subset) is regular
        def is_regular_subset(mask, k):
            # Get vertices in mask
            # Instead of extracting vertices, we can iterate bits
            # But we need degrees for each vertex in the mask
            # deg[v] = popcount(adj[v] & mask)
            # All deg[v] must be equal.
            
            # For k=6, possible degrees: 0, 1, 2, 3, 4, 5
            # We can check if all degrees are the same.
            
            # Extract vertices
            verts = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            # Check degrees
            if not verts:
                return False
            
            d0 = (adj[verts[0]] & mask).bit_count()
            for v in verts[1:]:
                d = (adj[v] & mask).bit_count()
                if d != d0:
                    return False
            return True

        # Iterate subsets
        # To save time, we can stop early if we find a regular subset.
        # We need to check all k from 6 to n.
        
        # Optimization: If n is large, checking all k is expensive.
        # But for n=20, it's manageable.
        
        # Precompute masks? No, generate on fly or use combinations.
        # combinations(range(n), k) is slow for large k?
        # For n=20, k=10, 184k combinations.
        # Total combinations for k=6..20 is ~500k.
        # Each check involves ~k bit_count ops.
        # 500k * 10 = 5M ops. In Python, 5M ops might take ~1-2 seconds.
        # So one check takes ~2s.
        # We can do ~50 checks in 100s.
        
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check regularity
                # We can optimize by checking if degrees are all equal
                # Using the verts extraction method above is slow due to list creation.
                
                # Alternative:
                # For a given mask, compute degrees.
                # We can do this more efficiently if we iterate vertices.
                
                # Let's stick to the simple check but optimize the inner loop.
                # If k is large, it's less likely to be regular.
                # Most failures will be at k=6.
                
                verts = list(subset)
                d0 = (adj[verts[0]] & mask).bit_count()
                regular = True
                for v in verts[1:]:
                    d = (adj[v] & mask).bit_count()
                    if d != d0:
                        regular = False
                        break
                if regular:
                    return False
        
        return True

    def try_random_graph(n, seed):
        random.seed(seed)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    edges.append([i, j])
        return edges

    best_n = 5
    best_edges = []
    
    # Try to find a valid graph on 20 vertices
    # Use multiple seeds
    for seed in range(100):
        if time.time() > deadline:
            break
            
        edges = try_random_graph(20, seed)
        if is_valid(20, edges):
            return {'n': 20, 'edges': edges}
            
    # If 20 fails, try 19
    for seed in range(100):
        if time.time() > deadline:
            break
        edges = try_random_graph(19, seed)
        if is_valid(19, edges):
            return {'n': 19, 'edges': edges}
            
    # Fallback
    return {'n': best_n, 'edges': best_edges}
