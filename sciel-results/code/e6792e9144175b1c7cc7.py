import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    # We want to maximize n. Try n=21 first. If successful, great.
    # If not, fall back to n=20.
    
    def popcount(x):
        return bin(x).count('1')
    
    def find_violation(adj, n):
        """
        Find a subset of size >= 6 that induces a regular subgraph.
        adj: list of ints, adj[i] is bitmask of neighbors.
        Returns (subset_mask, size) or (None, 0) if none found.
        """
        # Check sizes from 6 to n
        # We can iterate over all masks, but it's better to iterate by size
        # Using Gosper's hack or just combinations for small k
        # For n=21, total subsets is ~2M. Iterating all is fine.
        
        # Optimization: Check size 6 first as they are most common violations?
        # Actually, if a size 7 is regular, it's also a violation.
        # We just need ANY violation.
        
        # Iterate over all masks from 1 to 2^n - 1?
        # That's 2^21 = 2M masks.
        # For each mask, check popcount >= 6.
        # Then check regularity.
        
        # Better: Iterate by size k from 6 to n.
        vertices = range(n)
        
        # To speed up, we can just iterate all masks and filter by popcount
        # But generating combinations is cleaner in Python for small k.
        # However, for n=21, combinations of size 6 are 54k.
        # Size 7: 116k.
        # Size 8: 203k.
        # Total is ~1.1M.
        
        # Let's just iterate all masks from 0 to 2^n - 1
        # This is 2M iterations. In Python, this might take 1-2 seconds.
        # We can do this ~50 times in 120s.
        
        limit = 1 << n
        # We only care about masks with popcount >= 6
        # We can precompute popcounts? No, bin(x).count('1') is fast enough in C.
        
        for mask in range(limit):
            k = popcount(mask)
            if k < 6:
                continue
            
            # Check if induced subgraph on 'mask' is regular
            # Get the vertices in the mask
            # We need to calculate degree of each vertex in the subset
            # deg[v] = popcount(adj[v] & mask)
            
            # We can early exit if degrees are not equal
            # To do this efficiently, we can iterate through the set bits of mask
            # Or just iterate through all vertices 0..n-1 and check if they are in mask
            
            # Let's collect degrees
            # Optimization: If k is small, this is fast.
            
            # We can check regularity by:
            # 1. Pick first vertex v in mask.
            # 2. Calculate its degree d0.
            # 3. For all other vertices u in mask, check if deg(u) == d0.
            
            # Find first vertex
            # We can just iterate u from 0 to n-1
            first_deg = -1
            is_regular = True
            count = 0
            
            for u in range(n):
                if (mask >> u) & 1:
                    deg = popcount(adj[u] & mask)
                    if count == 0:
                        first_deg = deg
                    else:
                        if deg != first_deg:
                            is_regular = False
                            break
                    count += 1
            
            if is_regular and count == k:
                return mask, k
        
        return None, 0

    def get_edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    edges.append([i, j])
        return edges

    def solve(n, time_budget):
        """
        Attempt to find a valid graph on n vertices within time_budget.
        Returns (n, edges) or (None, None).
        """
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        best_adj = adj[:]
        best_status = None # None means not valid yet
        
        while time.time() - start_time < time_limit - time_budget:
            # Check for violation
            mask, k = find_violation(adj, n)
            
            if mask is None:
                # Valid graph found!
                return n, get_edges_from_adj(adj, n)
            
            # Found a violation. Flip a random edge within the violating subset
            # to try to break regularity.
            # Get vertices in mask
            verts = [i for i in range(n) if (mask >> i) & 1]
            
            # Pick two random vertices from the subset to flip the edge between them
            # Flipping an edge inside the subset is most effective
            u, v = random.sample(verts, 2)
            
            # Flip edge (u, v)
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Occasionally restart if stuck?
            # For now, just keep flipping.
            
            # If we are taking too long on one graph, maybe restart?
            # The time check is at the top of the loop.
            
        # If we run out of time, return None
        return None, None

    # Strategy:
    # 1. Try to find n=21. Give it 60 seconds.
    # 2. If successful, return it.
    # 3. If not, try to find n=20. Give it 50 seconds.
    # 4. If successful, return it.
    # 5. Fallback to n=5.
    
    # Try n=21
    result = solve(21, 60.0)
    if result[0] is not None:
        return {'n': result[0], 'edges': result[1]}
        
    # Try n=20
    result = solve(20, 50.0)
    if result[0] is not None:
        return {'n': result[0], 'edges': result[1]}
        
    # Fallback
    return {'n': 5, 'edges': []}
