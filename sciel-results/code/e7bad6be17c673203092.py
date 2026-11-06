import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Target n values to try, in descending order
    for n in [22, 21, 20]:
        if time.time() > deadline - 20.0:
            break
            
        # Try to find a valid graph of size n using local search
        # Initialize with random graph
        # Use multiple random restarts if needed
        
        max_restarts = 10 if n == 22 else 50
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Random initialization
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search loop
            found_valid = False
            max_steps = 500
            for step in range(max_steps):
                if time.time() > deadline - 5.0:
                    break
                    
                # Find a bad subset
                bad_subset = find_bad_subset(n, adj)
                
                if bad_subset is None:
                    # Valid graph found
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                
                # Flip a random edge in the bad subset
                # Get vertices in bad_subset
                verts = []
                m = bad_subset
                while m:
                    v = (m & -m).bit_length() - 1
                    verts.append(v)
                    m &= m - 1
                
                # Pick two random distinct vertices from the subset
                # and flip the edge between them
                if len(verts) < 2:
                    # Should not happen for k>=6
                    break
                u, v = random.sample(verts, 2)
                
                # Flip edge (u, v)
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # If we exited the step loop without finding valid, try next restart
    
    # Fallback: return a small valid graph
    # n=5 is always valid (no subset of size >=6)
    return {'n': 5, 'edges': []}

def find_bad_subset(n, adj):
    """
    Find any subset of size >= 6 that induces a regular subgraph.
    Returns the bitmask of the subset, or None if no such subset exists.
    """
    # Iterate over all subsets of size k from n down to 6
    # To speed up, we can randomize the order or just iterate.
    # Iterating all 2^n subsets is too slow if done naively in Python for n=22 (4M subsets).
    # But we can prune: if a subset S is regular, any superset is not necessarily regular.
    # Actually, we just need *one*.
    
    # Efficient way:
    # For each mask from 1 to 2^n - 1:
    #   if popcount(mask) >= 6:
    #      check regular
    
    # Optimization:
    # We can iterate by size k.
    # For k in range(n, 5, -1):
    #    for each subset of size k:
    
    # Generating all subsets of size k can be done with Gosper's hack.
    
    for k in range(n, 5, -1):
        # Generate all subsets of size k
        # Start with the first subset of size k
        # mask = (1 << k) - 1
        # Then use Gosper's hack to get next
        # But we need to stop if time is running out? 
        # The caller checks time. We can just return as soon as we find one.
        
        # Initialize first mask
        mask = (1 << k) - 1
        limit = 1 << n
        
        while mask < limit:
            # Check if this subset is regular
            if is_regular(mask, adj):
                return mask
            
            # Gosper's hack to get next mask of same popcount
            c = mask & -mask
            r = mask + c
            mask = (((r ^ mask) >> 2) // c) | r
            if mask >= limit:
                break
        
        # If we finished all subsets of size k, try k-1
        # Note: if we didn't find a bad subset of size k, we must check smaller k.
        # It is possible that no subset of size n is regular, but a subset of size n-1 is.
        
    return None

def is_regular(mask, adj):
    """
    Check if the induced subgraph on vertices in 'mask' is regular.
    """
    # Get degrees of vertices in mask
    # We need to check if all degrees are equal.
    
    # First, get the list of vertices
    # Or just iterate and compare.
    
    # Optimization: 
    # deg(v) in mask = popcount(adj[v] & mask)
    
    # We can pick the first vertex, get its degree, and compare with others.
    
    # Get first vertex
    v0 = (mask & -mask).bit_length() - 1
    target_deg = (adj[v0] & mask).bit_count()
    
    # Check all other vertices
    m = mask & ~(1 << v0)
    while m:
        v = (m & -m).bit_length() - 1
        m &= m - 1
        deg = (adj[v] & mask).bit_count()
        if deg != target_deg:
            return False
            
    return True
