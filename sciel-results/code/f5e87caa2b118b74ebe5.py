import time
import random
import itertools

def _is_regular(adj, subset):
    """
    Check if the induced subgraph on 'subset' is regular.
    adj: list of bitmasks.
    subset: tuple of vertex indices.
    """
    k = len(subset)
    if k < 2:
        return True
    
    mask = 0
    for v in subset:
        mask |= (1 << v)
        
    first_deg = -1
    for u in subset:
        deg = (adj[u] & mask).bit_count()
        if first_deg == -1:
            first_deg = deg
        elif deg != first_deg:
            return False
    return True

def _find_bad_subset(adj, n):
    """
    Find one induced regular subgraph of size >= 6.
    Returns the subset as a tuple, or None if none found.
    """
    # Check sizes from 6 upwards
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular(adj, subset):
                return subset
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Try to find a valid graph for n=21 first, then n=20
    for target_n in [21, 20]:
        if time.time() > deadline:
            break
            
        n = target_n
        # Allow more time for n=20 as it's more likely to succeed
        attempts = 100 if n == 20 else 5
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            max_steps = 3000 if n == 20 else 1000
            
            found_valid = False
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                    
                # Find a bad subset
                bad_sub = _find_bad_subset(adj, n)
                
                if bad_sub is None:
                    # No bad subset found. Graph is valid.
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    found_valid = True
                    break
                
                # Found a bad subset
                # Pick a random edge within the subset to flip
                subset_list = list(bad_sub)
                u_idx, v_idx = random.sample(range(len(subset_list)), 2)
                u = subset_list[u_idx]
                v = subset_list[v_idx]
                
                # Flip the edge
                if (adj[u] >> v) & 1:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            if found_valid:
                break
            if time.time() > deadline:
                break

    # Fallback if no valid graph found
    if best_n == 0:
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
