def _generate_random_k_regular(n, k, seed=None):
    if (n * k) % 2 != 0:
        return None
        
    # Use numpy for speed
    # Stubs: [0]*k + [1]*k + ...
    stubs = np.repeat(np.arange(n), k)
    np.random.shuffle(stubs)
    
    # Pair
    edges = []
    edge_set = set()
    
    # Vectorized pairing?
    # stubs[0], stubs[1] -> edge
    # stubs[2], stubs[3] -> edge
    u = stubs[0::2]
    v = stubs[1::2]
    
    # Filter loops
    mask = u != v
    if not np.all(mask):
        return None
        
    u = u[mask]
    v = v[mask]
    
    # Filter multi-edges
    # Sort edges and check for duplicates
    # Canonical form: min(u,v), max(u,v)
    min_uv = np.minimum(u, v)
    max_uv = np.maximum(u, v)
    
    # Combine into a single array for uniqueness check
    # Since n < 2^32, we can pack into int64
    # key = min_uv * n + max_uv
    keys = min_uv * n + max_uv
    
    # Check unique
    if len(np.unique(keys)) != len(keys):
        return None
        
    # Convert back to list of tuples
    for i in range(len(u)):
        edges.append((int(min_uv[i]), int(max_uv[i])))
        
    return edges
