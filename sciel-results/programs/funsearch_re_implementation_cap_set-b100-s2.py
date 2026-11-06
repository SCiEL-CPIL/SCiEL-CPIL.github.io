def priority(vector, n):
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    target = n / 3.0
    
    dev = (c0 - target)**2 + (c1 - target)**2 + (c2 - target)**2
    
    # Small deterministic tie-breaker based on vector pattern
    # Use a simple hash that's stable
    h = 0
    for i, v in enumerate(vector):
        h = (h * 31 + v) & 0x7FFFFFFF
    
    # Add a tiny perturbation to break ties
    # Scale: dev is O(n), so perturbation should be << 1
    eps = 1e-9
    return -dev + eps * (h / 0x7FFFFFFF)
