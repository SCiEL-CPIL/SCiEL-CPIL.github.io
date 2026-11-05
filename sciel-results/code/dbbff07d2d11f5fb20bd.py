    # Search n starting from Moore bound
    # Moore bound for k=5, g=9 is 426.
    # Let's start at 430 and go up.
    
    # If k is generic, calculate Moore bound?
    # r = (g-1)//2
    # mb = 1 + k * sum((k-1)^i for i in range(r))
    
    r = (g - 1) // 2
    mb = 1
    term = 1
    for i in range(r):
        term *= (k - 1)
        mb += k * term
    # Ensure n*k is even
    if (k * mb) % 2 != 0:
        mb += 1
        
    # Start search
    n = mb
    while n <= 1200 and time.time() < deadline:
        # Try more random graphs since girth check is faster
        found = False
        for _ in range(50):
            if time.time() > deadline:
                break
            adj = _pairing_regular(n, k, rng)
            if adj is None:
                continue
            if consider(n, adj):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
        
        # If not found randomly, try local search on one instance
        adj = _pairing_regular(n, k, rng)
        if adj is not None:
            # Try to increase girth
            for _ in range(10000):
                if time.time() > deadline:
                    break
                if _swap_regular(adj, n, k, rng):
                    if consider(n, adj):
                        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
        
        n += 2
