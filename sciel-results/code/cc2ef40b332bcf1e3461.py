def _has_k6_in_complement(adj, n):
    # Build complement adjacency
    comp = [0] * n
    mask = (1 << n) - 1
    for i in range(n):
        comp[i] = (~adj[i]) & mask & ~(1 << i)
        
    def bronk(R, P, X):
        if not P and not X:
            if len(R) >= 6:
                return True
            return False
        # Pivoting
        # Pick u from P union X with max degree in P
        # For simplicity, just pick first vertex in P if P is not empty
        # Or use a simple heuristic
        if not P:
            return False
            
        # Pick pivot u
        u = (P | X).bit_length() - 1 
        # Better: pick u in P|X with max popcount(comp[u] & P)
        # For speed, just pick first bit in P
        lsb = P & -P
        u = lsb.bit_length() - 1
        
        # Iterate over P \ N(u)
        candidates = P & ~comp[u] & ~(1 << u)
        
        while candidates:
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            candidates ^= lsb
            
            if bronk(R | (1 << v), P & comp[v], X & comp[v]):
                return True
            
            P ^= lsb
            X |= lsb
        return False

    return bronk(0, (1 << n) - 1, 0)
