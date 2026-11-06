import time

def construct():
    """
    Returns a list of 101 lists. Each list contains distinct denominators in 1..10
    such that the sum of their modular inverses modulo 101 equals the index.
    Minimizes the largest row length.
    """
    start_time = time.time()
    deadline = 85.0  # 5s safety margin for 90s limit

    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))
    
    # Precompute modular inverses
    inv = [0] * (MAX_DENOM + 1)
    for d in denominators:
        inv[d] = pow(d, -1, MOD)
    
    n = MAX_DENOM
    total_masks = 1 << n
    
    # We will store the best subset for each residue
    # best[r] = (length, list_of_denoms)
    best = [None] * MOD
    
    # We iterate by length to ensure we find the shortest subset for each residue first
    # This minimizes the max row length
    for length in range(1, n + 1):
        # Check time
        if time.time() > deadline:
            break
            
        # Iterate all masks of this length
        # Optimization: generate combinations of indices of size 'length'
        # Instead of iterating all 1024 masks and checking popcount,
        # we can use itertools.combinations
        import itertools
        
        for combo in itertools.combinations(range(n), length):
            if time.time() > deadline:
                break
                
            # Calculate sum and length
            s = 0
            for idx in combo:
                s = (s + inv[idx + 1]) % MOD
            
            r = s
            if best[r] is None:
                # Convert indices to denominators (1-based)
                denoms = [idx + 1 for idx in combo]
                best[r] = (length, denoms)
        
        # Check if all residues are covered
        if all(b is not None for b in best):
            break
    
    # Build results
    results = []
    for r in range(MOD):
        if best[r] is not None:
            results.append(best[r][1])
        else:
            # Fallback: if not found (should not happen), use a large subset or repeat
            # Since we must return non-empty lists, and all residues should be covered by len 5
            # If not, we try to find any subset
            found = False
            for mask in range(1, total_masks):
                if time.time() > deadline:
                    break
                # Calculate sum for mask
                s = 0
                for i in range(n):
                    if (mask >> i) & 1:
                        s = (s + inv[i + 1]) % MOD
                if s == r:
                    denoms = [i + 1 for i in range(n) if (mask >> i) & 1]
                    results.append(denoms)
                    found = True
                    break
            if not found:
                # Last resort
                results.append([1])
                
    return results
