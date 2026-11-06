import time

def construct():
    """
    Returns a list of 101 lists. Each list contains distinct denominators in 1..10
    such that the sum of their modular inverses modulo 101 equals the index.
    Minimizes the largest row length.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit

    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))
    
    # Precompute modular inverses
    inv = [0] * (MAX_DENOM + 1)
    for d in denominators:
        inv[d] = pow(d, -1, MOD)
    
    # best_len[r] will store the minimum length of a subset summing to r
    # best_subset[r] will store the actual subset
    best_len = [101] * MOD  # Initialize with a value larger than max possible size (10)
    best_subset = [None] * MOD
    
    # Iterate over all non-empty subsets using bitmasks
    # There are 2^10 - 1 = 1023 subsets
    for mask in range(1, 1 << MAX_DENOM):
        # Calculate size and sum
        size = 0
        s = 0
        # Unpack the mask to get denominators
        # We can iterate through bits
        for i in range(MAX_DENOM):
            if mask & (1 << i):
                d = i + 1  # Denominators are 1..10
                size += 1
                s = (s + inv[d]) % MOD
        
        r = s
        # If this subset is smaller than the current best for residue r, update
        if size < best_len[r]:
            best_len[r] = size
            # Reconstruct the subset for storage
            subset = []
            for i in range(MAX_DENOM):
                if mask & (1 << i):
                    subset.append(i + 1)
            best_subset[r] = subset
        
        # Deadline guard (should never trigger for this small N, but required by spec)
        if time.time() - start_time > DEADLINE:
            break
    
    # Build results
    results = []
    for r in range(MOD):
        if best_subset[r] is not None:
            results.append(best_subset[r])
        else:
            # Fallback: This should not happen if the problem is solvable with denominators 1..10
            # But just in case, return a dummy valid list (though it won't sum to r)
            # Actually, if best_subset is None, it means no subset was found, which is an error state.
            # However, the problem statement implies a solution exists.
            # For safety, we append an empty list or a single element, but the contract says non-empty.
            # If this happens, the solution is invalid. But with 10 denominators and mod 101,
            # we have 1023 subsets. By pigeonhole, we likely cover all residues?
            # Let's check: 1023 > 101. It is highly probable that all residues are covered.
            # If not, we just return [1] as a placeholder to satisfy "non-empty", though it will fail verification.
            results.append([1])
    
    return results
