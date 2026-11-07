"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    # Determine the maximum gap to know which primes to precompute
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    
    # Generate primes up to max_gap - 1
    # We need primes p such that p < gap_size
    # If gap_size is G, we check primes p < G.
    # So we need primes up to max_gap - 1.
    limit = max_gap
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
                
    # List of primes less than max_gap
    # Actually, for a specific gap G, we need primes p < G.
    # So we just keep all primes up to max_gap and filter by p < gap_size later.
    primes = [i for i, val in enumerate(is_prime) if val and i < max_gap]
    
    result = []
    for lo, hi in P['gaps']:
        gap_size = hi - lo
        # The candidates are integers m such that lo < m < hi
        num_candidates = gap_size - 2
        if num_candidates <= 0:
            result.append(None)
            continue
            
        # valid[i] corresponds to candidate m = lo + 1 + i
        valid = [True] * num_candidates
        
        # Sieve out multiples of primes p < gap_size
        for p in primes:
            if p >= gap_size:
                break
            # Find the first multiple of p in [lo + 1, hi - 1]
            # lo + 1 is the start of the range
            start = lo + 1
            # First multiple of p >= start
            rem = start % p
            if rem == 0:
                first_mult = start
            else:
                first_mult = start + (p - rem)
                
            # Mark all multiples of p in the range as invalid
            # The index in 'valid' for a number m is m - (lo + 1)
            # So for m = first_mult, index = first_mult - lo - 1
            idx = first_mult - lo - 1
            while idx < num_candidates:
                valid[idx] = False
                idx += p
                
        # Find the first valid candidate
        witness = None
        for i in range(num_candidates):
            if valid[i]:
                witness = lo + 1 + i
                break
                
        result.append(witness)
        
    return result
# EVOLVE-BLOCK-END
