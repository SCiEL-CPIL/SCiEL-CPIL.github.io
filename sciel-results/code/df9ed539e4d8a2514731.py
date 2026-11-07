"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    import time
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Precompute primes up to ~50000 (max number in our range is ~10289, 
    # but we need primes up to sqrt(10289) ≈ 101 for trial division,
    # and primes up to max gap size for the "small primes" exclusion)
    # Actually, we need to check divisibility by primes < gap_size.
    # Max gap is 28, so we need primes < 28: 2,3,5,7,11,13,17,19,23
    # But also for finding lpf, we need trial division up to sqrt(m)
    
    from math import isqrt
    
    # Generate primes up to 200 (enough for sqrt(10289)~101 and gap sizes)
    def sieve(n):
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_p[i]:
                for j in range(i*i, n + 1, i):
                    is_p[j] = False
        return [i for i in range(2, n + 1) if is_p[i]]
    
    primes = sieve(200)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Find the first m in (lo, hi) with lpf(m) >= gap
        # lpf(m) >= gap means m is not divisible by any prime < gap
        small_primes = [p for p in primes if p < gap]
        
        found = False
        for m in range(lo + 1, hi):
            # Check if m is divisible by any prime < gap
            if any(m % p == 0 for p in small_primes):
                continue
            # m passes: lpf(m) >= gap
            result[i] = m
            found = True
            break
    
    # Ensure all entries are proper Python ints or None
    result = [int(x) if x is not None else None for x in result]
    return result
# EVOLVE-BLOCK-END
