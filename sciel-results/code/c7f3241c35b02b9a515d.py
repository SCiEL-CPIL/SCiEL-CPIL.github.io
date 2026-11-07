"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    result = [None] * len(P['gaps'])
    
    # Determine the maximum gap to know which small primes we need to check
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    
    # Generate all primes strictly less than max_gap
    # We need to check if m has any prime factor p < gap.
    # If it has no prime factor p < gap, then LPF(m) >= gap.
    small_primes = []
    n = 2
    while n < max_gap:
        is_p = True
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                is_p = False
                break
        if is_p:
            small_primes.append(n)
        n += 1
        
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Filter primes that are strictly less than the current gap
        # If p >= gap, it doesn't disqualify m (since we only care about LPF < gap)
        relevant_primes = [p for p in small_primes if p < gap]
        
        found = False
        for m in range(lo + 1, hi):
            # Check if m is divisible by any relevant prime
            # If yes, LPF(m) <= p < gap, so m is invalid.
            # If no, LPF(m) >= gap (or m is prime, in which case LPF(m)=m >= gap since m>lo>=10000>gap).
            valid = True
            for p in relevant_primes:
                if m % p == 0:
                    valid = False
                    break
            
            if valid:
                result[i] = m
                found = True
                break
                
        # If not found, result[i] remains None
        
    return result
# EVOLVE-BLOCK-END
