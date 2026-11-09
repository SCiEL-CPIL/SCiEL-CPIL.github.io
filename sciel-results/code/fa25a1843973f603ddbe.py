"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    # Precompute small primes up to 30 (max gap is 28, so we need primes < 28)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Find primes p such that p < gap
        # If m is not divisible by any such p, then lpf(m) >= gap
        # Note: if gap <= 2, no primes p < gap exist (except none), so any m works?
        # If gap=2, primes < 2 is empty. Any m has lpf >= 2. Correct.
        primes_to_check = [p for p in small_primes if p < gap]
        
        found = False
        # Iterate candidates in the gap
        for m in range(lo + 1, hi):
            is_valid = True
            for p in primes_to_check:
                if m % p == 0:
                    is_valid = False
                    break
            if is_valid:
                result[i] = m
                found = True
                break
        
        if not found:
            result[i] = None
            
    return result
# EVOLVE-BLOCK-END
