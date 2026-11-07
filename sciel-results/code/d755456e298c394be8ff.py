"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    result = [None] * len(P['gaps'])
    
    # Determine the maximum value we need to check
    max_val = max(hi for _, hi in P['gaps'])
    
    # Precompute the smallest prime factor (SPF) for all numbers up to max_val
    # spf[x] will store the smallest prime factor of x.
    # If x is prime, spf[x] = x.
    spf = list(range(max_val + 1))
    
    # Sieve to compute SPFs
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i: # i is prime
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j: # j hasn't been processed by a smaller prime
                    spf[j] = i
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # We need m in (lo, hi) such that least prime factor of m >= gap
        for m in range(lo + 1, hi):
            lpf = spf[m]
            if lpf >= gap:
                result[i] = m
                break
                
    return result
# EVOLVE-BLOCK-END
