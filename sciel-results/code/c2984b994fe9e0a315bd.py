"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    from math import isqrt
    
    # Precompute primes up to sqrt(max possible value) for LPF computation
    all_hi = max(hi for lo, hi in P['gaps'])
    limit = isqrt(all_hi) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    small_primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Compute LPF for all relevant m values
    min_val = min(lo for lo, hi in P['gaps'])
    max_val = all_hi
    lpf = {}
    for m in range(min_val + 1, max_val):
        lpf[m] = m  # default: assume prime
        for p in small_primes:
            if p * p > m:
                break
            if m % p == 0:
                lpf[m] = p
                break
    
    # Find witnesses for each gap
    result = []
    for lo, hi in P['gaps']:
        gap = hi - lo
        witness = None
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                witness = m
                break
        result.append(witness)
    
    # Ensure all entries are Python int or None (not numpy types)
    final_result = []
    for entry in result:
        if entry is None:
            final_result.append(None)
        else:
            final_result.append(int(entry))
    
    return final_result
# EVOLVE-BLOCK-END
