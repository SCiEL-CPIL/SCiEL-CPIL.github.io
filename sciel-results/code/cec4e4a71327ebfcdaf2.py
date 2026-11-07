"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    # Determine the range
    min_val = min(lo for lo, hi in P['gaps']) + 1  # 10008
    max_val = max(hi for lo, hi in P['gaps']) - 1  # 10288
    
    # Generate primes up to sqrt(max_val)
    limit = int(math.isqrt(max_val)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # Segmented sieve for LPF in [min_val, max_val]
    size = max_val - min_val + 1
    # lp_seg[i] stores the least prime factor of min_val + i
    # Initialize to 0. If 0 at the end, the number is prime.
    lp_seg = [0] * size
    
    for p in primes:
        # Find the first multiple of p in [min_val, max_val]
        start = ((min_val + p - 1) // p) * p
        # If start is p itself, and p >= min_val, we need to be careful.
        # But p <= sqrt(max_val) ~ 101, and min_val ~ 10000.
        # So start will be a multiple of p greater than p.
        # Thus, start is composite (unless start == p, which is not possible here).
        for j in range(start, max_val + 1, p):
            idx = j - min_val
            if lp_seg[idx] == 0:
                lp_seg[idx] = p
                
    # Fill in primes for numbers that were not marked (lp_seg[idx] == 0)
    # For a prime number m, LPF(m) = m.
    for i in range(size):
        if lp_seg[i] == 0:
            lp_seg[i] = min_val + i
            
    # Now find witnesses
    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        for m in range(lo + 1, hi):
            idx = m - min_val
            if lp_seg[idx] >= gap:
                result[i] = m
                break
                
    return result
# EVOLVE-BLOCK-END
