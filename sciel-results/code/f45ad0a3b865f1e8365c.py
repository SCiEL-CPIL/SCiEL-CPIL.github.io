"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s deadline, leaving 40s margin
    
    gaps = P['gaps']
    n = len(gaps)
    max_prime = max(hi for lo, hi in gaps)
    
    # Sieve for least prime factor up to max_prime
    # We need LPF for numbers in the range [lo+1, hi-1] for each gap
    # Max value we need LPF for is max_prime - 1
    lp_max = max_prime
    lp = list(range(lp_max + 1))
    lp[0] = 0
    if lp_max >= 1:
        lp[1] = 1
    for i in range(2, int(lp_max**0.5) + 1):
        if lp[i] == i:  # i is prime
            for j in range(i*i, lp_max + 1, i):
                if lp[j] == j:
                    lp[j] = i
    
    result = [None] * n
    
    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Find the first m in (lo, hi) such that lp[m] >= gap
        # lp[m] is the least prime factor of m, or m itself if m is prime
        for m in range(lo + 1, hi):
            if lp[m] >= gap:
                result[i] = m
                break
    
    return result
# EVOLVE-BLOCK-END
