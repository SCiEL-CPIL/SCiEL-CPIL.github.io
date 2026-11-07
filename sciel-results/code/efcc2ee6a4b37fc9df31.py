"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    from math import isqrt
    
    # Determine the range we need to cover
    # We need LPF for all integers in [min_lo+1, max_hi-1]
    # min_lo = 10007, max_hi = 10289
    # So we need LPF for [10008, 10288]
    
    min_val = min(lo for lo, hi in P['gaps']) + 1  # 10008
    max_val = max(hi for lo, hi in P['gaps']) - 1  # 10288
    
    # We'll compute LPF for all numbers in [2, max_val]
    # Using a segmented-like approach: just do a simple sieve for LPF up to max_val
    # Since max_val is ~10289, this is very fast
    
    n = max_val
    # lp[i] will store the least prime factor of i
    lp = list(range(n + 1))  # lp[i] = i initially (meaning prime)
    
    for i in range(2, isqrt(n) + 1):
        if lp[i] == i:  # i is prime
            for j in range(i * i, n + 1, i):
                if lp[j] == j:
                    lp[j] = i
    
    # Now for each gap, find a witness m with lo < m < hi and lp[m] >= gap
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Scan from lo+1 to hi-1
        for m in range(lo + 1, hi):
            # lp[m] is the least prime factor of m
            # We need lp[m] >= gap
            # Note: if m is prime, lp[m] = m, and m > lo >= 2, so m >= gap? 
            # Actually m > lo and m < hi, so m < hi = lo + gap, so m could be less than gap
            # But if m is prime, lp[m] = m. We need m >= gap.
            # Since m < hi = lo + gap and m > lo, we have m < lo + gap.
            # gap = hi - lo. So m < lo + gap.
            # For lp[m] = m >= gap, we need m >= gap.
            # This is possible when lo + 1 >= gap, i.e., lo + 1 >= hi - lo, i.e., 2*lo + 1 >= hi.
            # Anyway, just check lp[m] >= gap directly.
            if lp[m] >= gap:
                result[i] = m
                break
    
    # Ensure all entries are Python int or None
    # The values in result are already Python ints from range()
    
    return result
# EVOLVE-BLOCK-END
