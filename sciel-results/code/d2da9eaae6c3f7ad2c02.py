"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    # Determine the range of numbers to cover
    min_val = min(lo for lo, hi in P['gaps'])
    max_val = max(hi for lo, hi in P['gaps'])
    
    # We need LPF for numbers in the intervals (lo, hi).
    # The smallest number to check is min_val + 1.
    # The largest number to check is max_val - 1.
    # Let's create an array lpf where lpf[i] is the least prime factor of i.
    # We need this for indices up to max_val.
    
    size = max_val + 1
    lpf = [0] * size
    
    # Sieve for Least Prime Factor
    # lpf[i] = i if i is prime, else smallest prime factor.
    # Initialize lpf[i] = i for all i >= 2? Or just use 0 to mean "unvisited/prime".
    # Standard LPF sieve:
    for i in range(2, size):
        if lpf[i] == 0: # i is prime
            lpf[i] = i
            # Mark multiples
            if i * i < size:
                for j in range(i * i, size, i):
                    if lpf[j] == 0:
                        lpf[j] = i
                    # Optimization: only set if not already set? 
                    # Actually, we want the LEAST prime factor.
                    # Since we iterate i in increasing order, the first i that divides j is the least.
                    # So we should only set if lpf[j] is 0.
                    
    # Wait, the standard simple sieve above works if we only set when 0.
    # But if we start from i*i, we might miss smaller factors?
    # No, if j has a factor p < i, then j would have been marked by p earlier.
    # So lpf[j] would not be 0.
    # So the logic holds.

    result = [None] * len(P['gaps'])
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Find first m in (lo, hi) such that lpf[m] >= gap
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                break
    
    return result
# EVOLVE-BLOCK-END
