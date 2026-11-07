"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    import time
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Determine the maximum value we need to check
    max_val = max(hi for lo, hi in P['gaps'])
    
    # Precompute least prime factor (LPF) via sieve up to max_val
    # lpf[n] = smallest prime factor of n, or n itself if n is prime
    lpf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i*i, max_val + 1, i):
                if lpf[j] == j:
                    lpf[j] = i
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # We need m strictly between lo and hi, with lpf[m] >= gap
        # Check from lo+1 to hi-1
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                break
        # Check deadline
        if time.time() > deadline:
            break
    
    # Ensure all entries are Python int or None
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))
    
    return final_result
# EVOLVE-BLOCK-END
