"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    # Determine the maximum value needed for the sieve
    max_val = max(hi for _, hi in P['gaps'])
    
    # Sieve for Least Prime Factor (LPF)
    lpf = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime
            lpf[i] = i
            # Mark multiples of i
            if i * i <= max_val:
                for j in range(i * i, max_val + 1, i):
                    if lpf[j] == 0:
                        lpf[j] = i
    
    # Precompute for each possible gap size, a boolean array where
    # valid[m] = True means lpf[m] >= gap_size
    # We'll compute this on the fly per gap using the "mark multiples" approach
    
    result = [None] * len(P['gaps'])
    
    # Collect all unique gap sizes for potential precomputation
    gap_sizes = sorted(set(hi - lo for lo, hi in P['gaps']))
    
    # For each unique gap size, precompute a boolean array marking
    # numbers whose LPF < gap_size (i.e., invalid witnesses)
    # Then a valid witness is simply an unmarked number in (lo, hi)
    
    # Precompute invalid marks for each gap size
    # invalid_for_gap[g] = set/array of numbers with lpf < g
    # Actually, let's use a different approach:
    # For a given gap size g, a number m is valid if lpf[m] >= g
    # This means m is NOT divisible by any prime < g
    # So we can precompute for each gap size which numbers are "bad"
    
    # Better: precompute a 2D structure or just compute per gap
    # Since gaps are small (max ~18), let's precompute per gap size
    
    # For each unique gap size, build a boolean array bad[m] = True if lpf[m] < gap_size
    # Then scan (lo, hi) for first m where bad[m] is False
    
    # Precompute bad arrays for each unique gap size
    bad_cache = {}
    for g in gap_sizes:
        # bad[m] = True if m has a prime factor < g
        # i.e., lpf[m] < g
        bad = [False] * (max_val + 1)
        # Mark all multiples of primes < g
        for i in range(2, g):
            if lpf[i] == i:  # i is prime
                for j in range(i, max_val + 1, i):
                    bad[j] = True
        bad_cache[g] = bad
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap_size = hi - lo
        bad = bad_cache[gap_size]
        # Search for a witness m in (lo, hi) where bad[m] is False
        for m in range(lo + 1, hi):
            if not bad[m]:
                result[i] = m
                break
    
    return result
# EVOLVE-BLOCK-END
