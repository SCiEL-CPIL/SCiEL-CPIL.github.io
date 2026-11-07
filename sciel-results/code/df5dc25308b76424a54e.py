"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    # Compute least prime factor (LPF) for all integers in the relevant range
    # Range needed: from min(lo)+1 to max(hi)-1 across all gaps
    # Min lo = 10007, max hi = 10289
    # So we need LPF for numbers 10008 through 10288
    
    N = 10289  # one past the max hi
    lpf = list(range(N + 1))  # lpf[i] will hold the least prime factor of i
    lpf[0] = 0
    lpf[1] = 1
    
    # Sieve to compute least prime factor
    for i in range(2, int(N**0.5) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if lpf[j] == j:  # not yet marked
                    lpf[j] = i
    
    # Now lpf[m] gives the least prime factor of m for all m in our range
    # For primes, lpf[p] = p itself
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        
        gap = hi - lo
        
        # Find any m with lo < m < hi and lpf[m] >= gap
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                break
    
    # Ensure all entries are proper Python ints or None
    final_result = []
    for entry in result:
        if entry is None:
            final_result.append(None)
        else:
            final_result.append(int(entry))
    
    return final_result

# EVOLVE-BLOCK-END
