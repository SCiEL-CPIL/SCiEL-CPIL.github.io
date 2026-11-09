"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit

    # Determine max prime for sieve
    max_prime = max(max(p) for p in P['gaps'])
    
    # Build Least Prime Factor (LPF) sieve
    lpf = list(range(max_prime + 1))
    for i in range(2, int(max_prime**0.5) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i*i, max_prime + 1, i):
                if lpf[j] == j:
                    lpf[j] = i

    result = [None] * len(P['gaps'])
    
    # Process each gap
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        found = False
        # Search for m in (lo, hi) such that lpf[m] >= gap
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                found = True
                break
        
        # Note: If not found, result[i] remains None.
        # The problem states "None never asserts nonexistence", 
        # but we must maximize certified gaps. 
        # Since we check all candidates, if we don't find one, 
        # it truly doesn't exist in this small range.
        
    return result
# EVOLVE-BLOCK-END
