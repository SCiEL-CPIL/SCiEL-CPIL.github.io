"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

# The pinned consecutive prime pairs
P = {'gaps': [
    [10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061],
    [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091],
    [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111],
    [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151],
    [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177],
    [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223],
    [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259],
    [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]
]}

def construct():
    # Safety margin for execution time
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Find the maximum value to sieve
    max_val = max(hi for lo, hi in P['gaps'])
    
    # Sieve of Eratosthenes variant to find Least Prime Factor (LPF)
    # lpf[n] will store the smallest prime factor of n
    lpf = [0] * (max_val + 1)
    
    # Initialize for all numbers
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime
            lpf[i] = i
            # Mark multiples
            for j in range(i * i, max_val + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
    
    result = [None] * len(P['gaps'])
    
    # For each gap, find a witness
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        
        # Search for m in (lo, hi) such that lpf[m] >= gap
        # This implies m has no prime factor smaller than gap
        found_witness = None
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                found_witness = m
                break
        
        if found_witness is not None:
            # Ensure it is a Python int
            result[i] = int(found_witness)
        else:
            result[i] = None
    
    # Final check to ensure types are correct
    final_result = []
    for v in result:
        if v is None:
            final_result.append(None)
        else:
            # v is already int, but cast to be sure
            final_result.append(int(v))
            
    return final_result

# EVOLVE-BLOCK-END
