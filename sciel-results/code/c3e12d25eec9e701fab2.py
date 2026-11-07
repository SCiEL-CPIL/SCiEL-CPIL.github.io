"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_lpf(limit):
    """Generate list where lpf[n] is the least prime factor of n (or 0 if n is prime)."""
    lpf = [0] * (limit + 1)
    for i in range(2, isqrt(limit) + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i * i, limit + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
    return lpf

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 second safety margin
    
    max_val = max(hi for lo, hi in P['gaps'])
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    
    # Sieve for least prime factor up to max_val
    lpf = _sieve_lpf(max_val)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        
        # Find an integer m strictly between lo and hi whose least prime factor >= gap
        # lpf[m] == 0 means m is prime (LPF = m itself, which is >= gap since m > lo >= 10007)
        # lpf[m] >= gap means the least prime factor is at least gap
        found = False
        
        # Scan from the middle outward for balance, or just scan linearly
        # Linear scan is fine since gaps are small (max 20)
        for m in range(lo + 1, hi):
            if lpf[m] == 0 or lpf[m] >= gap:
                result[i] = m
                found = True
                break
        
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
