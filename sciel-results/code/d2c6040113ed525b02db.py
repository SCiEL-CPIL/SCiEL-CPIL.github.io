import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds safety margin
    
    P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}
    
    # Precompute least prime factor (LPF) for all numbers in the range
    lo_min = min(lo for lo, hi in P['gaps'])
    hi_max = max(hi for lo, hi in P['gaps'])
    size = hi_max - lo_min + 1
    
    # Sieve to compute LPF
    lpf = [0] * size
    for i in range(lo_min, hi_max + 1):
        lpf[i - lo_min] = i if i >= 2 else 0
    
    # Sieve primes up to hi_max
    is_prime = [True] * (hi_max + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(hi_max**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, hi_max + 1, i):
                is_prime[j] = False
    
    # Compute LPF for each number
    for p in range(2, hi_max + 1):
        if is_prime[p]:
            for m in range(p, hi_max + 1, p):
                if lpf[m - lo_min] == m:  # m is prime or not yet assigned
                    lpf[m - lo_min] = p
                elif lpf[m - lo_min] > p:
                    lpf[m - lo_min] = p
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        # We need a number m in (lo, hi) such that lpf[m] >= gap
        # (if m is prime, lpf[m] = m, which is >= gap since m > lo and gap = hi - lo)
        
        found = False
        # Iterate through candidates in the gap
        for m in range(lo + 1, hi):
            if lpf[m - lo_min] >= gap:
                result[i] = m
                found = True
                break
        
        # If not found, result[i] remains None
    
    return result
