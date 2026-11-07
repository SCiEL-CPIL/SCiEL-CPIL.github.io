import time

def construct():
    # Pinned data
    P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}
    
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    # Determine max value for sieve
    max_val = max(hi for lo, hi in P['gaps'])
    
    # Sieve of Eratosthenes to find Least Prime Factor (LPF)
    # lpf[n] will store the smallest prime factor of n
    # If n is prime, lpf[n] = n
    lpf = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, max_val + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
                    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Find the first m in (lo, hi) such that lpf[m] >= gap
        # Note: For primes m, lpf[m] = m. Since m > 10000 and gap < 100, m >= gap is always true.
        # So any prime in the gap is a valid witness.
        # For composites, we check the condition explicitly.
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                break
                
    return result
