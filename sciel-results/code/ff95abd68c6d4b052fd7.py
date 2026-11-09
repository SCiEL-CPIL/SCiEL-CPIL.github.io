"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Find max value to size the sieve
    max_val = 0
    for p, q in P['gaps']:
        if q > max_val:
            max_val = q
    max_val += 10  # Buffer
    
    # Sieve for Least Prime Factor
    # lpf[i] will store the smallest prime factor of i
    lpf = [0] * (max_val + 1)
    for i in range(2, int(max_val**0.5) + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i*i, max_val + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
    
    # Fill in primes for numbers > sqrt(max_val) that are still 0
    # For composite numbers, lpf is already set.
    # For prime numbers, lpf is 0. We set it to the number itself.
    # For 1, lpf is 0, but we don't care about 1.
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            lpf[i] = i
            
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Check time
        if time.time() > deadline:
            break
            
        # Search for m in (lo, hi) such that lpf[m] >= gap
        # We iterate from lo+1 to hi-1
        # Any m with no prime factor < gap is a valid witness.
        # Since lpf[m] is the smallest prime factor, lpf[m] >= gap is the condition.
        
        found = False
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                found = True
                break
        
        if not found:
            result[i] = None
            
    # Ensure types are correct (Python int, list)
    # The loop assigns Python ints or None.
    return result
# EVOLVE-BLOCK-END
