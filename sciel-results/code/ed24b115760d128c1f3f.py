import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

# Primes less than the maximum possible gap (which is 20)
# Primes: 2, 3, 5, 7, 11, 13, 17, 19
SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19]

def construct():
    deadline = time.time() + 80
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        
        g = hi - lo
        # Find primes strictly less than g
        # Since g is small, we can just filter SMALL_PRIMES
        relevant_primes = [p for p in SMALL_PRIMES if p < g]
        
        # If g <= 2, relevant_primes is empty.
        # Any m > 1 has lpf(m) >= 2.
        # Since lo, hi are primes > 2, m > 2, so lpf(m) >= 2 >= g.
        # So for g=2, the first m works.
        
        found = False
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            
            is_rough = True
            for p in relevant_primes:
                if m % p == 0:
                    is_rough = False
                    break
            
            if is_rough:
                result[i] = m
                found = True
                break
        
        if not found:
            result[i] = None
            
    return result
