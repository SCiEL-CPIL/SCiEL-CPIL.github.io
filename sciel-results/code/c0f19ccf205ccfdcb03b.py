import time

def construct():
    # Pairs from the task description
    P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}
    
    start_time = time.time()
    deadline = start_time + 50

    # Generate primes up to a reasonable limit (max gap is 28, primes < 28 are small)
    # We need primes < gap. Max gap is 28. Primes < 28: 2, 3, 5, 7, 11, 13, 17, 19, 23.
    # We can just hardcode or generate.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Find primes strictly less than gap
        small_primes = [p for p in primes if p < gap]
        
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            is_rough = True
            for p in small_primes:
                if m % p == 0:
                    is_rough = False
                    break
            if is_rough:
                result[i] = int(m)
                break
                
    return result
