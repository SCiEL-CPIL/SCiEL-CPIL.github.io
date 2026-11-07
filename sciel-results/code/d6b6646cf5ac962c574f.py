import math
import time

# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    all_lo = min(lo for lo, hi in P['gaps'])
    all_hi = max(hi for lo, hi in P['gaps'])
    
    # We need LPF for integers strictly between the pairs, 
    # so we cover [all_lo + 1, all_hi - 1]
    lo_bound = all_lo + 1
    hi_bound = all_hi - 1
    size = hi_bound - lo_bound + 1
    
    # Segmented sieve to compute least prime factor for [lo_bound, hi_bound]
    # We need primes up to sqrt(hi_bound)
    sqrt_hi = int(math.isqrt(hi_bound)) + 1
    
    # Simple sieve for small primes up to sqrt_hi
    is_prime_small = [True] * (sqrt_hi + 1)
    is_prime_small[0] = False
    if sqrt_hi > 1:
        is_prime_small[1] = False
    for i in range(2, int(math.isqrt(sqrt_hi)) + 1):
        if is_prime_small[i]:
            for j in range(i * i, sqrt_hi + 1, i):
                is_prime_small[j] = False
    small_primes = [i for i, v in enumerate(is_prime_small) if v]
    
    # Initialize LPF array: 0 means "not yet assigned" (likely prime or not hit by small primes yet)
    lpf = [0] * size
    
    # Mark multiples of small primes
    for p in small_primes:
        # Find the first multiple of p in [lo_bound, hi_bound] that is >= p*p
        # Actually, we just need the first multiple in the range.
        # If the multiple is p itself, it's prime, but p < lo_bound usually?
        # lo_bound is ~10008, sqrt_hi is ~101. So p is always < lo_bound.
        # So the first multiple is just ceil(lo_bound / p) * p.
        first = ((lo_bound + p - 1) // p) * p
        if first > hi_bound:
            continue
            
        for m in range(first, hi_bound + 1, p):
            idx = m - lo_bound
            if lpf[idx] == 0:
                lpf[idx] = p
                
    # For numbers still with lpf=0, they are prime (or 1, but range is large)
    # Their least prime factor is the number itself.
    for i in range(size):
        if lpf[i] == 0:
            lpf[i] = lo_bound + i

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        g = hi - lo
        # We want to find m in (lo, hi) such that lpf(m) >= g.
        # To maximize the probability, we pick the m with the maximum LPF in that interval.
        # If max_lpf >= g, then that m is a valid witness.
        
        start_idx = (lo + 1) - lo_bound
        end_idx = (hi - 1) - lo_bound
        
        if start_idx < 0: start_idx = 0
        if end_idx >= size: end_idx = size - 1
        
        if start_idx > end_idx:
            continue
            
        best_m = None
        best_lpf_val = 0
        
        # Scan the interval to find the maximum LPF
        # Since the range is small (max gap ~28), this is very fast.
        for idx in range(start_idx, end_idx + 1):
            val = lpf[idx]
            if val > best_lpf_val:
                best_lpf_val = val
                best_m = lo_bound + idx
                
        if best_m is not None and best_lpf_val >= g:
            result[i] = best_m

    # Ensure all entries are Python int or None
    final = []
    for v in result:
        if v is None:
            final.append(None)
        else:
            final.append(int(v))

    return final
# EVOLVE-BLOCK-END
