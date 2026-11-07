"""Rewritten program with optimized witness search."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin within 60s limit
    
    # Determine the range of values we need to check
    min_val = min(lo for lo, _ in P['gaps']) + 1  # smallest possible witness
    max_val = max(hi for _, hi in P['gaps']) - 1  # largest possible witness
    
    # Compute least prime factor (LPF) for all integers in [min_val, max_val]
    # using a segmented sieve approach
    n = max_val - min_val + 1
    lpf = [0] * n  # 0 means "not yet assigned" (will be the number itself if prime)
    
    # Sieve up to sqrt(max_val)
    sieve_limit = int(math.isqrt(max_val)) + 1
    is_prime = [True] * (sieve_limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(sieve_limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, sieve_limit + 1, i):
                is_prime[j] = False
    
    small_primes = [i for i in range(2, sieve_limit + 1) if is_prime[i]]
    
    # For each small prime, mark its multiples in the range [min_val, max_val]
    for p in small_primes:
        # Find the first multiple of p in [min_val, max_val]
        first = ((min_val + p - 1) // p) * p
        if first < p * p:
            first = p * p  # We only need to start from p*p for LPF correctness
        # Actually for LPF we want the SMALLEST prime factor, so we should start from the first multiple
        # But we need to be careful: if first < p, skip (but first >= min_val >= 10008 > p for all our primes)
        first = ((min_val + p - 1) // p) * p
        for m in range(first, max_val + 1, p):
            idx = m - min_val
            if lpf[idx] == 0:
                lpf[idx] = p
    
    # For numbers where lpf is still 0, they are prime (or 1), so LPF = the number itself
    for i in range(n):
        if lpf[i] == 0:
            lpf[i] = min_val + i  # prime, LPF is itself
    
    result = []
    
    for lo, hi in P['gaps']:
        if time.time() > deadline:
            result.append(None)
            continue
        
        gap = hi - lo
        
        # Find any m in (lo, hi) such that lpf[m] >= gap
        witness = None
        for m in range(lo + 1, hi):
            idx = m - min_val
            if lpf[idx] >= gap:
                witness = m
                break
        
        result.append(witness)
    
    # Ensure types are correct
    final_result = []
    for entry in result:
        if entry is None:
            final_result.append(None)
        else:
            final_result.append(int(entry))
    
    return final_result
# EVOLVE-BLOCK-END
