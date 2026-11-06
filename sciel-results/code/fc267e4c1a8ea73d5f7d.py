"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    lo, hi = 10000, 1000000
    limit = hi + 1  # need to factor up to 1000001
    
    # Build smallest prime factor array
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute powerful parts for all n in [lo, hi+1]
    # Use a sieve-like approach: for each number, compute its powerful part
    pp = [1] * (limit + 1)
    
    # For each prime p, compute p^e for e>=2 and multiply into all multiples
    # But we need the FULL prime power, so we need to track exponents
    
    # Alternative: factor each number using spf, but optimize the inner loop
    for n in range(lo, limit + 1):
        m = n
        result = 1
        while m > 1:
            p = spf[m]
            count = 0
            while m % p == 0:
                m //= p
                count += 1
            if count >= 2:
                result *= p ** count
        pp[n] = result
        if time.time() > deadline:
            # If we ran out of time, we still need to return something valid
            # Use what we have so far
            break
    
    # Find max product among computed values
    best_n = lo
    best_val = 0
    # Determine how far we computed
    max_computed = limit
    for i in range(lo, limit + 1):
        if pp[i] == 0:  # wasn't computed
            max_computed = i - 1
            break
    
    for n in range(lo, min(hi, max_computed)):
        prod = pp[n] * pp[n + 1]
        if prod > best_val:
            best_val = prod
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
