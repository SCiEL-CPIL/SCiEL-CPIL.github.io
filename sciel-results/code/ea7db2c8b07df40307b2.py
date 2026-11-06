import time
import math
import numpy as np

def _build_spf(limit):
    """Build smallest prime factor sieve up to limit."""
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _compute_powerful_parts(limit, spf):
    """Compute powerful part for all numbers from 0 to limit."""
    pp = [1] * (limit + 1)
    for n in range(2, limit + 1):
        temp = n
        val = 1
        while temp > 1:
            p = spf[temp]
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            if e >= 2:
                val *= p ** e
        pp[n] = val
    return pp

def construct():
    start_time = time.time()
    deadline = 55  # safety margin before 60s
    
    LO = 10000
    HI = 1000000
    
    # Build SPF up to HI
    spf = _build_spf(HI)
    
    # Check time
    if time.time() - start_time > deadline:
        return LO
    
    # Compute powerful parts for all numbers in [LO, HI]
    pp = _compute_powerful_parts(HI, spf)
    
    if time.time() - start_time > deadline:
        return LO
    
    # Convert to numpy array for fast vectorized multiplication and max
    # pp values are <= 10^6, so product <= 10^12, fits in int64
    pp_arr = np.array(pp[LO:HI+1], dtype=np.int64)
    
    # Calculate products pp[n] * pp[n+1] for n in [LO, HI-1]
    # pp_arr[0] corresponds to pp[LO], pp_arr[1] to pp[LO+1], etc.
    # We want pp[n] * pp[n+1]
    products = pp_arr[:-1] * pp_arr[1:]
    
    # Find index of maximum product
    best_idx = np.argmax(products)
    best_n = LO + best_idx
    
    return best_n
