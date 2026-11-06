import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    LO = 10000
    HI = 1000000
    MAX_VAL = HI + 1
    
    # Sieve of Eratosthenes to identify primes
    is_prime = np.ones(MAX_VAL + 1, dtype=bool)
    is_prime[0:2] = False
    
    for i in range(2, int(np.sqrt(MAX_VAL)) + 1):
        if is_prime[i]:
            is_prime[i*i:MAX_VAL+1:i] = False
            
    primes = np.where(is_prime)[0]
    
    # Compute Largest Prime Factor (LPF) for all numbers
    # The last prime to divide a number in increasing order is its largest prime factor
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)
    
    for p in primes:
        lpf[p::p] = p
    
    if time.time() > deadline:
        return LO
    
    # Find n in [LO, HI] minimizing max(LPF(n), LPF(n+1))
    lpf_n = lpf[LO:HI+1]
    lpf_n1 = lpf[LO+1:HI+2]
    
    costs = np.maximum(lpf_n, lpf_n1)
    min_idx = np.argmin(costs)
    best_n = LO + min_idx
    best_val = int(costs[min_idx])
    
    # Verification: spot-check the best candidate and nearby candidates
    # to ensure LPF computation is correct
    def trial_lpf(x):
        """Compute largest prime factor via trial division for verification."""
        if x <= 1:
            return x
        lpf_val = 1
        d = 2
        while d * d <= x:
            while x % d == 0:
                lpf_val = d
                x //= d
            d += 1
        if x > 1:
            lpf_val = max(lpf_val, x)
        return lpf_val
    
    # Verify the best candidate
    verify_n = int(best_n)
    verify_cost = max(trial_lpf(verify_n), trial_lpf(verify_n + 1))
    
    # If verification fails, do a broader search around the best region
    if verify_cost != best_val:
        # Search a window around the best candidate for safety
        search_lo = max(LO, verify_n - 1000)
        search_hi = min(HI, verify_n + 1000)
        
        best_n = search_lo
        best_val = float('inf')
        
        for n in range(search_lo, search_hi + 1):
            if time.time() > deadline:
                break
            c = max(trial_lpf(n), trial_lpf(n + 1))
            if c < best_val:
                best_val = c
                best_n = n
        return int(best_n)
    
    return int(best_n)
