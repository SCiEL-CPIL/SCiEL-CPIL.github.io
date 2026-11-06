import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit, leaving 10s buffer for 60s total
    
    N = 1000001
    min_n = 10000
    max_n = 1000000
    
    # Initialize array for largest prime factor
    # lpf[i] will store the largest prime factor of i
    lpf = np.zeros(N + 1, dtype=np.int32)
    
    # Sieve to compute largest prime factors
    # We iterate through all numbers, if it's prime, we mark its multiples
    # Actually, standard sieve for LPF:
    # For i from 2 to N:
    #   if lpf[i] == 0: # i is prime
    #      for j from i to N step i:
    #          lpf[j] = i
    # This works because we process primes in increasing order, so the last assignment is the largest prime factor.
    
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is a prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # Check deadline
    if time.time() > deadline:
        # Fallback to a known good value or just return min_n if computation is too slow
        # But sieve for 1e6 is ~0.1s, so this shouldn't happen.
        return min_n

    # Compute M(n) = max(lpf[n], lpf[n+1]) for n in [min_n, max_n]
    # n ranges from 10000 to 1000000
    # lpf[n] is lpf[min_n : max_n + 1]
    # lpf[n+1] is lpf[min_n + 1 : max_n + 2]
    
    # Ensure we don't go out of bounds. N=1000001, so lpf has indices 0..1000001.
    # max_n = 1000000. lpf[max_n+1] is lpf[1000001], which is valid.
    
    left = lpf[min_n:max_n + 1]
    right = lpf[min_n + 1:max_n + 2]
    
    # Compute max for each pair
    max_factors = np.maximum(left, right)
    
    # Find index of minimum value
    min_idx = np.argmin(max_factors)
    
    # The actual n is min_n + min_idx
    best_n = min_n + min_idx
    
    # Optional: Verify and return
    # The problem asks to return integer n.
    return int(best_n)
