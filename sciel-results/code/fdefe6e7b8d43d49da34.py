"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    max_val = 1000001  # We need lpf for n+1 where n goes up to 1000000
    
    # Step 1: Build Smallest Prime Factor (SPF) sieve
    spf = list(range(max_val + 1))
    limit = int(math.isqrt(max_val)) + 1
    for i in range(2, limit):
        if spf[i] == i:  # i is prime
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Step 2: Precompute largest prime factor for all numbers using dynamic programming
    lpf = [0] * (max_val + 1)
    lpf[1] = 1
    for i in range(2, max_val + 1):
        p = spf[i]
        q = i // p
        while q % p == 0:
            q //= p
        if q == 1:
            lpf[i] = p
        else:
            lpf[i] = max(p, lpf[q])
    
    # Step 3: Search for best n in [10000, 1000000]
    best_n = 10000
    best_score = float('inf')
    
    # Check all candidates using precomputed lpf array
    for n in range(10000, 1000001):
        if time.time() > deadline:
            break
        
        score = max(lpf[n], lpf[n + 1])
        
        if score < best_score:
            best_score = score
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
