import math
import time

def construct():
    N_MIN = 10000
    N_MAX = 1000000
    
    # Deadline guard
    start_time = time.time()
    DEADLINE = 85.0  # 5s safety margin
    
    # Sieve for Smallest Prime Factor (SPF)
    max_val = N_MAX + 1
    spf = list(range(max_val + 1))
    
    # Standard SPF sieve
    limit = int(math.isqrt(max_val)) + 1
    for i in range(2, limit):
        if spf[i] == i:  # i is prime
            # Mark multiples of i
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Precompute largest prime factor for all numbers up to max_val
    # lpf[x] = largest prime factor of x
    lpf = [0] * (max_val + 1)
    lpf[1] = 1
    for i in range(2, max_val + 1):
        # The largest prime factor of x is the largest prime factor of x/spf[x] 
        # OR spf[x] if x/spf[x] is 1? 
        # Actually, lpf[x] = max(spf[x], lpf[x // spf[x]])
        # But since spf[x] is the SMALLEST prime factor, 
        # the largest prime factor of x is the largest prime factor of x // spf[x], 
        # UNLESS x is prime, in which case it is x.
        # Wait, if x = p * k, where p is smallest prime factor.
        # Then lpf(x) = max(p, lpf(k)).
        # This recurrence works.
        p = spf[i]
        if i // p == 1:
            lpf[i] = p
        else:
            lpf[i] = max(p, lpf[i // p])
            
    best_n = N_MIN
    best_val = float('inf')
    
    # Search the entire valid range
    for n in range(N_MIN, N_MAX + 1):
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
            
        val = max(lpf[n], lpf[n + 1])
        
        if val < best_val:
            best_val = val
            best_n = n
            
    return best_n
