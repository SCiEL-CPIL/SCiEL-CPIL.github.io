"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin for 90s limit
    
    # Search bounds
    N_MIN = 10000
    N_MAX = 1000000
    
    # Precompute smallest prime factor (SPF) up to N_MAX + 1
    # This allows O(log n) factorization for largest prime factor queries
    max_n = N_MAX + 1
    spf = list(range(max_n + 1))
    for i in range(2, int(max_n**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute largest prime factor for all numbers up to max_n using SPF
    # lpf[n] = largest prime factor of n
    lpf = [1] * (max_n + 1)
    for i in range(2, max_n + 1):
        # Use SPF to factorize and find largest prime factor
        n = i
        lp = 1
        while n > 1:
            p = spf[n]
            lp = p
            while n % p == 0:
                n //= p
        lpf[i] = lp
    
    # Now find n in [N_MIN, N_MAX] that minimizes max(lpf[n], lpf[n+1])
    best_n = N_MIN
    best_val = max(lpf[N_MIN], lpf[N_MIN + 1])
    
    for n in range(N_MIN, N_MAX + 1):
        # Time guard
        if time.time() > deadline - 1.0:
            break
        
        current_val = max(lpf[n], lpf[n + 1])
        
        if current_val < best_val:
            best_val = current_val
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
