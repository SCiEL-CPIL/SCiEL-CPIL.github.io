"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    N_MAX = 1000000
    N_MIN = 10000
    
    # Sieve for largest prime factor up to N_MAX + 1
    # lpf[i] = largest prime factor of i
    lpf = [0] * (N_MAX + 2)
    
    # Mark composites and set largest prime factor
    # For each number i from 2 to N_MAX+1, if lpf[i] == 0, i is prime
    # Then for all multiples of i, set lpf[multiple] = i
    # Since we go in increasing order, the last prime to set lpf[m] is the largest prime factor
    
    for i in range(2, N_MAX + 2):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N_MAX + 2, i):
                lpf[j] = i
    
    # Now find n in [N_MIN, N_MAX] that minimizes max(lpf[n], lpf[n+1])
    best_n = N_MIN
    best_val = max(lpf[N_MIN], lpf[N_MIN + 1])
    
    for n in range(N_MIN, N_MAX + 1):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
        
        # Check time
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
