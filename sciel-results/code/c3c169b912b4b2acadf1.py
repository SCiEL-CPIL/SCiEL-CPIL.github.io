"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    N_MAX = 1000000
    N_MIN = 10000
    
    # Precompute largest prime factor for all numbers up to N_MAX
    # lpf[i] will hold the largest prime factor of i
    lpf = list(range(N_MAX + 1))
    
    for i in range(2, int(N_MAX**0.5) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i*i, N_MAX + 1, i):
                lpf[j] = i
    
    # Now lpf[x] gives the largest prime factor of x
    # We want to minimize max(lpf[n], lpf[n+1]) for n in [N_MIN, N_MAX-1]
    # Note: n+1 can be up to N_MAX, so we need lpf up to N_MAX
    
    best_n = N_MIN
    best_val = max(lpf[N_MIN], lpf[N_MIN + 1])
    
    for n in range(N_MIN, N_MAX):
        if time.time() > deadline:
            break
        v = max(lpf[n], lpf[n + 1])
        if v < best_val:
            best_val = v
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
