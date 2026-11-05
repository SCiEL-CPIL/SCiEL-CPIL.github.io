"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    N = 1000000
    phi = list(range(N + 1))
    
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
    
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
    
    return N  # fallback (shouldn't reach here)
# EVOLVE-BLOCK-END
