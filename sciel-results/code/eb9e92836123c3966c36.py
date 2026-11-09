"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a sieve to compute phi(n) for all n up to 1,000,000, then finds the
    smallest n satisfying the condition.
    """
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin for safety
    
    N = 1000000
    
    # Sieve for phi
    phi = list(range(N + 1))
    
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
    
    # Find smallest n with phi(n) % 120 == 0
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
    
    # Fallback (should not reach here)
    return 14400
# EVOLVE-BLOCK-END
