"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    We precompute the largest prime factor (LPF) for all integers up to 1000001
    using a sieve-like method. For each prime p, we iterate through its multiples
    and update lpf[multiple] = p. Since we process primes in increasing order,
    the final value stored is the largest prime factor.
    
    Then we scan n from 10000 to 1000000, computing max(lpf[n], lpf[n+1]) for
    each, and track the minimum.
    """
    N = 1000001  # We need lpf up to 1000001 since n can be 1000000, so n+1=1000001
    
    # Initialize lpf array with 0
    lpf = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for largest prime factor
    for i in range(2, N + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i  # Since we go in increasing order of i, this stores the largest prime factor
    
    # Now scan for the best n
    best_n = 10000
    best_lpf_val = lpf[10000] if lpf[10000] >= lpf[10001] else lpf[10001]
    
    for n in range(10000, N):  # n goes up to 1000000, so n+1 up to 1000001
        # We need n+1 <= N, so n <= N-1 = 1000000
        v1 = lpf[n]
        v2 = lpf[n + 1]
        current_max = v1 if v1 >= v2 else v2
        if current_max < best_lpf_val:
            best_lpf_val = current_max
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
