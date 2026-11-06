"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    def time_exceeded():
        return time.time() - start_time >= DEADLINE

    def compute_phi_sieve(N):
        """Compute phi[1..N] using a vectorized sieve approach."""
        phi = np.arange(N + 1, dtype=np.int64)
        # Sieve: for each prime p, multiply phi by (1 - 1/p) for all multiples of p
        # Vectorized: iterate over primes only
        for i in range(2, N + 1):
            if phi[i] == i:  # i is prime
                multiples = np.arange(i, N + 1, i)
                phi[multiples] = phi[multiples] * (i - 1) // i
        return phi

    def find_longest_distinct_run(phi_arr):
        """Find the longest run of consecutive integers with distinct phi values."""
        N = len(phi_arr) - 1  # phi_arr is 0-indexed, valid from 1 to N
        last_seen = {}
        left = 1
        best_k = 1
        best_s = 1
        
        for right in range(1, N + 1):
            v = phi_arr[right]
            if v in last_seen and last_seen[v] >= left:
                left = last_seen[v] + 1
            last_seen[v] = right
            window_len = right - left + 1
            if window_len > best_k:
                best_k = window_len
                best_s = left
        
        return best_s, best_k

    # Start with a moderate range
    N1 = 200000
    phi1 = compute_phi_sieve(N1)
    best_s, best_k = find_longest_distinct_run(phi1)
    
    if time_exceeded():
        return {'start': best_s, 'length': best_k}
    
    # Try progressively larger ranges
    for N2 in [500000, 1000000, 2000000, 5000000, 10000000]:
        if time_remaining() < 20.0:
            break
        try:
            phi2 = compute_phi_sieve(N2)
            s2, k2 = find_longest_distinct_run(phi2)
            if k2 > best_k:
                best_k = k2
                best_s = s2
        except (MemoryError, Exception):
            break
    
    # Ensure constraints are met
    if best_s + best_k - 1 > 10**12:
        best_k = 10**12 - best_s + 1
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
