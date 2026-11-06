"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    N = 123457
    
    # Use NumPy for fast sieve to compute largest prime factor
    # lp[n] = largest prime factor of n
    lp = np.ones(N + 1, dtype=np.int64)
    lp[0] = 0
    lp[1] = 1
    
    # Sieve: for each prime p (in increasing order), set lp[m] = p for all multiples m
    # Since primes are processed in increasing order, the last write gives the largest prime factor
    for p in range(2, N + 1):
        if lp[p] == 1:  # p is prime
            lp[p::p] = p
    
    # Convert to Python list for fast indexing in the scan
    lp_list = lp.tolist()
    
    # Now scan a from 1 to N-1, b = N - a
    # We want to minimize max(lp[a], lp[b])
    # Since a+b=N, both a and b are in [1, N-1], so lp is defined.
    # Use symmetry: only need to check a from 1 to N//2
    
    best_val = float('inf')
    best_a = 1
    best_b = N - 1
    
    half = N // 2
    for a in range(1, half + 1):
        b = N - a
        va = lp_list[a]
        vb = lp_list[b]
        val = va if va > vb else vb
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
            if best_val == 2:
                # Can't do better than 2 (both even, but sum is odd so impossible)
                # Actually if best_val == 2, both must be powers of 2, sum = 123457 (odd)
                # So at least one is odd, meaning P >= 3. So 2 is impossible.
                # But 3 is possible if both are powers of 2 and 3.
                break
    
    # Check time
    if time.time() > deadline:
        # Shouldn't happen, but safety
        pass
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
