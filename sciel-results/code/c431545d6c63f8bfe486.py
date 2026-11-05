import math
import time

def construct():
    N = 123457
    deadline = time.time() + 75.0  # safety margin before 90s kill
    
    # Precompute largest prime factor for all numbers 1..N using a sieve
    lp = [0] * (N + 1)
    lp[1] = 1
    
    for i in range(2, N + 1):
        if lp[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lp[j] = i
    
    # Now lp[n] is the largest prime factor of n for all n >= 2
    # lp[1] = 1
    
    # Scan all decompositions to find the optimal one
    best_max = N
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        if time.time() > deadline:
            break
        b = N - a
        pa = lp[a] if a >= 2 else 1
        pb = lp[b] if b >= 2 else 1
        m = pa if pa > pb else pb
        if m < best_max:
            best_max = m
            best_a = a
            best_b = b
        # Early termination: if best_max == 2, that's the absolute minimum possible
        if best_max == 2:
            break
    
    return [best_a, best_b]
