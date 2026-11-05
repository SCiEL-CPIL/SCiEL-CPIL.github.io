import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    L = 10000
    R = 1000000
    max_val = R + 1  # we need lpf up to 1000001
    
    # Sieve for Largest Prime Factor (LPF) using numpy for speed
    # Initialize with 0 to distinguish unvisited (composite not yet processed or 1) from primes
    # Actually, 1 is not prime. We can initialize with 0.
    # If lpf[p] == 0, p is prime (for p >= 2).
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    lpf[1] = 1
    
    # For each prime p, mark all multiples with p
    # Since we iterate p in increasing order, the last (largest) prime to divide m wins
    for p in range(2, max_val + 1):
        if lpf[p] == 0:  # p is prime
            lpf[p::p] = p
    
    # Now find n in [L, R] that minimizes max(lpf[n], lpf[n+1])
    # Vectorize the scan
    # lpf[L:R+2] gives lpf for L, L+1, ..., R+1
    # We need max(lpf[n], lpf[n+1]) for n in [L, R]
    a = lpf[L:R+1]      # lpf[L], lpf[L+1], ..., lpf[R]
    b = lpf[L+1:R+2]    # lpf[L+1], lpf[L+2], ..., lpf[R+1]
    vals = np.maximum(a, b)
    
    best_idx = np.argmin(vals)
    best_n = L + best_idx
    best_val = vals[best_idx]
    
    return int(best_n)
