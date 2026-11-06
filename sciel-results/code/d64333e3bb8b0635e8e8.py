import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    LIMIT = 1000000
    MIN_N = 10000
    MAX_N = LIMIT
    
    # Precompute LPF for all numbers up to MAX_N + 1
    # lpf[i] will store the largest prime factor of i
    lpf = np.zeros(MAX_N + 2, dtype=np.int64)
    
    # Sieve for largest prime factor
    # Initialize lpf[i] = i for all i >= 2
    # We can do this efficiently by iterating through primes
    # Or simply: for i in range(2, MAX_N + 2): lpf[i] = i
    # Then for each prime p, update multiples
    
    # Better approach:
    # 1. Identify primes using a simple sieve or just iterate and check if lpf[i] == i
    # 2. For each prime p, for multiples m of p, set lpf[m] = p (since p is increasing, later primes are larger)
    
    # Let's use a standard sieve to generate primes first, or just iterate.
    # Iterating i from 2 to MAX_N+1:
    # If lpf[i] == 0, then i is prime.
    # For each prime i, iterate j from i to MAX_N+1 step i, and set lpf[j] = i.
    # Since we go in increasing order of i (primes), the last prime that divides j will be the largest.
    
    for i in range(2, MAX_N + 2):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, MAX_N + 2, i):
                lpf[j] = i
                
    # Now lpf[x] contains the largest prime factor of x for all x in [2, MAX_N+1]
    # lpf[0] and lpf[1] are 0, which is fine as we start from MIN_N=10000
    
    # We want to minimize max(lpf[n], lpf[n+1]) for n in [MIN_N, MAX_N]
    # Vectorized operation:
    # lpf_n = lpf[MIN_N : MAX_N + 1]
    # lpf_np1 = lpf[MIN_N + 1 : MAX_N + 2]
    # scores = np.maximum(lpf_n, lpf_np1)
    # min_score = np.min(scores)
    # best_idx = np.argmin(scores)
    # best_n = MIN_N + best_idx
    
    lpf_n = lpf[MIN_N : MAX_N + 1]
    lpf_np1 = lpf[MIN_N + 1 : MAX_N + 2]
    
    # Calculate max for each pair
    # Using numpy maximum for speed
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_val = np.min(max_lpf)
    best_idx = np.argmin(max_lpf)
    
    best_n = MIN_N + best_idx
    
    return int(best_n)
