import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    MAX_N = 1000001
    
    # 1. Generate primes up to MAX_N using NumPy sieve
    is_prime = np.ones(MAX_N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = np.nonzero(is_prime)[0]
    
    # 2. Compute Largest Prime Factor (LPF) for all numbers up to MAX_N
    # Initialize with 1
    lpf_np = np.ones(MAX_N + 1, dtype=np.int32)
    # For each prime p, set lpf[multiples of p] = p
    # Since we iterate p in increasing order, the last assignment for a number 
    # will be by its largest prime factor.
    for p in primes:
        lpf_np[p::p] = p
        
    # 3. Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    # We need lpf values for indices 10000 to 1000001
    # lpf_np is 0-indexed.
    # We want max(lpf_np[i], lpf_np[i+1]) for i in [10000, 1000000]
    
    # Extract the relevant slice
    # lpf_np[10000:1000002] gives lpf for 10000..1000001
    # Let arr = lpf_np[10000:1000002]
    # We want min(max(arr[i], arr[i+1])) for i in 0..len(arr)-2
    # This is equivalent to min over i of max(arr[i], arr[i+1])
    
    relevant_lpf = lpf_np[10000:1000002]
    # Compute pairwise max
    # We can use numpy roll or just slicing
    # max_vals[i] = max(relevant_lpf[i], relevant_lpf[i+1])
    max_vals = np.maximum(relevant_lpf[:-1], relevant_lpf[1:])
    
    idx = int(np.argmin(max_vals))
    best_n = 10000 + idx
    best_val = int(max_vals[idx])
    
    return best_n
