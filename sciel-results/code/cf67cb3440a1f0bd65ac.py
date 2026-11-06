import time
import random
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    # Keep a safety margin. 150s limit. Use 140s for safety.
    DEADLINE = 140.0
    
    # Increase LIMIT_N to 2 * 10^8
    # Memory check: 
    # phi: 200M * 4 bytes = 800 MB
    # is_prime: 200M * 1 byte = 200 MB
    # Total ~ 1 GB. Should be safe.
    LIMIT_N = 200 * 10**6
    
    # 1. Sieve of Eratosthenes using NumPy
    is_prime = np.ones(LIMIT_N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT_N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    # 2. Compute phi using NumPy
    # Use int32 as phi(n) < n <= 2*10^8
    phi = np.arange(LIMIT_N + 1, dtype=np.int32)
    
    # Get primes
    primes = np.where(is_prime)[0]
    
    for p in primes:
        # Multiples of p
        multiples = np.arange(p, LIMIT_N + 1, p)
        # Update phi
        phi[multiples] -= phi[multiples] // p
        
    # 3. Scan for longest run of distinct phi values
    # Use a numpy array for counts
    counts = np.zeros(LIMIT_N + 1, dtype=np.int32)
    
    left = 0
    best_k = 0
    best_s = 1
    
    phi_arr = phi
    counts_arr = counts
    
    # Optimization: Check time periodically
    last_time_check = start_time
    
    for right in range(1, LIMIT_N + 1):
        v = phi_arr[right]
        counts_arr[v] += 1
        
        # If we have a duplicate, shrink window from left
        while counts_arr[v] > 1:
            counts_arr[phi_arr[left]] -= 1
            left += 1
            
        current_k = right - left + 1
        if current_k > best_k:
            best_k = current_k
            best_s = left + 1
            
        # Early termination if we hit max possible k
        if best_k == 200:
            break
            
        # Time check every 10000 iterations
        if (right & 0x1FFF) == 0:
            if time.time() - last_time_check > 5.0:
                last_time_check = time.time()
                if time.time() - start_time > DEADLINE:
                    break

    return {'start': best_s, 'length': best_k}
