import time
import random
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # We will scan up to LIMIT_N
    # Increased to 5*10^7 to cover more ground and find longer runs
    # Using int32 to save memory
    LIMIT_N = 50 * 10**6
    
    # 1. Sieve of Eratosthenes using NumPy
    # is_prime array
    is_prime = np.ones(LIMIT_N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT_N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    # 2. Compute phi using NumPy
    # Use int32 as phi(n) < n <= 5*10^7
    phi = np.arange(LIMIT_N + 1, dtype=np.int32)
    
    # Get primes
    primes = np.where(is_prime)[0]
    
    for p in primes:
        # Multiples of p
        multiples = np.arange(p, LIMIT_N + 1, p)
        # Update phi
        phi[multiples] -= phi[multiples] // p
        
    # 3. Scan for longest run of distinct phi values
    # Use a numpy array for counts to speed up the sliding window
    # phi values are in [0, LIMIT_N], so we can index directly
    # Note: phi[1]=1, phi[2]=1, etc.
    # We need to track counts of each phi value in the current window.
    # Max phi value is LIMIT_N.
    counts = np.zeros(LIMIT_N + 1, dtype=np.int32)
    
    left = 0
    best_k = 0
    best_s = 1
    
    # We iterate over the phi array
    # phi[0] is 0, phi[1] is 1.
    # We care about s >= 1, so indices 1 to LIMIT_N.
    
    # Optimization: Access phi and counts as local variables for speed
    phi_arr = phi
    counts_arr = counts
    
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
            best_s = left + 1  # Convert to 1-based index for 'start'
            
        # Early termination if we hit max possible k
        if best_k == 200:
            break
            
        # Time check
        if time.time() - start_time > DEADLINE:
            break

    return {'start': best_s, 'length': best_k}
