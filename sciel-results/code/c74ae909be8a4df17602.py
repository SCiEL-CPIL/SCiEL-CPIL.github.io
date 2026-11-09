import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # We need to calculate LPF for n and n+1
    # Max n is 1,000,000, so max n+1 is 1,000,001
    max_val = 1000001
    size = max_val + 1
    
    # Step 1: Sieve of Eratosthenes to get list of primes up to sqrt(max_val)
    # But actually we need all primes up to max_val for LPF computation
    # Let's use a two-pass approach:
    # Pass 1: Find all primes using a simple sieve
    # Pass 2: For each prime, update LPF of its multiples
    
    # Initialize is_prime array
    is_prime = np.ones(size, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    # Sieve of Eratosthenes
    limit = int(np.sqrt(max_val)) + 1
    for i in range(2, limit):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    # Get list of primes
    primes = np.where(is_prime)[0]
    
    # Initialize LPF array
    lpf = np.zeros(size, dtype=np.int64)
    
    # For each prime, update all its multiples with this prime as LPF
    # Since we process primes in increasing order, the last prime to divide a number
    # will be its largest prime factor
    for p in primes:
        if time.time() > deadline:
            break
        # Update multiples of p: p, 2p, 3p, ...
        lpf[p::p] = p
    
    # Calculate max LPF for consecutive pairs n, n+1
    lpf_n = lpf[10000:1000001]
    lpf_n1 = lpf[10001:1000002]
    
    # Compute element-wise maximum
    max_lpf_pairs = np.maximum(lpf_n, lpf_n1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(max_lpf_pairs)
    
    # The actual n value
    best_n = 10000 + min_idx
    
    return int(best_n)
