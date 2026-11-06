import time
import math
import random
import numpy as np

def sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            start = i * i
            step = i
            is_prime[start::step] = bytearray(len(is_prime[start::step]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Primes up to 10^6 for trial division
    primes = sieve_primes(10**6)
    
    def compute_phi_range_fast(s, k):
        """Compute phi values for [s, s+k-1] using segmented approach."""
        if k <= 0:
            return []
        
        # Initialize phi and remaining arrays
        phi_vals = np.arange(s, s + k, dtype=np.int64)
        remaining = np.arange(s, s + k, dtype=np.int64)
        
        # Process each prime
        max_n = s + k - 1
        for p in primes:
            if p * p > max_n:
                break
            
            # Find first multiple of p in [s, s+k-1]
            start_idx = ((s + p - 1) // p) * p - s
            
            # Vectorized update for all multiples of p
            indices = np.arange(start_idx, k, p)
            
            # Check which have p as factor (they should all, but verify)
            mask = remaining[indices] % p == 0
            if np.any(mask):
                idxs = indices[mask]
                # Divide out all factors of p
                while np.any(remaining[idxs] % p == 0):
                    remaining[idxs] //= p
                # Update phi: phi(n) = phi(n) * (p-1) / p
                phi_vals[idxs] = phi_vals[idxs] * (p - 1) // p
        
        # Handle remaining large prime factors
        for i in range(k):
            if remaining[i] > 1:
                q = remaining[i]
                phi_vals[i] = phi_vals[i] * (q - 1) // q
        
        return phi_vals.tolist()
    
    def check_run(s, k):
        if s < 1 or s + k - 1 > 10**12:
            return False
        vals = compute_phi_range_fast(s, k)
        return len(set(vals)) == k
    
    target_k = 200
    
    # Strategy: Dense scan near 10^12 with step 1
    # Near large numbers, collisions are rare
    s_start = 10**12 - 199
    s_end = max(1, 10**12 - 500000)
    
    # Scan downwards from 10^12
    for s in range(s_start, s_end - 1, -1):
        if time.time() > deadline:
            break
        if check_run(s, target_k):
            return {'start': s, 'length': target_k}
    
    # If not found, try random spots
    random.seed(42)
    for _ in range(10000):
        if time.time() > deadline:
            break
        s_try = random.randint(10**8, 10**12 - 199)
        if check_run(s_try, target_k):
            return {'start': s_try, 'length': target_k}
    
    # Fallback: return a valid short run
    # Find any valid run of length >= 1
    for k in [100, 50, 20, 10, 5, 3, 2, 1]:
        if time.time() > deadline:
            break
        s_try = 10**12 - k
        if check_run(s_try, k):
            return {'start': s_try, 'length': k}
    
    return {'start': 1, 'length': 1}
