import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # Pre-compute primes up to 10^6 for fast division
    # We need primes up to sqrt(10^12) = 10^6
    LIMIT = 10**6
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    def compute_phi_block(L, R):
        """
        Computes phi(n) for all n in [L, R] using a segmented sieve approach.
        Returns a list of phi values.
        """
        size = R - L + 1
        # Initialize phi[i] = L + i
        phi = [L + i for i in range(size)]
        
        # For each prime p, update phi for multiples of p
        # We only need primes p such that p*p <= R
        # Actually, we can use all primes p <= R, but if p > sqrt(R), 
        # there is at most one multiple in the block, or we can handle the remainder later.
        # Standard segmented phi:
        # 1. For p <= sqrt(R): divide out p from all multiples.
        # 2. For remaining phi[i] > 1, it is a prime factor. Divide it out.
        
        sqrt_R = int(math.isqrt(R))
        
        for p in primes:
            if p > sqrt_R:
                break
            # Find first multiple of p in [L, R]
            m = ((L + p - 1) // p) * p
            # Iterate through multiples
            for j in range(m, R + 1, p):
                idx = j - L
                if phi[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
        
        # Handle remaining prime factors
        for i in range(size):
            if phi[i] > 1:
                # phi[i] is a prime factor (or 1)
                phi[i] -= phi[i] // phi[i]
            # Special case: phi(1) = 1. The loop above leaves phi[1] as 1 if L=1, i=0.
            # If L=1, phi[0] starts as 1. p loop doesn't touch it (no multiples of p in 1..R for p>=2? 
            # m for p=2 is 2. So phi[0] remains 1.
            # phi[i] > 1 check: 1 is not > 1. So it stays 1. Correct.
            
        return phi

    def check_window_from_phi(phi_vals, max_len=200):
        """
        Finds the longest run of distinct values in phi_vals.
        """
        best_k = 0
        best_start_offset = 0
        n = len(phi_vals)
        
        # We want max k such that phi_vals[start:start+k] are distinct.
        # Use a sliding window or just check all starts?
        # n is ~10^6. max_len is 200.
        # Checking all starts is O(n * max_len) = 2 * 10^8. Might be slow in Python.
        # Better: Use a set for each start? No.
        # Use a "last seen" array or dict?
        # Since we just need the *longest* run, we can iterate.
        
        # Actually, we can just scan.
        # For a fixed start s, we extend until collision.
        # This is O(n * avg_run_length).
        
        # Optimization: If we find a run of length k, we don't need to check starts 
        # that are within that run if we just want the global max? 
        # No, a shorter run might be part of a longer one starting earlier? 
        # No, if phi[s..s+k-1] are distinct, any sub-run is distinct.
        # We just need to find the max length.
        
        # Let's use a simple loop.
        # To speed up, we can break early if n - s < best_k.
        
        for s in range(n):
            if n - s <= best_k:
                break
            
            seen = set()
            k = 0
            for i in range(s, min(s + 200, n)):
                v = phi_vals[i]
                if v in seen:
                    break
                seen.add(v)
                k += 1
            
            if k > best_k:
                best_k = k
                best_start_offset = s
        
        return best_k, best_start_offset

    best_k = 1
    best_s = 1
    
    # Strategy: Check several blocks
    # Block 1: 1 to 10^6
    # Block 2: 10^6 to 2*10^6
    # ...
    # Maybe a block near 10^12?
    
    block_size = 10**6
    
    # Check small blocks
    num_blocks = 10
    for b in range(num_blocks):
        if time.time() - start_time > DEADLINE - 10:
            break
        L = b * block_size + 1
        R = L + block_size - 1
        if R > 10**12:
            R = 10**12
        if L > R:
            break
            
        phi_vals = compute_phi_block(L, R)
        k, offset = check_window_from_phi(phi_vals)
        if k > best_k:
            best_k = k
            best_s = L + offset
    
    # Check a block near 10^12?
    # The distribution of phi(n) for large n is different.
    # Let's try one large block if time permits.
    if time.time() - start_time < DEADLINE - 30:
        L = 10**12 - block_size
        R = 10**12
        if L < 1: L = 1
        phi_vals = compute_phi_block(L, R)
        k, offset = check_window_from_phi(phi_vals)
        if k > best_k:
            best_k = k
            best_s = L + offset
            
    return {'start': best_s, 'length': best_k}
