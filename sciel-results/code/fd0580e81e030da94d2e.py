"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s limit
    
    def compute_phi_segment(L, R):
        """Compute phi(n) for all n in [L, R] using segmented sieve."""
        n = R - L + 1
        phi = list(range(L, R + 1))
        
        # We need primes up to sqrt(R)
        limit = int(math.isqrt(R)) + 1
        if limit >= 2:
            prime_sieve = [True] * (limit + 1)
            prime_sieve[0] = prime_sieve[1] = False
            for i in range(2, int(math.isqrt(limit)) + 1):
                if prime_sieve[i]:
                    for j in range(i * i, limit + 1, i):
                        prime_sieve[j] = False
            primes = [i for i in range(2, limit + 1) if prime_sieve[i]]
        else:
            primes = []
        
        for p in primes:
            # Find first multiple of p in [L, R]
            start = ((L + p - 1) // p) * p
            for j in range(start, R + 1, p):
                idx = j - L
                if phi[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
        
        # For numbers that are still equal to their original value, they are prime
        for i in range(n):
            val = L + i
            if val == 1:
                phi[i] = 1
            elif phi[i] == val:
                # This means no prime factor <= sqrt(val) was found, so val is prime
                phi[i] = val - 1
        
        return phi
    
    best_k = 1
    best_s = 1
    
    # Quick scan of small range
    small_limit = 200000
    phi_vals = compute_phi_segment(1, small_limit)
    seen = set()
    k = 0
    for i in range(small_limit):
        if phi_vals[i] not in seen:
            seen.add(phi_vals[i])
            k += 1
            if k > best_k:
                best_k = k
                best_s = i + 1
        else:
            if k > best_k:
                best_k = k
                best_s = i - k + 2
            seen = {phi_vals[i]}
            k = 1
    
    if k > best_k:
        best_k = k
        best_s = small_limit - k + 2
    
    # Now do a broader search using segmented sieves
    chunk_size = 200000
    search_limit = 10**12
    
    L = small_limit + 1
    while L + chunk_size - 1 <= search_limit and time.time() < deadline - 15:
        phi_seg = compute_phi_segment(L, L + chunk_size - 1)
        
        seen = set()
        k = 0
        for i in range(chunk_size):
            if phi_seg[i] not in seen:
                seen.add(phi_seg[i])
                k += 1
                if k > best_k:
                    best_k = k
                    best_s = L + i
            else:
                if k > best_k:
                    best_k = k
                    best_s = L + i - k + 1
                seen = {phi_seg[i]}
                k = 1
        
        if k > best_k:
            best_k = k
            best_s = L + chunk_size - k
        
        L += chunk_size
    
    # If we have time, do some random searches
    if time.time() < deadline - 20:
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            rand_start = random.randint(1, max(1, search_limit - chunk_size))
            if rand_start + chunk_size - 1 > search_limit:
                continue
            
            phi_seg = compute_phi_segment(rand_start, rand_start + chunk_size - 1)
            seen = set()
            k = 0
            for i in range(chunk_size):
                if phi_seg[i] not in seen:
                    seen.add(phi_seg[i])
                    k += 1
                    if k > best_k:
                        best_k = k
                        best_s = rand_start + i
                else:
                    if k > best_k:
                        best_k = k
                        best_s = rand_start + i - k + 1
                    seen = {phi_seg[i]}
                    k = 1
            
            if k > best_k:
                best_k = k
                best_s = rand_start + chunk_size - k
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
