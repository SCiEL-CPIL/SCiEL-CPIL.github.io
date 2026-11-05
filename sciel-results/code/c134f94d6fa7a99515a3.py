import math
import time
import random

def construct():
    start_time = time.time()
    LIMIT = 115.0  # seconds, leaving 5s margin for 120s limit
    MAX_S = 10**12
    MAX_K = 200

    # 1. Generate primes up to 10^6 (sqrt(10^12))
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]

    # 2. Segmented Sieve for Phi
    def compute_phis(s, k):
        """
        Computes phi(n) for n in [s, s+k-1] using a segmented sieve.
        Returns a list of k phi values.
        """
        phi = [s + i for i in range(k)]
        rem = [s + i for i in range(k)]
        
        # Sieve with small primes
        for p in primes:
            if p * p > s + k:
                break
            
            # Find the first multiple of p in [s, s+k-1]
            # start = ceil(s/p) * p
            start = ((s + p - 1) // p) * p
            
            for j in range(start, s + k, p):
                idx = j - s
                # Update phi
                phi[idx] -= phi[idx] // p
                # Remove p from rem
                while rem[idx] % p == 0:
                    rem[idx] //= p
        
        # Handle remaining large prime factors
        for i in range(k):
            if rem[i] > 1:
                phi[i] -= phi[i] // rem[i]
                
        return phi

    # 3. Helper to find longest distinct run
    def longest_distinct_run(vals):
        n = len(vals)
        if n == 0:
            return 0, 0
        max_len = 1
        max_start = 0
        cur_start = 0
        seen = {}
        for i in range(n):
            v = vals[i]
            if v in seen and seen[v] >= cur_start:
                cur_start = seen[v] + 1
            seen[v] = i
            cur_len = i - cur_start + 1
            if cur_len > max_len:
                max_len = cur_len
                max_start = cur_start
        return max_len, max_start

    # 4. Search Strategy
    # Since collisions are rare, random search is effective.
    # We focus on high numbers where phi(n) is large and distinctness is likely.
    
    rng = random.Random(42)
    best_k = 1
    best_s = 1
    
    # Search in [10^11, 10^12]
    low = 10**11
    high = MAX_S - MAX_K
    
    while time.time() - start_time < LIMIT:
        s = rng.randint(low, high)
        
        phis = compute_phis(s, MAX_K)
        
        if len(set(phis)) == MAX_K:
            return {'start': s, 'length': MAX_K}
        else:
            # Track best run in this window
            k, offset = longest_distinct_run(phis)
            if k > best_k:
                best_k = k
                best_s = s + offset
                
    # If we exit loop without finding 200, return best found
    return {'start': best_s, 'length': best_k}
