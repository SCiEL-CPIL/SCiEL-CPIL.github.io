import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # 2. Fast phi function
    def phi(n):
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            result -= result // temp
        return result
    
    # 3. Search for a valid start s
    max_k = 200
    best_s = 1
    best_k = 1
    
    def check_range(s, length):
        """Check if phi(s)...phi(s+length-1) are pairwise distinct.
        Returns the longest distinct run starting from s."""
        seen = {}
        start_idx = 0
        max_run = 0
        for i in range(length):
            val = phi(s + i)
            if val in seen and seen[val] >= start_idx:
                start_idx = seen[val] + 1
            seen[val] = i
            run_len = i - start_idx + 1
            if run_len > max_run:
                max_run = run_len
                if max_run >= max_k:
                    break
        return max_run, start_idx
    
    # Strategy: Try several candidate ranges
    # Larger numbers tend to have longer runs of distinct phi values
    candidates = []
    
    # Structured candidates at various scales
    for base in [10**9, 10**10, 10**11, 10**12 - 1000]:
        for offset in range(0, 2000, 50):
            s = base + offset
            if s >= 1 and s + max_k - 1 <= 10**12:
                candidates.append(s)
    
    # Random candidates
    random.seed(42)
    for _ in range(3000):
        s = random.randint(10**9, 10**12 - max_k)
        candidates.append(s)
    
    # Sort candidates to try promising ones first
    # Larger s values are more likely to have long runs
    candidates.sort(reverse=True)
    
    for s in candidates:
        if time.time() > deadline - 5.0:
            break
        
        # Check a window of 200 values starting at s
        run_len, offset = check_range(s, max_k)
        
        if run_len > best_k:
            best_k = run_len
            best_s = s + offset
            if best_k == max_k:
                break
    
    # Fallback: ensure we have a valid result
    if best_k < 1:
        best_s = 1
        best_k = 1
    
    # Final validation: make sure best_s + best_k - 1 <= 10^12
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
        if best_k < 1:
            best_s = 1
            best_k = 1
    
    return {'start': best_s, 'length': best_k}
