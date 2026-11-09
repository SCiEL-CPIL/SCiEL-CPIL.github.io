import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Generate primes up to 10^6 and SPF table
    limit = 10**6
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    primes = [i for i in range(2, limit + 1) if spf[i] == i]
    
    def phi(x):
        result = x
        n = x
        while n > 1:
            p = spf[n] if n <= limit else n
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            else:
                # n is prime (or has a prime factor > limit)
                break
        if n > 1:
            result -= result // n
        return result
    
    best_k = 1
    best_s = 1
    
    # 2. Search in high range
    low = 10**11
    high = 10**12 - 200
    
    random.seed(42)
    
    def check_run(s, max_len=200):
        """Check for a run of distinct phi values starting at s."""
        seen = set()
        k = 0
        for i in range(max_len):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k
    
    def try_extend(s, initial_k):
        """Try to extend a partial run both forward and backward."""
        # First, verify the initial run
        seen = set()
        actual_k = 0
        for i in range(initial_k):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            actual_k += 1
        
        if actual_k == 0:
            return 1, s
        
        # Try extending forward
        end_s = s + actual_k
        for i in range(1, 200 - actual_k + 1):
            if time.time() > deadline:
                break
            val = phi(end_s + i)
            if val in seen:
                break
            seen.add(val)
            actual_k += 1
        
        # Try extending backward
        start_s = s
        for i in range(1, 200 - actual_k + 1):
            if time.time() > deadline:
                break
            val = phi(start_s - i)
            if val in seen:
                break
            seen.add(val)
            start_s -= i
            actual_k += 1
        
        return actual_k, start_s
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        s = random.randint(low, high)
        
        # Quick check: compute phi for first few to see if there's a collision early
        seen = set()
        k = 0
        for i in range(200):
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        
        if k == 200:
            return {'start': s, 'length': 200}
        
        # If we got a decent partial run, try to extend it
        if k >= 10:
            ext_k, ext_s = try_extend(s, k)
            if ext_k > best_k:
                best_k = ext_k
                best_s = ext_s
                if best_k == 200:
                    return {'start': best_s, 'length': 200}
        elif k > best_k:
            best_k = k
            best_s = s
    
    return {'start': best_s, 'length': best_k}
