import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    def phi(n, primes):
        if n == 1:
            return 1
        result = n
        y = n
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    best_k = 1
    best_s = 1
    
    # Search in large regions where collisions are less likely
    # We try a few starting points spread across the upper range
    candidates = []
    # Add some structured candidates
    for base in range(10**11, 10**12, 10**10):
        candidates.append(base)
    # Add a few random-ish large numbers
    import random
    random.seed(42)
    for _ in range(20):
        candidates.append(random.randint(10**11, 10**12 - 200))
    
    # Sort to ensure we check diverse regions
    candidates = sorted(set(candidates))
    
    for L in candidates:
        if time.time() > deadline:
            break
            
        # Compute phi for L to L+199
        phis = []
        for i in range(200):
            n = L + i
            if n > 10**12:
                break
            phis.append(phi(n, primes))
        
        if len(phis) < 200:
            continue
            
        # Check for distinctness
        seen = set()
        k = 0
        for p_val in phis:
            if p_val in seen:
                break
            seen.add(p_val)
            k += 1
            
        if k > best_k:
            best_k = k
            best_s = L
            if best_k == 200:
                break

    return {'start': best_s, 'length': best_k}
