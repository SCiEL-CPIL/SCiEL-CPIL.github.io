import math
import time
import random

def construct():
    # Deadline guard
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
    # We aim for k=200.
    # Try random large s values.
    # Range: [10^9, 10^12]
    max_k = 200
    best_s = 1
    best_k = 1
    
    # Try a few deterministic large values first, then random
    # To ensure we don't miss a trivial solution, we can also check small s, 
    # but small s has higher collision risk. Large s is better.
    
    # Let's try s = 10^11 + i for i in 0..1000
    # And random s.
    
    candidates = []
    # Add some structured candidates
    base = 10**11
    for i in range(1000):
        candidates.append(base + i * 12345)
        
    # Add random candidates
    while len(candidates) < 5000:
        s = random.randint(10**9, 10**12 - max_k)
        candidates.append(s)

    for s in candidates:
        if time.time() > deadline:
            break
            
        # Check if phi(s)...phi(s+max_k-1) are distinct
        seen = set()
        k = 0
        valid = True
        for i in range(max_k):
            val = phi(s + i)
            if val in seen:
                valid = False
                break
            seen.add(val)
            k += 1
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == max_k:
                # Found the maximum possible length
                # We can break early to save time, or verify one more time
                # Given the contract, we just need to return a valid pair.
                # Returning early is safe.
                break

    # If we didn't find 200, we might have found a smaller k.
    # We should ensure best_s + best_k - 1 <= 10^12
    if best_s + best_k - 1 > 10**12:
        # Fallback to a very small range which is safe
        best_s = 1
        # Recalculate for small range just in case, though 1..200 is likely fine
        seen = set()
        k = 0
        for i in range(200):
             if 1 + i > 10**12: break
             val = phi(1 + i)
             if val in seen:
                 break
             seen.add(val)
             k += 1
        best_k = k
        if best_k < 1:
             best_k = 1

    return {'start': best_s, 'length': best_k}
