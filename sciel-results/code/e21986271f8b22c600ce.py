import math
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Sieve primes up to 10^6
    limit = 1000000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    def calc_phi(n):
        if n == 1:
            return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            result -= result // temp
        return result

    # We want to find a run of length 200.
    # Start from the top.
    max_s = 10**12
    target_k = 200
    
    # Try s from max_s - target_k + 1 downwards
    # We can check a batch to save time? No, we need to verify distinctness.
    # Just iterate s downwards.
    
    # Optimization: If we find a collision in the first check, we can just move s down.
    # The probability of collision is low, so we likely succeed on the first try.
    
    s = max_s - target_k + 1
    while s >= 1:
        if time.time() > deadline:
            # Return best found so far? 
            # We haven't found any yet in this loop if we just started.
            # But we can return a small valid run if we had one.
            # Since we expect success immediately, this is just a fallback.
            # Let's return a known small valid run if we fail.
            # e.g. s=1, k=1 is always valid.
            # But we want to maximize k.
            # If we time out, it means we checked many s and found no k=200.
            # This is extremely unlikely.
            # Fallback: return s=1, k=1.
            return {'start': 1, 'length': 1}
            
        vals = []
        valid = True
        seen = set()
        for i in range(target_k):
            n = s + i
            phi_val = calc_phi(n)
            if phi_val in seen:
                valid = False
                break
            seen.add(phi_val)
            vals.append(phi_val)
            
        if valid:
            return {'start': s, 'length': target_k}
            
        s -= 1
        
    return {'start': 1, 'length': 1}
