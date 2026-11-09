import random
import math

def construct():
    # Sieve of Eratosthenes for primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    def compute_phis(start, length):
        tot = [start + i for i in range(length)]
        rem = [start + i for i in range(length)]
        
        # Process small primes
        # We only need primes up to sqrt(start + length)
        max_p = int(math.sqrt(start + length)) + 1
        
        for p in primes:
            if p > max_p:
                break
            # Find first index i such that (start + i) % p == 0
            # start + i = k*p => i = k*p - start
            # i % p = (-start) % p
            i0 = (-start) % p
            for i in range(i0, length, p):
                if rem[i] % p == 0:
                    tot[i] -= tot[i] // p
                    while rem[i] % p == 0:
                        rem[i] //= p
                        
        # Handle remaining large prime factors
        for i in range(length):
            if rem[i] > 1:
                tot[i] -= tot[i] // rem[i]
                
        return tot

    # We want to maximize k. Max possible is 200.
    # Search in high range where collisions are rare.
    # Random search
    
    best_k = 1
    best_s = 1
    
    # Try a few random starting points in [10^11, 10^12]
    for _ in range(10): # 10 attempts
        s = random.randint(10**11, 10**12 - 200)
        
        phis = compute_phis(s, 200)
        
        if len(set(phis)) == 200:
            return {'start': s, 'length': 200}
            
    # Fallback if no 200 found (extremely unlikely)
    # Try to find max k for the last s
    s = random.randint(10**11, 10**12 - 200)
    phis = compute_phis(s, 200)
    seen = set()
    k = 0
    for val in phis:
        if val in seen:
            break
        seen.add(val)
        k += 1
    return {'start': s, 'length': k}
