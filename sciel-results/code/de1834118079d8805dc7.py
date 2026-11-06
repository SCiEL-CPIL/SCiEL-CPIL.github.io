import math
import time
import numpy as np

DEADLINE = 85.0  # safety margin before 90s kill

def construct():
    start_time = time.time()
    N = 1000000
    MAX_VAL = N + 1
    
    # Sieve of Eratosthenes to get primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = [True] * (limit + 1)
    if limit >= 0:
        is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # Initialize powerful part array
    # Use int64 to be safe for products, though P(n) itself fits in int32
    P = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # Compute powerful parts
    for p in primes:
        pe = p * p
        if pe > MAX_VAL:
            break
        pe_next = pe * p
        while pe <= MAX_VAL:
            # Get indices divisible by pe
            # np.arange is efficient here
            indices = np.arange(pe, MAX_VAL + 1, pe)
            
            if pe_next > MAX_VAL:
                # All these indices have exactly exponent k (or more, but no higher power exists in range)
                # Actually, if pe_next > MAX_VAL, no multiple of pe_next is in range.
                # So all indices in 'indices' are divisible by pe but not pe_next.
                P[indices] *= pe
                break
            else:
                # Filter out indices divisible by pe_next
                # We want indices where n % pe_next != 0
                mask = indices % pe_next != 0
                P[indices[mask]] *= pe
                pe = pe_next
                pe_next = pe * p
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            return 10000

    if time.time() - start_time > DEADLINE:
        return 10000

    # Find n in [10000, 1000000] maximizing P[n] * P[n+1]
    # P is 1-indexed effectively for the values we care about, but array is 0-indexed.
    # P[i] is the powerful part of i.
    # We need P[n] * P[n+1] for n in [10000, 1000000].
    # n+1 goes up to 1000001.
    # P has size MAX_VAL + 1 = 1000002. Indices 0..1000001.
    # So P[1000001] is valid.
    
    products = P[10000:N+1] * P[10001:N+2]
    
    # Ensure we don't have any 0s or issues, though P should be >= 1
    idx = int(np.argmax(products))
    best_n = 10000 + idx
    
    return best_n
