import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 50  # safety margin

    MAX_N = 1000000
    MIN_N = 10000
    LIMIT = MAX_N + 1

    # Sieve of Eratosthenes
    is_prime = np.ones(LIMIT + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i : LIMIT + 1 : i] = False
    primes = np.where(is_prime)[0]

    # Compute powerful_part
    pp = np.ones(LIMIT + 1, dtype=np.int64)
    
    for p in primes:
        p = int(p)
        p2 = p * p
        if p2 > LIMIT:
            break
            
        # Calculate exponent of p for all n
        # exp[n] will hold v_p(n)
        exp = np.zeros(LIMIT + 1, dtype=np.int8)
        
        # Iteratively set exp for multiples of p^k
        pk = p
        k = 1
        while pk <= LIMIT:
            # Multiples of pk
            mask = np.arange(pk, LIMIT + 1, pk)
            exp[mask] = k
            k += 1
            pk *= p
            if pk > LIMIT:
                break
        
        # Contribution: p^e if e >= 2, else 1
        # We need p**exp, but only where exp >= 2
        # To avoid large powers or errors, we can compute p**exp safely
        # p**exp might overflow int64 if exp is large, but p^exp <= n <= 10^6, so it's safe.
        
        # Create contribution array
        contrib = np.ones(LIMIT + 1, dtype=np.int64)
        # Only update where exp >= 2
        mask_ge2 = exp >= 2
        if np.any(mask_ge2):
            # Compute p ** exp for those indices
            # np.power(p, exp[mask_ge2])
            contrib[mask_ge2] = p ** exp[mask_ge2]
            
        pp *= contrib

    # Scan for best n
    # pp is array of size LIMIT+1 (indices 0..LIMIT)
    # We need n in [MIN_N, MAX_N]
    # Score for n is pp[n] * pp[n+1]
    
    # Vectorize the scoring?
    # scores = pp[MN_N:MAX_N+1] * pp[MN_N+1:MAX_N+2]
    
    scores = pp[MIN_N:MAX_N+1] * pp[MIN_N+1:MAX_N+2]
    
    # Find index of max score
    max_idx = np.argmax(scores)
    best_n = MIN_N + max_idx
    best_score = scores[max_idx]
    
    if time.time() > DEADLINE:
         # This shouldn't happen if NumPy is fast
         pass

    return int(best_n)
