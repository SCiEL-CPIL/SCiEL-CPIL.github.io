import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    N = 1000000
    # Precompute powerful_part for all numbers 1..N+1
    pp = np.ones(N + 2, dtype=np.int64)  # index 0..N+1
    
    # Sieve for primes
    is_prime = np.ones(N + 2, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N + 1)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = np.where(is_prime)[0]
    
    # For each prime, compute its contribution to the powerful part
    for p in primes:
        p = int(p)
        if p * p > N + 1:
            break
            
        # Check time
        if time.time() - start_time > DEADLINE - 5:
            # If we run out of time, we return the best found so far.
            # However, the sieve is fast, so this is unlikely.
            # We can just break and use the partial pp array.
            break

        factor = np.ones(N + 2, dtype=np.int64)
        
        power = p * p
        # Multiplicative contribution:
        # If v_p(n) = e >= 2, we want to multiply pp[n] by p^e.
        # We achieve this by multiplying by p^2 for multiples of p^2,
        # and by p for multiples of p^k (k>=3).
        
        # Step 2: multiply by p^2
        factor[power::power] *= p * p
        
        power *= p
        while power <= N + 1:
            # Step k>=3: multiply by p
            factor[power::power] *= p
            power *= p
            
        pp *= factor
        
    best_n = 10000
    best_val = pp[10000] * pp[10001]
    
    if time.time() - start_time > DEADLINE - 2:
        return int(best_n)

    # Vectorized search for max product
    # n ranges from 10000 to N-1
    # products[i] = pp[10000+i] * pp[10001+i]
    products = pp[10000:N] * pp[10001:N+1]
    max_idx = np.argmax(products)
    best_n = 10000 + max_idx
    best_val = products[max_idx]
    
    return int(best_n)
