import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    def elapsed():
        return time.time() - t0

    MAXN = 1_000_000
    
    # Initialize powerful part array with 1s
    pp = np.ones(MAXN + 2, dtype=np.int64)
    
    # Sieve for primes up to MAXN
    # We only need primes up to sqrt(MAXN) for the powerful part sieve,
    # but finding all primes is fast.
    is_prime = np.ones(MAXN + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAXN**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Calculate powerful parts
    for p in primes:
        if p * p > MAXN:
            break
            
        p_j = p * p
        while p_j <= MAXN:
            p_next = p_j * p
            
            if p_next > MAXN:
                # All multiples of p_j have v_p(m) == j
                pp[p_j::p_j] *= p_j
            else:
                # Multiples of p_j but NOT p_next
                count = MAXN // p_j - MAXN // p_next
                if count > 500:
                    indices = np.arange(p_j, MAXN + 1, p_j)
                    mask = indices % p_next != 0
                    pp[indices[mask]] *= p_j
                else:
                    for m in range(p_j, MAXN + 1, p_j):
                        if m % p_next != 0:
                            pp[m] *= p_j
            
            p_j = p_next
            
            if elapsed() > DEADLINE:
                break
    
    if elapsed() > DEADLINE:
        return 10000
    
    # Scan for the best pair
    # We can vectorize the product calculation for the whole range
    # pp[n] * pp[n+1] for n in [10000, MAXN-1]
    # Note: pp is 1-indexed in the logic (pp[1] is for 1), but Python is 0-indexed.
    # The task asks for n in [10000, 1000000].
    # pp[n] corresponds to the powerful part of n.
    
    # Create a view or slice for n from 10000 to MAXN-1
    # We need pp[n] and pp[n+1]
    # Let start = 10000
    # products = pp[start:MAXN] * pp[start+1:MAXN+1]
    
    start = 10000
    # Ensure we don't go out of bounds
    # n ranges from start to MAXN-1
    # pp[n] is valid for n <= MAXN
    # pp[n+1] is valid for n+1 <= MAXN+1, i.e., n <= MAXN
    # Our array size is MAXN+2, so index MAXN+1 is valid.
    
    products = pp[start:MAXN] * pp[start+1:MAXN+1]
    
    # Find the index of the max product
    # max_idx is relative to the slice
    max_idx = np.argmax(products)
    best_val = products[max_idx]
    best_n = start + max_idx
    
    return int(best_n)
