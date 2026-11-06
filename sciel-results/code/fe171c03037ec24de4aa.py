import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s margin
    
    max_n = 1000001  # We need powerful_part for n and n+1, where n <= 1000000
    powerful = np.ones(max_n + 1, dtype=np.int64)
    
    # Sieve for primes up to max_n
    is_prime = np.ones(max_n + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(max_n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # For each prime, compute the powerful part contribution
    for p in primes:
        if p * p > max_n:
            break
        
        # For each power p^k where k >= 2
        pk = p * p  # p^2
        while pk <= max_n:
            # Find all multiples of pk that have EXACTLY p^k in their factorization
            # i.e., divisible by pk but not by pk * p
            pk_next = pk * p
            if pk_next <= max_n:
                # Multiples of pk
                multiples = np.arange(pk, max_n + 1, pk)
                # Filter out those also divisible by pk_next
                mask = (multiples % pk_next != 0)
                multiples = multiples[mask]
            else:
                # All multiples of pk have exactly p^k (since pk * p > max_n)
                multiples = np.arange(pk, max_n + 1, pk)
            
            if len(multiples) > 0:
                powerful[multiples] *= pk
            
            pk = pk_next
    
    # Compute product of consecutive powerful parts for n in [10000, 1000000]
    # powerful[n] * powerful[n+1]
    products = powerful[10000:1000001] * powerful[10001:1000002]
    
    # Find the index of maximum
    max_idx = np.argmax(products)
    best_n = 10000 + max_idx
    
    # Verify it's in range
    if best_n < 10000 or best_n > 1000000:
        best_n = 10000
    
    return int(best_n)
