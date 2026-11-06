"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    N = 1000000
    # powerful[i] will store the powerful part of i
    # powerful[i] = product of p^e for all primes p where e = v_p(i) >= 2
    powerful = np.ones(N + 1, dtype=np.int64)
    
    # Sieve to find primes
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    limit = int(np.sqrt(N)) + 1
    for i in range(2, limit):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    # Convert to list for faster iteration in Python
    primes = np.where(is_prime)[0].tolist()
    
    # For each prime, compute its contribution to the powerful part
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        # Iterate over powers p^k for k >= 2
        pk = p2
        while pk <= N:
            # Multiples of pk
            multiples = np.arange(pk, N + 1, pk)
            if len(multiples) == 0:
                break
            
            # We need to identify which of these have v_p(m) == k
            # m = j * pk. v_p(m) = k + v_p(j).
            # We want v_p(m) == k, which means v_p(j) == 0, i.e., j is not divisible by p.
            # j = m / pk
            js = multiples // pk
            # Mask for js not divisible by p
            mask = (js % p) != 0
            
            if np.any(mask):
                powerful[multiples[mask]] *= pk
            
            # Move to next power
            pk *= p
    
    # Now find the pair (n, n+1) with n in [10000, 999999] that maximizes powerful[n]*powerful[n+1]
    start = 10000
    end = 999999  # inclusive, so n+1 = 1000000 max
    
    products = powerful[start:end+1] * powerful[start+1:end+2]
    
    best_idx = np.argmax(products)
    best_n = start + best_idx
    
    return int(best_n)
# EVOLVE-BLOCK-END
