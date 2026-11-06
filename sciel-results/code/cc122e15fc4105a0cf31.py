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
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # For each prime, compute its contribution to the powerful part
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        # Iterate over powers p^k for k >= 2
        pk = p2
        while pk <= N:
            limit = N // pk
            # Create mask for local indices 0..limit-1 corresponding to multiples 1..limit
            # We want local index i such that (i+1) is NOT divisible by p
            # i+1 divisible by p => i % p == p-1
            # So we set mask[p-1::p] to False
            mask = np.ones(limit, dtype=bool)
            if p <= limit:
                mask[p-1::p] = False
            
            # Get local indices where mask is True
            local_indices = np.nonzero(mask)[0]
            if len(local_indices) > 0:
                # Map to global indices
                # Global index = pk + local_index * pk
                global_indices = pk + local_indices * pk
                powerful[global_indices] *= pk
            
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
