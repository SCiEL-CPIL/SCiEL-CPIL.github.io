import numpy as np
import random

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p:limit+1:p] = False
    return np.where(sieve)[0]

def construct():
    primes = _primes(N)
    num_primes = len(primes)
    
    # Precompute bitmasks for each prime and residue
    # mask[p_idx][r] is an integer with bits set at indices i such that i % p == r
    # We store these in a list of lists
    prime_masks = []
    for p in primes:
        masks_for_p = [0] * p
        for r in range(p):
            m = 0
            # Set bits for i in [0, N] where i % p == r
            # i = r, r+p, r+2p, ...
            for i in range(r, N + 1, p):
                m |= (1 << i)
            masks_for_p[r] = m
        prime_masks.append(masks_for_p)
        
    full_mask = (1 << (N + 1)) - 1
    best_size = 0
    best_mask = 0
    
    # Bitset operations are fast, so we can do many restarts
    num_restarts = 2000
    
    for _ in range(num_restarts):
        mask = full_mask
        
        for p_idx, p in enumerate(primes):
            current_size = mask.bit_count()
            
            # If p > current_size, no constraint
            if p > current_size:
                break
            
            masks_for_p = prime_masks[p_idx]
            
            # Calculate counts for each residue
            min_count = N + 1
            min_indices = []
            
            for r in range(p):
                count = (mask & masks_for_p[r]).bit_count()
                if count == 0:
                    # Already admissible for this prime
                    min_count = 0
                    min_indices = [r]
                    break
                if count < min_count:
                    min_count = count
                    min_indices = [r]
                elif count == min_count:
                    min_indices.append(r)
            
            if min_count == 0:
                continue
                
            # Randomly pick one candidate to remove
            target_residue = random.choice(min_indices)
            mask &= ~masks_for_p[target_residue]
            
        final_size = mask.bit_count()
        if final_size > best_size:
            best_size = final_size
            best_mask = mask
            
    if best_mask == 0:
        return list(range(N + 1))
        
    # Convert bitmask to list of integers
    result = []
    m = best_mask
    i = 0
    while m:
        if m & 1:
            result.append(i)
        m >>= 1
        i += 1
    return result
# EVOLVE-BLOCK-END
