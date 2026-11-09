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
    best_size = 0
    best_mask = None
    
    # Number of restarts
    # Each run is O(N * num_primes * log N) roughly, or O(N * num_primes) with bincount.
    # N=5000, primes up to 5000 is ~669.
    # 50 restarts should be fast.
    num_restarts = 100
    
    for _ in range(num_restarts):
        mask = np.ones(N + 1, dtype=bool)
        
        for p in primes:
            current_size = np.sum(mask)
            
            # If p > current_size, no constraint from p or larger primes?
            # Condition: for every prime q <= len(tuple), some residue is missed.
            # If p > current_size, then for this prime p, p > size.
            # The condition applies to primes <= size.
            # So if p > size, we don't need to satisfy p.
            # And for any future prime q > p, q > p > size, so they don't apply either.
            # So we can break.
            if p > current_size:
                break
            
            # Get active indices
            active_indices = np.where(mask)[0]
            
            # Count residues
            residues = active_indices % p
            counts = np.bincount(residues, minlength=p)
            
            # Check if any residue class is already empty
            if counts.min() == 0:
                # Already admissible for this prime
                continue
            
            # Find residue classes with the minimum count
            min_count = counts.min()
            candidates = np.where(counts == min_count)[0]
            
            # Randomly pick one candidate to remove
            # This is the randomization step
            target_residue = candidates[random.randint(0, len(candidates) - 1)]
            
            # Identify elements to remove
            indices_to_remove = active_indices[residues == target_residue]
            
            # Create new mask
            mask[indices_to_remove] = False
        
        # Update best
        final_size = np.sum(mask)
        if final_size > best_size:
            best_size = final_size
            best_mask = mask.copy()
            
    if best_mask is None:
        return list(range(N + 1))
        
    return list(np.where(best_mask)[0])
# EVOLVE-BLOCK-END
