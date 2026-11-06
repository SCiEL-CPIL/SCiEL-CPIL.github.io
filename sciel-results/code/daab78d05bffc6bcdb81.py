"""Narrow admissible tuple: Greedy maximum independent set on conflict graph (optimized with NumPy)."""
import numpy as np
import math

N = 5000

def _primes(limit):
    """Return list of primes up to limit (inclusive)."""
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(math.isqrt(limit)) + 1):
        if sieve[p]:
            sieve[p*p::p] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    n = N
    # active[i] is 1 if i is in the tuple, 0 otherwise
    active = np.ones(n + 1, dtype=np.uint8)
    current_size = n + 1
    
    # Precompute primes up to n+1
    all_primes = _primes(n + 1)
    
    # We will iterate until no violating prime is found
    while True:
        violating_prime = None
        min_residue = -1
        min_count = float('inf')
        
        # Check primes in increasing order
        for p in all_primes:
            if p > current_size:
                break
            
            # Get indices of active elements
            # Instead of filtering every time, we can use the active mask
            # But np.where is efficient.
            active_indices = np.where(active == 1)[0]
            
            if len(active_indices) == 0:
                return []
                
            # Count residues
            # active_indices % p
            # np.bincount is fast
            counts = np.bincount(active_indices % p, minlength=p)
            
            # Check if any residue is 0
            if np.any(counts == 0):
                # This prime is satisfied (some residue is missing)
                continue
            else:
                # All residues are covered. This is a violation.
                # We found the smallest violating prime.
                violating_prime = p
                # Find min residue
                min_count = np.min(counts)
                min_residue = np.argmin(counts)
                break
        
        if violating_prime is None:
            # Admissible
            result = np.where(active == 1)[0]
            return result.tolist()
        
        # Remove all elements with residue min_residue mod violating_prime
        p = violating_prime
        r = min_residue
        
        # Create mask for elements to remove
        # i % p == r
        # We can do this on the range array
        # But we only care about active ones.
        # active_indices is already computed in the loop above if we stored it, 
        # but it goes out of scope. Let's recompute or use a global range.
        
        # Efficient removal:
        # indices = np.arange(n + 1)
        # remove_mask = (indices % p == r) & (active == 1)
        # active[remove_mask] = 0
        
        # Even faster:
        # We know which indices are active.
        # Let's just use the range for simplicity, N=5000 is small.
        indices = np.arange(n + 1)
        remove_mask = (indices % p == r) & (active == 1)
        active[remove_mask] = 0
        current_size -= np.sum(remove_mask)
