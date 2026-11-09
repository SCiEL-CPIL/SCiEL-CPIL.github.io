"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from math import isqrt

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

def _try_strategy(primes, sieve_threshold, use_least_populated_for_small=True):
    """Try a specific strategy and return the resulting set size and mask."""
    S_mask = np.ones(N + 1, dtype=bool)
    
    # Sieve primes up to threshold
    small_primes = primes[primes <= sieve_threshold]
    for p in small_primes:
        if use_least_populated_for_small:
            # Find least populated residue among current S
            S_indices = np.where(S_mask)[0]
            if len(S_indices) == 0:
                return 0, S_mask
            residues = S_indices % p
            counts = np.bincount(residues, minlength=p)
            best_r = np.argmin(counts)
            to_remove = S_indices[residues == best_r]
            S_mask[to_remove] = False
        else:
            # Remove residue 0
            multiples = np.arange(0, N + 1, p)
            S_mask[multiples] = False
    
    remaining_primes = primes[primes > sieve_threshold]
    
    changed = True
    while changed:
        current_size = np.count_nonzero(S_mask)
        if current_size == 0:
            break
        
        S_indices = np.where(S_mask)[0]
        candidates = remaining_primes[remaining_primes <= current_size]
        
        bad_p = None
        min_removal_count = float('inf')
        best_r = -1
        best_p = -1
        
        for p in candidates:
            residues = S_indices % p
            unique_res = np.unique(residues)
            if len(unique_res) == p:
                counts = np.bincount(residues, minlength=p)
                min_count = np.min(counts)
                if min_count < min_removal_count:
                    min_removal_count = min_count
                    best_r = np.argmin(counts)
                    best_p = p
                    bad_p = p
        
        if bad_p is not None:
            to_remove_mask = (S_indices % best_p) == best_r
            values_to_remove = S_indices[to_remove_mask]
            S_mask[values_to_remove] = False
            changed = True
        else:
            changed = False
    
    return np.count_nonzero(S_mask), S_mask

def construct():
    start_time = time.time()
    deadline = 290.0  # Safety margin
    
    primes = _primes(N)
    
    best_size = 0
    best_mask = None
    
    # Try different strategies with different sieve thresholds
    # Thresholds to try: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    # Plus some larger ones
    thresholds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    
    for threshold in thresholds:
        if time.time() - start_time > deadline:
            break
        
        # Try with least populated for small primes
        size, mask = _try_strategy(primes, threshold, use_least_populated_for_small=True)
        if size > best_size:
            best_size = size
            best_mask = mask.copy()
        
        # Try with residue 0 for small primes
        size, mask = _try_strategy(primes, threshold, use_least_populated_for_small=False)
        if size > best_size:
            best_size = size
            best_mask = mask.copy()
    
    if best_mask is None:
        # Fallback: just return empty or all
        return list(range(N + 1))
    
    return np.where(best_mask)[0].tolist()

# EVOLVE-BLOCK-END
