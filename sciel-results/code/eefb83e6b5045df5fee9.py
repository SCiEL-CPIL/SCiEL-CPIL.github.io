"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0: sieve[0] = False
    if limit >= 1: sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _greedy_construct(random_seed=0):
    """
    Greedy construction with random tie-breaking.
    """
    rng = random.Random(random_seed)
    # Use numpy array for S for faster filtering? 
    # S = np.arange(N + 1)
    # Filtering: S = S[S % p != r] is vectorized and fast.
    
    S = np.arange(N + 1)
    primes = _primes(N)
    
    while True:
        L = len(S)
        if L == 0:
            break
            
        # Find violating primes
        best_p = None
        best_min_size = float('inf')
        best_counts = None
        
        for p in primes:
            if p > L:
                break
            
            # Vectorized counting
            residues = S % p
            counts = np.bincount(residues, minlength=p)
            
            min_c = np.min(counts)
            if min_c > 0:
                # This prime is violating
                if min_c < best_min_size:
                    best_min_size = min_c
                    best_p = p
                    best_counts = counts
        
        if best_p is None:
            break  # S is admissible
        
        # Select residue to remove
        min_val = np.min(best_counts)
        candidates = np.where(best_counts == min_val)[0]
        
        if len(candidates) > 1:
            residue_to_remove = rng.choice(candidates)
        else:
            residue_to_remove = candidates[0]
            
        # Remove
        S = S[S % best_p != residue_to_remove]
        
        if len(S) == L:
            break
            
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 290s safety margin
    
    best_S = np.array([])
    best_len = -1
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Randomize tie-breaking
        S = _greedy_construct(random_seed=attempts)
        
        if len(S) > best_len:
            best_len = len(S)
            best_S = S
            
        # Optional: Check if we hit a "perfect" score? 
        # Just keep running until time is up.
        
    return best_S.tolist()
# EVOLVE-BLOCK-END
