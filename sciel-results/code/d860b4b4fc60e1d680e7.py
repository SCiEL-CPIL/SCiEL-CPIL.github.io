import numpy as np
import time
import random

N = 5000
RANGE = N + 1

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_size = 0
    best_points = []
    
    # Precompute x indices
    x_indices = np.arange(RANGE)
    
    # Run multiple randomized greedy attempts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        active = np.ones(RANGE, dtype=bool)
        current_size = RANGE
        
        # Use ascending order to ensure small primes (like 2) are always processed
        # and not skipped due to size reduction.
        primes_copy = PRIMES
        
        for i, p in enumerate(primes_copy):
            if p > current_size:
                break
            
            # Count active elements in each residue class mod p
            residues = x_indices[active] % p
            if len(residues) == 0:
                break
            
            # Count occurrences of each residue
            counts = np.bincount(residues, minlength=p)
            
            min_count = np.min(counts)
            
            if min_count == 0:
                # Already admissible for this prime
                continue
            
            # Find all classes with min count
            candidate_classes = np.where(counts == min_count)[0]
            
            if len(candidate_classes) == 1:
                r = candidate_classes[0]
            else:
                # Lookahead 1: Pick the residue that minimizes the min-count for the next prime
                best_r = None
                best_score = float('inf')
                
                # Get next prime
                if i + 1 < len(primes_copy):
                    next_p = primes_copy[i+1]
                    if next_p <= current_size:
                        for r_cand in candidate_classes:
                            # Simulate removal of r_cand
                            # temp_active = active & (x_indices % p != r_cand)
                            # We want to minimize min(counts_next)
                            
                            # Efficient simulation:
                            # We need counts of residues mod next_p for elements in active 
                            # that are NOT in class r_cand mod p.
                            
                            # active_indices = np.where(active)[0] # Too slow to do every time?
                            # Let's use the mask approach.
                            mask_remove = (x_indices % p == r_cand)
                            temp_active = active & ~mask_remove
                            
                            residues_next = x_indices[temp_active] % next_p
                            counts_next = np.bincount(residues_next, minlength=next_p)
                            score = np.min(counts_next)
                            
                            if score < best_score:
                                best_score = score
                                best_r = r_cand
                            
                            # Optional: Early break if score is 0?
                            # If score is 0, it's as good as it gets.
                            if best_score == 0:
                                break
                
                # If no lookahead available or no best_r found, fallback to random
                if best_r is None:
                    r = candidate_classes[random.randint(0, len(candidate_classes) - 1)]
                else:
                    r = best_r
            
            # Update active and size
            # Remove elements in class r mod p
            # We know min_count elements are removed
            current_size -= min_count
            active = active & (x_indices % p != r)
            
            if current_size == 0:
                break
        
        # Extract points
        points = np.where(active)[0].tolist()
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points

    return best_points
