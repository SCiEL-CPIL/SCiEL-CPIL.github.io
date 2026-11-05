import time
import numpy as np
import random

N = 5000
DEADLINE = 325  # safety margin before 330s kill

def _primes(limit):
    """Return list of primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def _is_admissible(points, primes):
    """Check admissibility: for every prime p <= len(points), some residue class mod p is missed."""
    n = len(points)
    if n == 0:
        return True
    pts = np.array(points)
    for p in primes:
        if p > n:
            break
        residues = np.unique(pts % p)
        if len(residues) == p:
            return False
    return True

def _greedy_pass(primes, seed, deadline):
    """
    Single pass of greedy construction with randomized tie-breaking.
    """
    start = time.time()
    present = np.ones(N + 1, dtype=bool)
    indices = np.arange(N + 1)
    rng = np.random.default_rng(seed)
    
    for p in primes:
        if p > N:
            break
        if time.time() - start > deadline:
            break
            
        # Only check if p is relevant to current size
        current_size = np.sum(present)
        if p > current_size:
            break

        mods = indices[present] % p
        counts = np.bincount(mods, minlength=p)
        
        if counts.min() == 0:
            continue

        # Find all residue classes with minimum count
        min_val = counts.min()
        candidates = np.where(counts == min_val)[0]
        
        # Randomly select one candidate to break ties
        target = candidates[rng.integers(len(candidates))]
        
        # Remove elements in this residue class
        mask = (indices % p) == target
        present[mask] = False

    return indices[present].tolist()

def construct():
    start = time.time()
    deadline = start + DEADLINE
    primes = _primes(N)
    
    best_result = []
    best_size = -1
    
    # Run multiple passes with different seeds to explore the solution space
    # Each pass is very fast, so we can run many.
    num_passes = 1000 # Adjust based on time
    
    for i in range(num_passes):
        elapsed = time.time() - start
        if elapsed > deadline - 2: # 2s safety margin
            break
            
        # Use a deterministic seed for reproducibility in case of timeout, 
        # but vary it to explore different paths.
        # We can also mix deterministic and random strategies.
        seed = i
        result = _greedy_pass(primes, seed, deadline - (time.time() - start))
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
            
    # Verify the best result
    if not _is_admissible(best_result, primes):
        # If invalid (shouldn't happen with greedy, but just in case), force fix
        # This is a fallback
        result_set = set(best_result)
        for p in primes:
            if p > len(result_set):
                break
            residues = set(v % p for v in result_set)
            if len(residues) == p:
                class_members = {}
                for v in result_set:
                    class_members.setdefault(v % p, []).append(v)
                best_class = min(class_members, key=lambda c: len(class_members[c]))
                result_set -= set(class_members[best_class])
        best_result = sorted(result_set)

    return sorted(set(best_result))

# EVOLVE-BLOCK-START
# Entry point is construct(); logic above.
# EVOLVE-BLOCK-END
