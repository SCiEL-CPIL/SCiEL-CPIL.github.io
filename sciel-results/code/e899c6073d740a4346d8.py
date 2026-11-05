"""Vectorized incremental hill-climbing solver for the covering problem."""
import time
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _count_covered(residues, primes, max_L):
    """
    Count how many integers in 1..max_L are covered at least twice.
    Returns the largest L such that every integer 1..L is covered at least twice.
    """
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        pos = np.arange(start, max_L + 1, p)
        if len(pos) > 0:
            counts[pos] += 1
    
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_L
    else:
        return below_two[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # Initial solution: simple heuristic
    initial_residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_residues = initial_residues[:]
    best_L = _count_covered(best_residues, primes, 100000)
    
    # Precompute coverage counts for the current solution
    counts = np.zeros(100001, dtype=np.int32)
    for r, p in zip(best_residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start <= 100000:
            pos = np.arange(start, 100001, p)
            if len(pos) > 0:
                counts[pos] += 1
    
    # Vectorized incremental hill climbing
    # For each prime i, we want to find the residue r that maximizes the new L
    # when we change residue of prime i from old_r to new_r.
    # 
    # Key insight: When we change residue for prime i, only the coverage of prime i changes.
    # Let old_mask be the positions covered by prime i with old_r.
    # Let new_mask be the positions covered by prime i with new_r.
    # The new counts = counts - old_mask + new_mask.
    # We want to maximize the largest L such that min(new_counts[1..L]) >= 2.
    #
    # For each candidate new_r, we can compute new_counts = counts - old_mask + new_mask
    # and find the new L. Doing this for all p candidates at once is expensive if we do it naively.
    #
    # Alternative: For each prime i, precompute the positions for each possible residue.
    # Then for each candidate residue, compute the delta in the minimum coverage.
    
    improved = True
    iteration = 0
    while improved and (time.time() - start_time) < deadline:
        improved = False
        iteration += 1
        for i in range(n_primes):
            if (time.time() - start_time) > deadline - 5:
                break
            p = primes[i]
            current_r = best_residues[i]
            
            # Get positions for current residue
            if current_r == 0:
                old_start = p
            else:
                old_start = current_r
            if old_start <= 100000:
                old_pos = np.arange(old_start, 100001, p)
            else:
                old_pos = np.array([], dtype=np.int32)
            
            # Try all residues for this prime (or a subset if p is large)
            # For small primes, try all. For large primes, sample.
            if p <= 200:
                candidates = list(range(p))
            else:
                # Sample a subset of residues
                np.random.seed(42 + i + iteration)
                candidates = list(np.random.choice(p, size=min(p, 100), replace=False))
            
            if current_r in candidates:
                candidates.remove(current_r)
            
            best_new_r = current_r
            best_new_L = best_L
            
            # For each candidate, compute the new L efficiently
            # We need to compute: new_counts = counts - old_mask + new_mask
            # and find the largest L where min(new_counts[1..L]) >= 2
            
            # Precompute old_mask as a boolean array
            old_mask = np.zeros(100001, dtype=np.int8)
            if len(old_pos) > 0:
                old_mask[old_pos] = 1
            
            for new_r in candidates:
                if (time.time() - start_time) > deadline - 5:
                    break
                
                # Compute new positions
                if new_r == 0:
                    new_start = p
                else:
                    new_start = new_r
                
                if new_start > 100000:
                    # No coverage, so new_counts = counts - old_mask
                    new_counts = counts - old_mask
                else:
                    new_pos = np.arange(new_start, 100001, p)
                    new_mask = np.zeros(100001, dtype=np.int8)
                    if len(new_pos) > 0:
                        new_mask[new_pos] = 1
                    new_counts = counts - old_mask + new_mask
                
                # Find the largest L where min(new_counts[1..L]) >= 2
                below_two = np.where(new_counts[1:] < 2)[0]
                if len(below_two) == 0:
                    new_L = 100000
                else:
                    new_L = below_two[0]
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_new_r = new_r
            
            if best_new_r != current_r:
                # Update the solution
                best_residues[i] = best_new_r
                best_L = best_new_L
                improved = True
                
                # Update the counts array
                # Remove old coverage, add new coverage
                if len(old_pos) > 0:
                    counts[old_pos] -= 1
                
                if best_new_r == 0:
                    new_start = p
                else:
                    new_start = best_new_r
                if new_start <= 100000:
                    new_pos = np.arange(new_start, 100001, p)
                    if len(new_pos) > 0:
                        counts[new_pos] += 1
    
    return best_residues
