"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def _get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]

def _evaluate(residues, primes, max_check=100000):
    """
    Returns the largest L such that every integer 1..L is covered by at least 2 congruences.
    Uses numpy for fast vectorized counting.
    """
    n_primes = len(primes)
    # We check up to max_check
    # Create an array of counts for integers 1..max_check
    # Instead of iterating over primes and adding, we can use a clever trick or just loop.
    # Looping over 168 primes and updating a numpy array of size 100000 is fast enough.
    
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    for i, p in enumerate(primes):
        r = residues[i]
        # Find indices in 1..max_check that satisfy x % p == r
        # x = r, r+p, r+2p, ...
        # First term: if r == 0, first is p (since x>=1). If r > 0, first is r.
        # Actually, x % p == r.
        # If r == 0, x is multiple of p.
        # If r > 0, x = r + k*p.
        
        # Vectorized approach:
        # Generate all numbers in range [1, max_check] congruent to r mod p
        # This might create large arrays if p is small.
        # Better: use slicing on the counts array? No, indices are not contiguous.
        
        # Let's just loop for small p? Or use numpy arange?
        # arange is fast.
        
        if r == 0:
            start = p
        else:
            start = r
            
        if start <= max_check:
            idx = np.arange(start, max_check + 1, p)
            counts[idx] += 1
        # If r == 0 and p > max_check, no updates.
        # If r > 0 and r > max_check, no updates.
        
    # Find the largest L such that counts[1:L+1] are all >= 2
    # We can find the first index where count < 2
    # If no such index in 1..max_check, return max_check
    
    # Create a boolean mask: count >= 2
    mask = counts[1:] >= 2
    if not mask.all():
        # Find first False
        first_false = np.argmax(~mask)
        return first_false
    else:
        return max_check

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    primes = _get_primes(1000)
    n = len(primes)
    
    # Initial heuristic: similar to seed but maybe better
    # Seed: r_i = ceil(i/2) % p_i
    # This covers integers 1..84 twice.
    # Let's start with the seed solution.
    best_residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_score = _evaluate(best_residues, primes)
    
    # Local search
    # Randomly perturb one or more residues and accept if score improves.
    # To speed up evaluation, we can limit the check range initially, then expand.
    # But _evaluate is vectorized and fast enough for 100k.
    
    rng = np.random.default_rng(42)
    
    # We can try to improve by focusing on the "bottleneck" integer.
    # The integer that breaks the chain is the one with count < 2.
    # If we can increase its count, the score might jump.
    
    # Simple hill climbing
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Try a random restart or larger perturbation
            # Or just continue, maybe increase max_no_improve
            pass
            
        # Pick a random prime to perturb
        i = rng.integers(0, n)
        p = primes[i]
        
        # Try a random new residue
        new_r = rng.integers(0, p)
        if new_r == best_residues[i]:
            continue
            
        old_r = best_residues[i]
        best_residues[i] = new_r
        
        # Evaluate
        # Optimization: only need to check up to current best_score + some margin?
        # No, because increasing count at a specific point might not extend the prefix,
        # but it might allow a later extension if the bottleneck is fixed.
        # Actually, if the current bottleneck is at L, and we fix it, the new bottleneck
        # could be at L+1 or later. So we must check further.
        # However, if we don't fix the bottleneck, the score stays the same.
        
        new_score = _evaluate(best_residues, primes)
        
        if new_score > best_score:
            best_score = new_score
            no_improve_count = 0
        else:
            # Revert if not improved? Or keep if equal?
            # Hill climbing usually reverts if not improved.
            # But sometimes moving to an equal score plateau can help escape local minima.
            # Let's revert to keep it simple hill climbing first.
            best_residues[i] = old_r
            no_improve_count += 1
            
        # Check deadline
        if time.time() > deadline - 5.0:
            break
            
    return best_residues

# EVOLVE-BLOCK-END
