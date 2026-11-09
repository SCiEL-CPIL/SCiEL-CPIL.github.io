"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import math

def _get_primes(n=1000):
    """Return list of primes <= n in increasing order."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _compute_L(counts, max_check=100000):
    """Find largest L such that every integer 1..L has count >= 2."""
    # Vectorized check: find first index where count < 2
    # counts is 1-indexed effectively, but we store 0..max_check
    # We care about 1..max_check
    # Use np.nonzero to find indices where condition fails
    # Condition: counts[1:max_check+1] >= 2
    # We want the first index where this is False.
    
    # To be fast, we can check in chunks or just use a loop if L is small, 
    # but vectorized is better for large L.
    # However, np.argmax on a boolean array gives the first True.
    # We want the first False in (counts >= 2).
    # So we look for the first True in (counts < 2).
    
    # Note: counts[0] is not used for x=1..L logic directly, 
    # but we must ensure indices 1..L are valid.
    
    bad = np.nonzero(counts[1:max_check+1] < 2)[0]
    if len(bad) == 0:
        return max_check
    else:
        # bad[0] is the 0-based index in the slice [1..max_check]
        # So the actual integer is bad[0] + 1
        # But we want the largest L such that 1..L are good.
        # If the first bad one is at integer x, then L = x - 1.
        first_bad_x = bad[0] + 1
        return first_bad_x - 1

def construct():
    primes = _get_primes(1000)
    n_primes = len(primes)
    max_check = 100000
    
    # Initialize residues
    # Strategy: Greedy initialization. 
    # Start with random residues, then do a quick pass to fix immediate failures in [1, 1000]
    residues = [random.randint(0, p - 1) for p in primes]
    
    # Build initial counts using NumPy vectorization
    counts = np.zeros(max_check + 2, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_check:
            counts[start::p] += 1
            
    current_L = _compute_L(counts, max_check)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    steps = 50000 # Adjust based on time limit. 120s is generous.
    # Each step: 1 update (vectorized) + 1 L calc (vectorized).
    # Vectorized update is ~microseconds. L calc is ~milliseconds.
    # 50000 * 1ms = 50s. Safe.
    
    best_L = current_L
    best_residues = residues.copy()
    
    for step in range(steps):
        # Pick a random prime index
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = residues[i]
        
        # Pick a new random residue
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
            
        # Update counts
        # Remove old
        if old_r == 0:
            start_old = p
        else:
            start_old = old_r
        if start_old <= max_check:
            counts[start_old::p] -= 1
            
        # Add new
        if new_r == 0:
            start_new = p
        else:
            start_new = new_r
        if start_new <= max_check:
            counts[start_new::p] += 1
            
        new_L = _compute_L(counts, max_check)
        
        # Acceptance criterion
        delta = new_L - current_L
        if delta > 0 or random.random() < math.exp(delta / T):
            current_L = new_L
            residues[i] = new_r
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        else:
            # Revert
            if start_new <= max_check:
                counts[start_new::p] -= 1
            if start_old <= max_check:
                counts[start_old::p] += 1
            residues[i] = old_r # Ensure consistency, though it was already old_r
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_residues
# EVOLVE-BLOCK-END
