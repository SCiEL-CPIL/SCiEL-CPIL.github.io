"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find 168 residues (one per prime <= 1000) maximizing the doubly-covered length L.
    Uses Simulated Annealing with a fast coverage evaluator.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = [False] * len(sieve[i*i::i])
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)  # 168
    
    # Initial residues: use the seed strategy
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    
    def compute_L_fast(residues):
        """Compute the largest L such that every integer 1..L satisfies at least two congruences."""
        max_check = 100000
        # For each prime, mark which numbers in 1..max_check it covers
        # We need count of coverage >= 2 for all numbers 1..L
        
        # Efficient approach: for each x in 1..max_check, count how many primes cover it
        # But that's O(max_check * n_primes) which is too slow
        
        # Better: use the fact that we can check incrementally
        # For each prime p with residue r, it covers numbers r, r+p, r+2p, ... in [1, max_check]
        # We maintain a count array
        
        # Actually, let's use a smarter approach:
        # Initialize coverage count array
        # For each prime, add 1 to all positions it covers
        # Then find the longest prefix where all counts >= 2
        
        # But building this from scratch each time is expensive.
        # Let's use numpy for vectorized operations.
        
        coverage = np.zeros(max_check + 1, dtype=np.int32)
        for p, r in zip(primes, residues):
            if r == 0:
                # Covers p, 2p, 3p, ...
                positions = np.arange(p, max_check + 1, p)
            else:
                # Covers r, r+p, r+2p, ...
                first = r
                if first > max_check:
                    continue
                positions = np.arange(first, max_check + 1, p)
            coverage[positions] += 1
        
        # Find largest L such that coverage[1..L] all >= 2
        # Use cumulative min or just find first position where coverage < 2
        below_two = np.where(coverage[1:max_check+1] < 2)[0]
        if len(below_two) == 0:
            return max_check
        else:
            return int(below_two[0])  # first index (0-based in the slice) where coverage < 2
    
    # Evaluate initial solution
    best_L = compute_L_fast(residues)
    best_residues = residues.copy()
    
    # Simulated Annealing
    current_L = best_L
    current_residues = residues.copy()
    
    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.995
    
    rng = np.random.default_rng(42)
    
    T = T_start
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart with a slightly perturbed version of best
            current_residues = best_residues.copy()
            # Perturb a few random residues
            n_perturb = rng.integers(3, 10)
            idxs = rng.choice(n_primes, size=n_perturb, replace=False)
            for idx in idxs:
                current_residues[idx] = rng.integers(0, primes[idx])
            current_L = compute_L_fast(current_residues)
            no_improve_count = 0
            T = T_start * 0.5
        
        # Pick a random prime to change
        idx = rng.integers(0, n_primes)
        old_val = current_residues[idx]
        new_val = rng.integers(0, primes[idx])
        
        if new_val == old_val:
            continue
        
        current_residues[idx] = new_val
        new_L = compute_L_fast(current_residues)
        
        delta = new_L - current_L
        
        if delta >= 0:
            # Accept
            current_L = new_L
            no_improve_count = 0
            if new_L > best_L:
                best_L = new_L
                best_residues = current_residues.copy()
        else:
            # Accept with probability exp(delta / T)
            if T > 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_L = new_L
                else:
                    # Reject
                    current_residues[idx] = old_val
            
            no_improve_count += 1
        
        T *= cooling_rate
        if T < T_end:
            T = T_end
    
    # Return the best residues found
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
