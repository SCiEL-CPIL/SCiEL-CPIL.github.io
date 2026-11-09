import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    num_primes = len(primes)
    p_arr = np.array(primes, dtype=np.int32)
    
    N = 1000
    x = np.arange(1, N + 1, dtype=np.int32)
    
    # Precompute masks for each prime and residue
    # mask[i, r] = boolean array of length N where True if x % p_i == r
    masks = np.zeros((num_primes, 0), dtype=bool)  # placeholder
    # Actually, we can't precompute all masks (too large). Instead, compute on the fly.
    
    def compute_counts(residues):
        """Compute coverage counts for all x in 1..N."""
        counts = np.zeros(N, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            counts += (x % p == r)
        return counts
    
    def compute_L(counts):
        """Compute L: largest L such that all x in 1..L have count >= 2."""
        # Find first index where count < 2
        fail_idx = np.argmax(counts < 2)
        if counts[fail_idx] >= 2:
            return N
        return fail_idx
    
    # Initial residues: use a simple heuristic
    # For each prime p, set r_p to cover a specific target
    # Let's start with random residues
    residues = np.array([random.randint(0, p-1) for p in primes], dtype=np.int32)
    
    counts = compute_counts(residues)
    current_L = compute_L(counts)
    
    # Simulated annealing
    T = 10.0  # initial temperature
    T_min = 0.01
    cooling = 0.999
    
    best_L = current_L
    best_residues = residues.copy()
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Pick a random prime to modify
        i = random.randint(0, num_primes - 1)
        p = primes[i]
        
        # Pick a new random residue
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        # Compute delta in counts
        # Remove coverage from old_r, add coverage from new_r
        old_mask = (x % p == old_r)
        new_mask = (x % p == new_r)
        
        # Update counts temporarily
        counts -= old_mask
        counts += new_mask
        
        new_L = compute_L(counts)
        
        # Accept or reject
        if new_L > current_L:
            # Accept (improvement)
            current_L = new_L
            residues[i] = new_r
            no_improve_count = 0
            
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            # Deterioration: accept with probability exp((new_L - current_L) / T)
            delta = new_L - current_L
            prob = np.exp(delta / T)
            if random.random() < prob:
                # Accept
                current_L = new_L
                residues[i] = new_r
                no_improve_count += 1
            else:
                # Reject: revert counts
                counts -= new_mask
                counts += old_mask
                no_improve_count += 1
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # If stuck for too long, restart with current best
        if no_improve_count > max_no_improve:
            residues = best_residues.copy()
            counts = compute_counts(residues)
            current_L = compute_L(counts)
            T = 5.0
            no_improve_count = 0
    
    # Return the best residues found
    return [int(r) for r in best_residues]
