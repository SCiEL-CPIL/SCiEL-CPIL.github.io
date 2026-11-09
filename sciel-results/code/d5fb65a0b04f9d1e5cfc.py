import numpy as np
import time

def construct():
    start_time = time.time()
    # Use 115s to be safe against the 120s limit
    deadline = start_time + 115.0
    
    # Generate primes <= 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p**0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)
    
    # Initialize residues using the seed heuristic
    # This provides a valid starting point
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)])
    
    # Optimize coverage up to L_MAX
    # Target is >= 1000, so 2000 is a good upper bound for the search space
    L_MAX = 2000
    x_range = np.arange(1, L_MAX + 1)
    
    # Initialize counts
    counts = np.zeros(L_MAX, dtype=np.int32)
    for p, r in zip(primes, residues):
        counts += (x_range % p == r)
    
    # Current score: sum of min(counts, 2)
    # This metric is 2*L if all 1..L are covered at least twice
    current_score = np.sum(np.minimum(counts, 2))
    
    best_score = current_score
    best_residues = residues.copy()
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.99995
    min_T = 0.001
    
    while time.time() < deadline:
        # Pick a random prime to change
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        
        # Pick a new random residue
        new_r = np.random.randint(p)
        
        if new_r == old_r:
            continue
            
        # Update counts efficiently
        old_mask = (x_range % p == old_r)
        new_mask = (x_range % p == new_r)
        
        counts -= old_mask
        counts += new_mask
        
        new_score = np.sum(np.minimum(counts, 2))
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            # Accept the move
            current_score = new_score
            residues[i] = new_r
            
            if current_score > best_score:
                best_score = current_score
                best_residues = residues.copy()
        else:
            # Reject the move, revert counts
            counts += old_mask
            counts -= new_mask
            
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
    return best_residues.tolist()
