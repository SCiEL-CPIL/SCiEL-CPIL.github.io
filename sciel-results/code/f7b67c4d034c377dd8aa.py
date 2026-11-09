import sys
import time
import random

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    primes.sort() # Increasing order for output and indexing
    
    n_primes = len(primes)
    # We estimate L_max. If we can cover 1..1000, maybe we can go higher.
    # Sum of 1/p approx log(log(1000)) ~ 2. But we need 2 covers.
    # Let's set a generous bound, say 2000 or 3000.
    # The score is capped at 10^5, but realistically L won't be huge.
    # Let's try L_max = 5000 to be safe, but we only care about prefix.
    L_max = 5000
    
    # Initial residues: Greedy
    residues = [0] * n_primes
    cov = [0] * (L_max + 1)
    
    # Greedy initialization
    # Sort primes by descending size to place larger moduli first? 
    # Or ascending? 
    # Large primes have sparse coverage. Small primes have dense.
    # It's often better to assign small primes first to establish a base coverage.
    # Let's stick to the order in the list (increasing).
    
    for i, p in enumerate(primes):
        best_r = 0
        best_gain = -1
        for r in range(p):
            gain = 0
            start = r if r > 0 else p
            for x in range(start, L_max + 1, p):
                if cov[x] < 2:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        residues[i] = best_r
        start = best_r if best_r > 0 else p
        for x in range(start, L_max + 1, p):
            cov[x] += 1
            
    # Calculate initial L
    def get_L(cov):
        L = 0
        for x in range(1, L_max + 1):
            if cov[x] >= 2:
                L = x
            else:
                break
        return L
        
    current_L = get_L(cov)
    best_L = current_L
    best_residues = residues.copy()
    
    # Local Search / Simulated Annealing
    # Time limit: 110 seconds to be safe.
    start_time = time.time()
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # To speed up, we can precompute the positions for each prime/residue?
    # No, updating cov is O(L/p).
    
    while time.time() - start_time < 110:
        # Pick a random prime
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = residues[i]
        
        # Pick a new random residue
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
            
        # Update coverage
        # Decrement old
        start_old = old_r if old_r > 0 else p
        for x in range(start_old, L_max + 1, p):
            cov[x] -= 1
            
        # Increment new
        start_new = new_r if new_r > 0 else p
        for x in range(start_new, L_max + 1, p):
            cov[x] += 1
            
        # Calculate new L
        new_L = get_L(cov)
        
        # Acceptance criterion
        if new_L > current_L or (T > T_min and random.random() < pow(2.718, (new_L - current_L) / T)):
            current_L = new_L
            residues[i] = new_r
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        else:
            # Revert
            # Decrement new
            for x in range(start_new, L_max + 1, p):
                cov[x] -= 1
            # Increment old
            for x in range(start_old, L_max + 1, p):
                cov[x] += 1
            # current_L remains unchanged, residues unchanged
        
        T *= cooling
        
    return best_residues
