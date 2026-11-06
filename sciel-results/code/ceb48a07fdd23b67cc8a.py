import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
        
    primes = sieve(1000)
    n_primes = len(primes)
    MAX_L = 100000
    coverage = np.zeros(MAX_L + 1, dtype=np.int32)
    
    # Initialize residues
    # Heuristic: Start with random, but maybe bias towards 0?
    # Random is fine, we will optimize.
    residues = [random.randint(0, p - 1) for p in primes]
    
    # Build initial coverage
    for i, p in enumerate(primes):
        r = residues[i]
        coverage[r::p] += 1
        
    # Find initial L
    current_L = 0
    while current_L < MAX_L and coverage[current_L + 1] >= 2:
        current_L += 1
        
    best_L = current_L
    best_residues = residues.copy()
    
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    exp = math.exp
    rand = random.random
    randint = random.randint
    
    # Pre-compute prime indices for faster access? 
    # We need to iterate primes. 
    # To favor p > G, we can sort indices by prime size descending?
    # But G changes. 
    # We can just iterate all primes, but check p > G first logic.
    
    # List of prime indices
    indices = list(range(n_primes))
    
    while time.time() < deadline:
        if current_L >= MAX_L:
            break
            
        G = current_L + 1
        move_made = False
        
        # 1. Try Safe Moves
        # Shuffle indices to avoid bias? Or just iterate.
        # Iterating in random order is better for SA-like behavior, 
        # but for "Safe Move", we want to find *any* safe move quickly.
        # Let's shuffle a subset or all.
        random.shuffle(indices)
        
        for i in indices:
            p = primes[i]
            r_old = residues[i]
            r_new = G % p
            
            if r_old == r_new:
                continue
                
            # Check safety
            safe = True
            if p > G:
                # Only r_old could be in [1, G-1]
                if 1 <= r_old < G:
                    if coverage[r_old] < 3:
                        safe = False
            else:
                # p <= G
                # Check min coverage in [1, G-1] for r_old
                # Slice: coverage[r_old:G:p]
                # Note: if r_old == 0, first index is p.
                # np.min on empty slice raises error, so check len.
                # We can use a helper or just check.
                # To speed up, we can just do the slice. 
                # But this is O(G/p).
                # We can limit this to only try a few small primes?
                # Or just try all, but it's slower.
                # Let's try to optimize: 
                # If G is large, and p is small, this is expensive.
                # We can skip small primes if we have many large primes available.
                # But let's just do it.
                # Optimization: Check if there are any points at all?
                # If r_old == 0:
                #   pts are p, 2p... < G
                # If r_old > 0:
                #   pts are r_old, r_old+p... < G
                
                # We can just do:
                if r_old == 0:
                    start_idx = p
                else:
                    start_idx = r_old
                    
                if start_idx < G:
                    # Get min
                    # Using numpy min is fast in C
                    min_cov = np.min(coverage[start_idx:G:p])
                    if min_cov < 3:
                        safe = False
            
            if safe:
                # Apply move
                coverage[r_old::p] -= 1
                coverage[r_new::p] += 1
                residues[i] = r_new
                
                # Update L
                # L cannot decrease because move was safe.
                # So we just extend.
                while current_L < MAX_L and coverage[current_L + 1] >= 2:
                    current_L += 1
                    
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                
                move_made = True
                break
                
        if not move_made:
            # 2. Kick (Simulated Annealing)
            i = randint(0, n_primes - 1)
            p = primes[i]
            old_r = residues[i]
            new_r = randint(0, p - 1)
            
            if old_r == new_r:
                continue
                
            coverage[old_r::p] -= 1
            coverage[new_r::p] += 1
            residues[i] = new_r
            
            # Find new L
            # This is expensive, but kicks are rare.
            new_L = 0
            while new_L < MAX_L and coverage[new_L + 1] >= 2:
                new_L += 1
            
            delta = new_L - current_L
            if delta >= 0:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                prob = exp(delta / T)
                if rand() < prob:
                    current_L = new_L
                else:
                    # Revert
                    coverage[new_r::p] -= 1
                    coverage[old_r::p] += 1
                    residues[i] = old_r
        
        T *= cooling
        if T < min_T:
            T = min_T

    return best_residues
