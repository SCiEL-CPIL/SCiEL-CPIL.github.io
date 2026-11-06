import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Generate primes <= 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Window size for coverage calculation
    W = 5000
    
    # Weights for fitness function: prioritize early integers
    # We care most about 1..1000
    x_range = np.arange(1, W + 1, dtype=np.float64)
    # Weights decay exponentially or 1/x. 1/x is good for "prefix" style objectives
    weights = 1.0 / x_range
    # Normalize weights to keep fitness values reasonable
    weights /= np.sum(weights)

    def compute_coverage(residues):
        """Compute coverage array from residues"""
        cov = np.zeros(W + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes_arr[i]
            r = residues[i]
            start = r if r > 0 else p
            if start <= W:
                cov[start:W+1:p] += 1
        return cov

    def compute_L(cov):
        """Find the largest L such that cov[1..L] >= 2"""
        # We only care up to W
        mask = cov[1:] < 2
        if not mask.any():
            return W
        first_bad_idx = np.argmax(mask)
        return first_bad_idx

    def calc_fitness(cov):
        """Calculate fitness based on weighted coverage"""
        # min(cov, 2) ensures extra coverage doesn't add infinite value
        # We sum weights * min(cov, 2)
        capped_cov = np.minimum(cov[1:], 2)
        return np.dot(weights, capped_cov)

    def update_cov(cov, p, old_r, new_r):
        """Update coverage array in place"""
        old_start = old_r if old_r > 0 else p
        new_start = new_r if new_r > 0 else p
        
        if old_start <= W:
            cov[old_start:W+1:p] -= 1
        if new_start <= W:
            cov[new_start:W+1:p] += 1

    def find_first_bad(cov):
        """Find the first index x >= 1 where cov[x] < 2"""
        mask = cov[1:] < 2
        if not mask.any():
            return W
        return np.argmax(mask) + 1

    # --- Simulated Annealing ---
    
    # Initialization: Try a few different starts
    best_L_global = 0
    best_residues_global = None
    
    def run_sa(init_residues, time_budget):
        residues = init_residues.copy()
        cov = compute_coverage(residues)
        current_F = calc_fitness(cov)
        current_L = compute_L(cov)
        
        best_F = current_F
        best_residues = residues.copy()
        best_L = current_L
        
        T = 0.5 # Initial temperature
        T_min = 0.01
        cooling = 0.9995 # Slow cooling
        
        steps = 0
        max_steps = 10000000 # Large number, time-limited
        
        while time.time() < time_budget:
            steps += 1
            
            # Generate move
            # 70% Targeted, 30% Random
            if random.random() < 0.7:
                # Targeted: Fix the first bad spot
                x_bad = find_first_bad(cov)
                if x_bad > W:
                    break # Fully covered within window
                
                # Find primes that do NOT cover x_bad
                # x_bad % p == r_p means p covers x_bad
                # We want to change a prime p such that x_bad % p != r_p
                # We pick a random prime and check
                i = random.randint(0, n_primes - 1)
                p = primes_arr[i]
                target_r = x_bad % p
                
                if target_r == residues[i]:
                    # This prime already covers x_bad. 
                    # Changing it might break other things, but doesn't help x_bad directly.
                    # Let's try a random prime that doesn't cover it?
                    # Or just pick a random other prime.
                    # For simplicity, if it already covers, maybe pick a different prime.
                    # But let's just do a random move in this case to avoid complexity
                    pass # Fallback to random logic below if this fails? 
                    # Actually, if it already covers, changing it is risky.
                    # Let's force a random move instead.
                    i = random.randint(0, n_primes - 1)
                    p = primes_arr[i]
                    new_r = random.randint(0, p - 1)
                else:
                    new_r = target_r
            else:
                # Random move
                i = random.randint(0, n_primes - 1)
                p = primes_arr[i]
                new_r = random.randint(0, p - 1)
            
            old_r = residues[i]
            if old_r == new_r:
                continue
            
            # Calculate Delta Fitness
            # We need to update cov to calculate new fitness, then revert if rejected
            # Or calculate delta directly?
            # Delta F = sum(weights[x] * (min(cov[x]+delta, 2) - min(cov[x], 2)))
            # This is complex to do without updating. 
            # Given W=5000 and p>=2, update is ~2500 ops. 
            # 10000 steps/s is feasible.
            
            update_cov(cov, p, old_r, new_r)
            new_F = calc_fitness(cov)
            residues[i] = new_r
            
            delta_F = new_F - current_F
            
            if delta_F > 0 or random.random() < np.exp(delta_F / T):
                # Accept
                current_F = new_F
                new_L = compute_L(cov)
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
            else:
                # Reject: Revert
                update_cov(cov, p, new_r, old_r)
                residues[i] = old_r
            
            T *= cooling
            if T < T_min:
                T = T_min # Keep it above 0 to allow some exploration
            
            # Check time
            if steps % 1000 == 0 and time.time() > time_budget:
                break
                
        return best_L, best_residues

    # Initializations
    # 1. All 1s
    # 2. Random
    # 3. "Greedy" small number cover: r_p = 1 % p
    
    inits = [
        np.array([1] * n_primes, dtype=np.int64),
        np.array([random.randint(0, p-1) for p in primes], dtype=np.int64),
        np.array([(i+1) % p for i, p in enumerate(primes)], dtype=np.int64)
    ]
    
    # Run SA for each init, allocating time
    # Total time ~ 140s
    # Split among inits? Or just run one good one?
    # Let's run the "All 1s" one for most of the time, and others briefly.
    
    # Actually, "All 1s" is a strong start because 1 is covered by all primes.
    # 2 is covered by p=2 (r=0? no r=1 means 1, 3, 5... so 2 is not covered by p=2 if r=1).
    # If r_p = 1, then x=1 is covered by all. x=2 is covered by p=2 (if 2%2==0!=1, no), p=3 (2%3=2!=1, no).
    # So "All 1s" covers 1 very well, but 2, 3, 4... poorly.
    
    # Better init: 
    # For p=2, r=0 or 1.
    # Let's just use the SA to find the structure.
    
    # Strategy: Run SA on "All 1s" for 60% of time.
    # Run SA on "Random" for 20%.
    # Run SA on "Greedy" for 20%.
    
    time_limit = deadline - 5.0
    t0 = time.time()
    
    # Init 1: All 1s
    t1 = t0 + 0.6 * (time_limit - t0)
    L1, R1 = run_sa(inits[0], t1)
    if L1 > best_L_global:
        best_L_global = L1
        best_residues_global = R1
    
    # Init 2: Random
    t2 = t0 + 0.8 * (time_limit - t0)
    L2, R2 = run_sa(inits[1], t2)
    if L2 > best_L_global:
        best_L_global = L2
        best_residues_global = R2
        
    # Init 3: Greedy
    L3, R3 = run_sa(inits[2], time_limit)
    if L3 > best_L_global:
        best_L_global = L3
        best_residues_global = R3

    # Final Polish: Local Search on the best result found
    if best_residues_global is not None:
        residues = best_residues_global.copy()
        cov = compute_coverage(residues)
        current_L = compute_L(cov)
        
        # Simple local search: try to fix the first bad spot
        for _ in range(10000):
            if time.time() > deadline - 1:
                break
            x_bad = find_first_bad(cov)
            if x_bad > W:
                break
            
            # Try all primes that don't cover x_bad
            improved = False
            for i in range(n_primes):
                p = primes_arr[i]
                target_r = x_bad % p
                if target_r == residues[i]:
                    continue
                
                old_r = residues[i]
                update_cov(cov, p, old_r, target_r)
                residues[i] = target_r
                
                new_L = compute_L(cov)
                if new_L > current_L:
                    current_L = new_L
                    improved = True
                    break
                else:
                    # Revert
                    update_cov(cov, p, target_r, old_r)
                    residues[i] = old_r
            
            if not improved:
                # Try random perturbation
                i = random.randint(0, n_primes - 1)
                p = primes_arr[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                if new_r != old_r:
                    update_cov(cov, p, old_r, new_r)
                    residues[i] = new_r
                    new_L = compute_L(cov)
                    if new_L < current_L:
                        # Revert if it got worse significantly? 
                        # For local search, usually we only accept improvements.
                        update_cov(cov, p, new_r, old_r)
                        residues[i] = old_r
                    else:
                        current_L = new_L
                        improved = True
            
            if not improved:
                # Stuck. Break or random jump?
                # Let's break to save time, SA already did the heavy lifting.
                pass

        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues

    if best_residues_global is None:
        return [1] * n_primes
        
    return [int(r) for r in best_residues_global]
