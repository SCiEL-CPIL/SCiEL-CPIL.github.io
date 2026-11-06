import time
import numpy as np
import random

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(counts, max_L):
    # Find first index where count < 2
    # counts is 1-indexed effectively for our purposes (index 0 is unused or 0)
    # We check 1..max_L
    below = np.where(counts[1:max_L + 1] < 2)[0]
    if len(below) == 0:
        return max_L
    return int(below[0])

def _build_counts(residues, primes, L_max):
    counts = np.zeros(L_max + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = int(residues[i])
        if r == 0:
            if p <= L_max:
                counts[p::p] += 1
        else:
            if r <= L_max:
                counts[r::p] += 1
    return counts

def _compute_energy(counts, L_max):
    # Energy = sum_{x=1}^{L_max} (1/x) * max(0, 2 - counts[x])^2
    # Vectorized
    c = counts[1:L_max + 1].astype(np.float64)
    deficit = np.maximum(0, 2 - c)
    weights = 1.0 / np.arange(1, L_max + 1, dtype=np.float64)
    return np.sum(weights * deficit * deficit)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    L_search = 20000  # Search in this range
    
    best_L = 0
    best_residues = None
    
    # Run multiple SA restarts
    # Time budget per restart
    num_restarts = 3
    time_per_restart = 35.0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        random.seed(12345 + restart)
        np.random.seed(12345 + restart)
        
        # Initialize residues
        residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
        
        # Initial counts
        counts = _build_counts(residues, primes, L_search)
        
        # Initial Energy
        E_old = _compute_energy(counts, L_search)
        
        # Initial L
        L_current = _compute_L(counts, L_search)
        
        if L_current > best_L:
            best_L = L_current
            best_residues = residues.copy()
            
        # Simulated Annealing
        T = 5.0  # Initial temperature
        T_min = 0.01
        cooling_rate = 0.9995
        
        steps = 0
        restart_deadline = start_time + (restart + 1) * time_per_restart
        if restart_deadline > deadline - 5.0:
            restart_deadline = deadline - 5.0
            
        while time.time() < restart_deadline:
            # Pick random prime index
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = int(residues[i])
            new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue
                
            # Update counts
            # Subtract old
            if old_r == 0:
                if p <= L_search:
                    counts[p::p] -= 1
            else:
                if old_r <= L_search:
                    counts[old_r::p] -= 1
            
            # Add new
            if new_r == 0:
                if p <= L_search:
                    counts[p::p] += 1
            else:
                if new_r <= L_search:
                    counts[new_r::p] += 1
            
            # New Energy
            E_new = _compute_energy(counts, L_search)
            
            # Check L
            L_new = _compute_L(counts, L_search)
            
            # Acceptance
            if E_new < E_old or (T > 0 and random.random() < np.exp(-(E_new - E_old) / T)):
                # Accept
                E_old = E_new
                residues[i] = new_r
                L_current = L_new
                
                if L_new > best_L:
                    best_L = L_new
                    best_residues = residues.copy()
            else:
                # Revert
                if new_r == 0:
                    if p <= L_search:
                        counts[p::p] -= 1
                else:
                    if new_r <= L_search:
                        counts[new_r::p] -= 1
                        
                if old_r == 0:
                    if p <= L_search:
                        counts[p::p] += 1
                else:
                    if old_r <= L_search:
                        counts[old_r::p] += 1
            
            # Cool
            T *= cooling_rate
            if T < T_min:
                T = T_min
            steps += 1

    # Final verification with larger L
    if best_residues is not None:
        L_verify = 50000
        counts_v = _build_counts(best_residues, primes, L_verify)
        final_L = _compute_L(counts_v, L_verify)
        
        if final_L >= L_verify and time.time() < deadline - 1.0:
            L_verify2 = 100000
            counts_v2 = _build_counts(best_residues, primes, L_verify2)
            final_L = _compute_L(counts_v2, L_verify2)
            
        return [int(r) for r in best_residues]
    else:
        # Fallback if something went wrong
        return [0] * n_primes
