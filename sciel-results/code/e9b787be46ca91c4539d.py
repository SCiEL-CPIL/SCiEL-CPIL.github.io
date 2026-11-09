import numpy as np
import time

def _get_primes(n=1000):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _compute_L(counts, max_check=100000):
    # Find first index in 1..max_check where count < 2
    # counts is 0-indexed array, but we treat counts[x] as count for integer x
    # We care about x in 1..max_check
    # bad[x] is True if counts[x] < 2
    bad = counts[1:max_check+1] < 2
    if bad.any():
        # argmax returns index of first True
        # If bad[0] is True, x=1 is bad, L=0
        # If bad[1] is True, x=2 is bad, L=1
        # So L = index of first bad element
        return np.argmax(bad)
    else:
        return max_check

def construct():
    primes = _get_primes(1000)
    n_primes = len(primes)
    max_check = 100000
    
    # Initialize with seed-like construction
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=int)
    
    counts = np.zeros(max_check + 2, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        counts[start::p] += 1
        
    current_L = _compute_L(counts, max_check)
    best_L = current_L
    best_residues = residues.copy()
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    alpha = 0.9995
    
    start_time = time.time()
    time_limit = 110.0
    
    while time.time() - start_time < time_limit:
        # Pick random prime
        idx = np.random.randint(n_primes)
        p = primes[idx]
        old_r = residues[idx]
        
        # Pick random new residue
        new_r = np.random.randint(p)
        if new_r == old_r:
            continue
            
        # Apply change
        if old_r == 0:
            start_old = p
        else:
            start_old = old_r
        counts[start_old::p] -= 1
        
        if new_r == 0:
            start_new = p
        else:
            start_new = new_r
        counts[start_new::p] += 1
        
        # Compute new L
        new_L = _compute_L(counts, max_check)
        
        # Acceptance
        delta = new_L - current_L
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_L = new_L
            residues[idx] = new_r
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        else:
            # Revert
            counts[start_new::p] -= 1
            counts[start_old::p] += 1
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    return list(best_residues)
